"""Prompts for codebase-adaptation memory co-training.

Mirrors the offline clbench ``icl_summary`` (replace) system so miles' ACT/WRITE
prompts match the official scaffold
(continual-learning-bench/src/systems/icl_summary/system.py):

  - ACT: **no** system message; the running memory is injected on the first turn
    of each issue as ``[Memory of what you've learned so far]``; command output
    is fed back as a separate ``FEEDBACK:`` user message (done by the caller, as
    the offline system does via ``observe``); the JSON schema is **not** inlined
    into the prompt — structure is enforced by sglang json_schema constrained
    decoding, mirroring the offline ``response_format`` path. The task's own
    ``_SYSTEM_TEMPLATE`` (thought/command format spec) already rides inside
    ``query.prompt``, so no extra format instruction is needed.
  - WRITE: ``MEMORY_SYSTEM`` + the finished issue's raw turns + a final
    instruction that rewrites the memory into the two-section format
    (``### Repository Knowledge`` / ``### Lessons & Pitfalls``).

The wording is inherited verbatim from the offline system (whose class is named
``icl_summary``), but here we call it *memory* throughout — it is the running,
co-evolved memory store, not a one-shot summary.
"""

from __future__ import annotations

import os


# ── WRITE (memory rewrite) — text verbatim from the offline icl_summary system ──
MEMORY_SYSTEM = (
    "You are a software engineer solving a sequence of coding tasks in shared "
    "repositories. You keep a running, structured MEMORY of what you have "
    "learned about these repositories and how to work in them effectively. At "
    "the end of each task you are given your PREVIOUS memory plus the task you "
    "just worked on, and you output the UPDATED memory — carrying forward what "
    "you already confirmed and folding in the new task's evidence — for use in "
    "future tasks."
)

MEMORY_INSTRUCTION = (
    "The task above is over. Update your running memory in EXACTLY these two "
    "sections:\n\n"
    "### Repository Knowledge\n"
    "One bullet per confirmed fact about each repository you have worked in: "
    "directory layout and key files, how to run the tests reliably, coding "
    "conventions, and where specific functionality lives.\n\n"
    "### Lessons & Pitfalls\n"
    "One bullet per lesson about HOW to work: approaches that succeeded, "
    "approaches that failed and why (e.g. fragile edit patterns, wrong test "
    "invocations, submitting without running the full test suite), and checks "
    "worth doing before submitting.\n\n"
    "Carry forward everything you already confirmed in your previous memory; "
    "ADD this task's new findings and only revise an entry when THIS task gives "
    "direct evidence against it — do NOT delete a confirmed fact just because a "
    "task went badly. Output ONLY the two sections."
)


def build_act_user_content(
    *,
    memory: str,
    query_prompt: str,
    include_memory: bool,
) -> str:
    """User content for an ACT turn, mirroring ``icl_summary.respond`` query_parts.

    The first turn of an issue injects the running memory; every turn ends with
    the task query. Feedback (command output) is a separate ``FEEDBACK:`` user
    message added by the caller — not merged here — exactly as the offline system
    does via ``observe``.
    """
    parts: list[str] = []
    if include_memory and memory and memory.strip():
        parts.append("[Memory of what you've learned so far]\n" + memory.strip())
    if query_prompt:
        parts.append(query_prompt)
    return "\n\n".join(parts) if parts else "(no content)"


# ── WRITE prompt v3(2026-07-19, 用户逐字审定; 离线 A/B: 串台 12-16/30 → 2-4/30,
#    严格合规 16-17/30 → 23-24/30): 三明治框架语 + 成败反思引导。
#    经 CODEBASE_WRITE_PROMPT_V3=1 启用; 默认关闭, 保证现役 run 重启口径不漂移。──
WRITE_TRANSCRIPT_PREAMBLE = (
    "Below is the complete, read-only transcript of the task you just finished. "
    "It is provided for reference only."
)

MEMORY_INSTRUCTION_V3_HEAD = (
    "Above is the entire transcript - the task is OVER. Do NOT continue it, do NOT "
    "run commands, do NOT output bash. Your ONLY job is to rewrite the memory based "
    "on the transcript.\n\n"
    "Reflect on the final outcome shown in the transcript before writing: if the "
    "submission PASSED, identify what was key to this success so it can be reused "
    "in future tasks; if it FAILED or the step budget ran out, diagnose the most "
    "likely root cause of the failure and record it as a lesson.\n\n"
)


# ── WRITE prompt for AgentCL coding streams (agentcl_code task, user-approved
#    2026-09-18). Same structure as v3; wording adapted from repositories to a
#    stream of function-implementation problems, first section renamed to
#    "Function Library". Selected per episode via prompt_set="agentcl_code". ──
AGENTCL_MEMORY_SYSTEM = (
    "You are a software engineer solving a sequence of Python function-implementation "
    "problems in the same environment. You keep a running, structured MEMORY of what "
    "you have learned: the functions you have implemented and how to work in this "
    "environment effectively. At the end of each problem you are given your PREVIOUS "
    "memory plus the problem you just worked on, and you output the UPDATED memory — "
    "carrying forward what you already confirmed and folding in the new problem's "
    "evidence — for use in future problems."
)

AGENTCL_MEMORY_INSTRUCTION = (
    "The problem above is over. Update your running memory in EXACTLY these two "
    "sections:\n\n"
    "### Function Library\n"
    "One bullet per function you have implemented: its name and signature, what it "
    "does and returns, and key implementation details.\n\n"
    "### Lessons & Pitfalls\n"
    "One bullet per lesson about HOW to work: approaches that succeeded, approaches "
    "that failed and why, and checks worth doing before submitting.\n\n"
    "Carry forward everything you already confirmed in your previous memory; "
    "ADD this problem's new findings and only revise an entry when THIS problem gives "
    "direct evidence against it — do NOT delete a confirmed fact just because a "
    "problem went badly. Output ONLY the two sections."
)

AGENTCL_WRITE_TRANSCRIPT_PREAMBLE = (
    "Below is the complete, read-only transcript of the problem you just finished. "
    "It is provided for reference only."
)

AGENTCL_MEMORY_INSTRUCTION_V3_HEAD = (
    "Above is the entire transcript - the problem is OVER. Do NOT continue it, do NOT "
    "run commands, do NOT output bash. Your ONLY job is to rewrite the memory based "
    "on the transcript.\n\n"
    "Reflect on the final outcome shown in the transcript before writing: if the "
    "submission PASSED, identify what was key to this success so it can be reused "
    "in future problems; if it FAILED or the step budget ran out, diagnose the most "
    "likely root cause of the failure and record it as a lesson.\n\n"
)

_WRITE_PROMPT_SETS = {
    None: (MEMORY_SYSTEM, MEMORY_INSTRUCTION, WRITE_TRANSCRIPT_PREAMBLE, MEMORY_INSTRUCTION_V3_HEAD,
           "The task above is over. Update your running memory"),
    "agentcl_code": (AGENTCL_MEMORY_SYSTEM, AGENTCL_MEMORY_INSTRUCTION, AGENTCL_WRITE_TRANSCRIPT_PREAMBLE,
                     AGENTCL_MEMORY_INSTRUCTION_V3_HEAD, "The problem above is over. Update your running memory"),
}


def build_write_messages(
    *,
    previous_memory: str,
    instance_messages: list[dict[str, str]],
    v3: bool = False,
    prompt_set: str | None = None,
) -> list[dict[str, str]]:
    """WRITE messages, mirroring the offline memory-rewrite call (window=1):

    ``MEMORY_SYSTEM`` + the finished issue's raw turns + a final user turn
    carrying the previous memory and the two-section rewrite instruction.
    v3=True additionally wraps the transcript in the sandwich framing and
    prepends the outcome-reflection guidance (see WRITE prompt v3 above).
    """
    # prompt_set selects the wording (None = codebase adaptation, the training
    # default; "agentcl_code" = AgentCL coding streams). Unknown values fall back
    # to the default so training episodes are never affected.
    system, instr, preamble, v3_head, over_sentence = _WRITE_PROMPT_SETS.get(
        prompt_set, _WRITE_PROMPT_SETS[None]
    )
    if v3:
        instr = instr.replace(over_sentence, "Update your running memory", 1)
    prev = (previous_memory or "").strip()
    if prev:
        instr = f"Your previous memory:\n{prev}\n\n{instr}"
    if v3:
        instr = v3_head + instr
    turns = [dict(m) for m in instance_messages]
    pre = [{"role": "user", "content": preamble}] if v3 else []
    return [
        {"role": "system", "content": system},
        *pre,
        *turns,
        {"role": "user", "content": instr},
    ]


# ── SkillRise baseline prompts (arXiv 2607.26784, user-approved 2026-10-21) ──
# Same message structure as WRITE v3 (system + read-only transcript + final instruction), but
# the content follows the paper's curate prompt: a task-agnostic SKILL DOCUMENT instead of the
# two-section repository memory, with the attempt's outcome stated explicitly. The ACT side
# injects the document under the paper's solve-prompt wording. Opt-in via
# CODEBASE_PROMPT_SET=skillrise; every other run keeps the prompts above unchanged.

SKILLRISE_EMPTY_SKILL = "(The skill document is currently empty. No skills have been distilled yet.)"

SKILLRISE_ACT_HEADER = (
    "## Current Skill Document\n"
    "The following skills were distilled from earlier tasks of the SAME family. Use them "
    "only when you are confident they apply to the current situation; ignore any skill "
    "that does not fit."
)

SKILLRISE_SYSTEM = (
    "You are a software engineer solving a sequence of related coding tasks. You maintain "
    "a SKILL DOCUMENT that distills transferable skills from the tasks you have finished. "
    "At the end of each task you are given your OLD skill document plus the task you just "
    "worked on, and you output a revised skill document that helps you solve the LATER "
    "tasks more reliably."
)

SKILLRISE_INSTRUCTION_HEAD = (
    "Above is the entire transcript - the task is OVER. Do NOT continue it, do NOT run "
    "commands, do NOT output bash. Your ONLY job is to rewrite the skill document based "
    "on the transcript."
)

SKILLRISE_INSTRUCTION_RULES = (
    "Rewrite the skill document. Start from the old document above and refine it with "
    "what this attempt revealed. Follow these rules strictly:\n"
    "- ABSTRACT, do not memorize. Remove task-specific instances: replace concrete file "
    "paths, function/class names, variable names and issue details with general concepts "
    '("the module under test", "the failing function").\n'
    "- Ground every statement in what actually happened in the attempt above. Do not "
    "invent steps you did not observe.\n"
    "- If the attempt failed, extract the failure point as a concrete pitfall.\n"
    "- Be CONCISE. Do not pad with generic advice or vague tips. A short, precise document "
    "beats a long one. If this attempt revealed little new, keep the old document almost "
    "unchanged.\n"
    "- Do NOT copy the transcript verbatim. Distill reusable procedure, not history.\n\n"
    "Suggested (not mandatory) structure — use the headers that fit:\n"
    "## When to use\n"
    "## Workflow\n"
    "## Pitfalls\n\n"
    "Output ONLY the full revised skill document."
)


def skillrise_prompts_enabled() -> bool:
    return os.environ.get("CODEBASE_PROMPT_SET", "").strip().lower() == "skillrise"


def build_skillrise_act_user_content(*, memory: str, query_prompt: str, include_memory: bool) -> str:
    """ACT user content with the skill document under the paper's solve-prompt header.

    Unlike build_act_user_content, an empty document is shown as the paper's placeholder
    (its solve prompt always carries the document section).
    """
    parts: list[str] = []
    if include_memory:
        doc = (memory or "").strip() or SKILLRISE_EMPTY_SKILL
        parts.append(f"{SKILLRISE_ACT_HEADER}\n{doc}")
    if query_prompt:
        parts.append(query_prompt)
    return "\n\n".join(parts) if parts else "(no content)"


def build_skillrise_write_messages(
    *,
    previous_memory: str,
    instance_messages: list[dict[str, str]],
    success: bool,
) -> list[dict[str, str]]:
    """WRITE (curate) messages: system + read-only transcript + the SkillRise rewrite instruction."""
    old = (previous_memory or "").strip() or SKILLRISE_EMPTY_SKILL
    outcome = "SUCCESS" if success else "FAILURE (task not completed)"
    instr = (
        f"{SKILLRISE_INSTRUCTION_HEAD}\n\n"
        f"Outcome of this attempt: {outcome}\n\n"
        f"## Old Skill Document\n{old}\n\n"
        f"{SKILLRISE_INSTRUCTION_RULES}"
    )
    return [
        {"role": "system", "content": SKILLRISE_SYSTEM},
        {"role": "user", "content": WRITE_TRANSCRIPT_PREAMBLE},
        *[dict(m) for m in instance_messages],
        {"role": "user", "content": instr},
    ]
