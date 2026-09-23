#!/usr/bin/env python3
"""Run the AgentCL eval data (gen_agentcl_eval_data.py) with an OpenAI API model.

Same scaffold as the miles rollout (codebase_rollout.generate), reused rather
than re-implemented where possible:
  - task: clbench agentcl_code (bash sandbox, submit sentinel, 40-step budget)
  - ACT: plain-text bash-block format (miles override of the system template),
    memory injected on an issue's first turn (build_act_user_content), budget
    line every turn, FEEDBACK messages, multi-block / missing-block notes,
    command parsing (_response_from_text), step-budget force-complete
  - WRITE: build_write_messages (prompt v3 by default) after every issue unless
    the episode is skip_write; memory replaced by the model's visible output
Differences forced by the API: the model's hidden reasoning is not part of the
context (only visible text is), and no token-level sampling params.

Trajectories are written in the miles eval format, one JSON per episode, so
agentcl_metrics.py reads them unchanged. Finished episodes are skipped on
re-run (resume).

--task swe_smith (2026-09-22) runs the held-out SWE-smith sequences instead:
the task is built by codebase_rollout._make_task exactly as the miles eval
does (CODEBASE_EVAL_TASK=swe_smith + --eval-dataset registry), and episodes
with metadata no_memory get the miles stateless path (bare issue prompt, no
WRITE). --n-samples repeats every row (avg@N for stateless).

Run inside the miles SIF (imports miles + clbench), e.g.
  python run_api_agentcl.py --data stage1.jsonl --out <traj dir> --model gpt-5.5
Needs OPENAI_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, "/home/qixinx/miles")
CLBENCH_ROOT = os.environ.setdefault("CLBENCH_ROOT", "/home/qixinx/continual-learning-bench")
sys.path.insert(0, CLBENCH_ROOT)

import examples.codebase_adaption.codebase_rollout as cr  # noqa: E402
from examples.codebase_adaption.prompts import build_act_user_content, build_write_messages  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from src.interface import Response  # noqa: E402
from src.tasks.agentcl_code.task import AgentCLCodeTask  # noqa: E402

USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0, "errors": 0}


async def chat(client: AsyncOpenAI, args, messages: list[dict], *, max_tokens: int) -> tuple[str, str]:
    """(visible text, finish_reason). Retries transient API failures with backoff."""
    delay = 5.0
    for attempt in range(8):
        try:
            kw = {"model": args.model, "messages": messages, "max_completion_tokens": max_tokens}
            if args.reasoning_effort:
                kw["reasoning_effort"] = args.reasoning_effort
            resp = await client.chat.completions.create(**kw)
            u = resp.usage
            USAGE["calls"] += 1
            if u is not None:
                USAGE["prompt_tokens"] += u.prompt_tokens or 0
                USAGE["completion_tokens"] += u.completion_tokens or 0
                d = getattr(u, "completion_tokens_details", None)
                USAGE["reasoning_tokens"] += (getattr(d, "reasoning_tokens", 0) or 0) if d else 0
                pd = getattr(u, "prompt_tokens_details", None)
                USAGE["cached_tokens"] += (getattr(pd, "cached_tokens", 0) or 0) if pd else 0
            choice = resp.choices[0]
            return cr._strip_lone_surrogates(choice.message.content or ""), choice.finish_reason or "stop"
        except Exception as e:  # rate limits, 5xx, timeouts
            USAGE["errors"] += 1
            if attempt == 7:
                raise
            print(f"[api] {type(e).__name__}: {str(e)[:200]} -- retry in {delay:.0f}s", flush=True)
            await asyncio.sleep(delay + random.uniform(0, 2))
            delay = min(delay * 2, 120)
    raise RuntimeError("unreachable")


async def run_episode(client, args, meta: dict, ep_index: int) -> dict:
    """One eval episode; mirrors codebase_rollout.generate for evaluation=True."""
    skip_write = bool(meta.get("skip_write"))
    no_memory = bool(meta.get("no_memory"))
    memory = str(meta.get("initial_memory") or "")
    max_steps = args.max_steps
    if args.task == "swe_smith":
        task = cr._make_task(
            SimpleNamespace(codebase_max_steps_per_issue=max_steps),
            str(meta.get("split") or "heldout"),
            list(meta["instance_ids"]),
            list(meta["stage_labels"]),
            task_family=meta.get("task_family"),
        )
    else:
        task = AgentCLCodeTask(
            instance_ids=list(meta["instance_ids"]),
            max_steps_per_issue=max_steps,
            work_root=args.work_root,
        )
    query = await asyncio.to_thread(task.reset)
    feedback_text = None
    episode_messages: list[dict] = []
    outcomes, trials, summaries, write_audit = [], [], [], []
    try:
        while query is not None:
            trial_pos = int((query.metadata or {}).get("active_issue_index", len(outcomes)))
            instance_id = str(query.instance_id)
            # no_memory (stateless) keeps one context across the episode, as in miles
            messages: list[dict] = episode_messages if no_memory else []
            trial_messages: list[dict] = []
            transcript: list[dict] = []
            memory_in = memory
            step_result = None
            first_turn = True
            exhausted = True
            for turn_idx in range(max_steps):
                if feedback_text:
                    fb = {"role": "user", "content": f"FEEDBACK: {feedback_text.strip()}"}
                    messages.append(fb)
                    trial_messages.append(dict(fb))
                if no_memory:
                    user_content = query.prompt or "(no content)"
                else:
                    user_content = build_act_user_content(memory=memory, query_prompt=query.prompt,
                                                          include_memory=first_turn)
                user_content = (
                    f"{user_content}\n\n"
                    f"[Budget] You have {max_steps - turn_idx} of {max_steps} turns remaining for this "
                    f"{getattr(task, 'PROMPT_UNIT', 'issue')}."
                )
                messages.append({"role": "user", "content": user_content})
                trial_messages.append({"role": "user", "content": user_content})
                act_text, act_finish = await chat(client, args, messages, max_tokens=args.max_act_tokens)
                act_text = cr._strip_special(act_text)
                action = cr._response_from_text(query, act_text)
                messages.append({"role": "assistant", "content": act_text})
                trial_messages.append({"role": "assistant", "content": act_text})
                first_turn = False
                step_result = await asyncio.to_thread(task.step, Response(action=action, metadata={"finish": act_finish}))
                obs = step_result.observation
                transcript.append({
                    "step": len(transcript),
                    "user": user_content,
                    "assistant": act_text,
                    "action": cr._jsonable(action),
                    "act_finish": act_finish,
                    "observation": obs.content,
                    "instance_complete": bool(obs.instance_complete),
                })
                if step_result.done or obs.instance_complete or step_result.next_query is None:
                    feedback_text = obs.content if no_memory else None
                    exhausted = False
                    break
                feedback_text = obs.content
                n_blocks = len(cr._BASH_BLOCK_RE.findall(act_text or ""))
                if not str(getattr(action, "command", "") or "").strip():
                    if n_blocks >= 2:
                        note = (
                            f"Your previous response contained {n_blocks} ```bash code blocks; "
                            "exactly ONE is required, so no command was executed. "
                            "Reply with a single ```bash block containing one command."
                        )
                    else:
                        note = "No complete ```bash code block was found in your previous response."
                    if act_finish == "length":
                        note += f" It may have been cut off by the {args.max_act_tokens}-token response limit."
                    feedback_text = f"{obs.content}\n{note}"
                query = step_result.next_query
            if exhausted and step_result is not None and not step_result.observation.instance_complete:
                step_result = cr._force_complete_current_issue(task, max_steps_per_issue=max_steps)
                feedback_text = step_result.observation.content if no_memory else None

            oo = step_result.instance_outcome if step_result is not None else None
            outcome = {
                "instance_id": oo.instance_id if oo else instance_id,
                "reward": float(oo.reward) if oo else 0.0,
                "success": bool(oo.success) if oo else False,
                "steps": (oo.metadata or {}).get("steps") if oo else None,
                "turns": len(transcript),
                "eval_status": (oo.metadata or {}).get("eval_status") if oo else "no_outcome",
                "trial_pos": trial_pos,
            }
            outcomes.append(outcome)
            labels = meta.get("stage_labels") or []
            trial = {"trial_pos": trial_pos, "instance_id": outcome["instance_id"],
                     "stage": labels[trial_pos] if trial_pos < len(labels) else None, "memory_in": memory_in,
                     "turns": transcript, "outcome": outcome, "write": None}

            if not (skip_write or no_memory):
                final_fb = (step_result.observation.content or "").strip() if step_result is not None else ""
                if final_fb:
                    trial_messages.append({"role": "user", "content": f"FEEDBACK: {final_fb}"})
                write_messages = build_write_messages(previous_memory=memory, instance_messages=trial_messages,
                                                      v3=args.write_v3, prompt_set=meta.get("task_family"))
                write_text, write_finish = await chat(client, args, write_messages, max_tokens=args.max_write_tokens)
                previous = memory
                memory = cr._strip_special(write_text) or previous
                summaries.append(memory)
                wr = {"rewrite_idx": trial_pos, "previous_memory": previous, "raw_output": write_text,
                      "memory": memory, "finish": write_finish}
                trial["write"] = wr
                write_audit.append(wr)
            trials.append(trial)

            if step_result is None or step_result.done or step_result.next_query is None:
                break
            if not no_memory:
                feedback_text = None
            query = step_result.next_query
    except Exception as exc:
        print(f"[episode {ep_index}] ABORTED: {exc}\n{traceback.format_exc()}", flush=True)
        outcomes.append({"instance_id": locals().get("instance_id", "?"), "reward": 0.0, "success": False,
                         "eval_status": "env_error", "trial_pos": len(outcomes)})
    finally:
        task._cleanup_container()

    return {
        "episode_id": f"api:{ep_index}",
        "index": ep_index,
        "evaluation": True,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "task_family": meta.get("task_family"),
        "agentcl": meta.get("agentcl"),
        "instance_ids": meta.get("instance_ids"),
        "stage_labels": meta.get("stage_labels"),
        "skip_write": skip_write,
        "no_memory": no_memory,
        "split": meta.get("split"),
        "order_seed": meta.get("order_seed"),
        "initial_memory": meta.get("initial_memory") or None,
        "max_steps_per_issue": max_steps,
        "outcomes": outcomes,
        "summaries": summaries,
        "write_audit": write_audit,
        "trials": trials,
    }


async def main_async(args) -> None:
    if args.task == "swe_smith":
        # read by codebase_rollout._make_task (eval branch), same as the miles eval
        os.environ["CODEBASE_EVAL_TASK"] = "swe_smith"
        os.environ["CODEBASE_EVAL_DATASET"] = args.eval_dataset
    cr._apply_text_format_override()
    rows = [json.loads(line) for line in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]
    # --n-samples: row i, sample k -> episode i * n + k
    rows = [r for r in rows for _ in range(args.n_samples)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(timeout=args.api_timeout, max_retries=0)
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    done = {"n": 0}
    todo = [(i, r) for i, r in enumerate(rows) if not (out / f"ep_{i:05d}.json").exists()]
    print(f"{len(rows)} episodes, {len(rows) - len(todo)} already done, running {len(todo)} "
          f"with concurrency {args.concurrency}", flush=True)

    async def one(i: int, r: dict) -> None:
        async with sem:
            rec = await run_episode(client, args, r["metadata"], i)
            tmp = out / f".ep_{i:05d}.json.tmp"
            tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
            tmp.rename(out / f"ep_{i:05d}.json")
            done["n"] += 1
            ok = sum(bool(o.get("success")) for o in rec["outcomes"])
            tag = rec["agentcl"] or {}
            print(f"[{time.time() - t0:7.0f}s] done {done['n']}/{len(todo)} ep {i} {tag.get('phase')} "
                  f"{tag.get('stream', '')} solved {ok}/{len(rec['outcomes'])} | usage {USAGE}", flush=True)
            (out / "usage.json").write_text(json.dumps({**USAGE, "model": args.model,
                                                        "elapsed_s": round(time.time() - t0)}, indent=1))

    await asyncio.gather(*(one(i, r) for i, r in todo))
    print(f"finished in {time.time() - t0:.0f}s; usage {USAGE}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="jsonl from gen_agentcl_eval_data.py")
    ap.add_argument("--out", required=True, help="trajectory output dir")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--reasoning-effort", default=None, help="omit to use the API default")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--max-act-tokens", type=int, default=32768)
    ap.add_argument("--max-write-tokens", type=int, default=16384,
                    help="covers hidden reasoning + the visible memory")
    ap.add_argument("--no-write-v3", dest="write_v3", action="store_false")
    ap.add_argument("--api-timeout", type=float, default=600.0)
    ap.add_argument("--work-root", default=None)
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows (pilot)")
    ap.add_argument("--task", choices=("agentcl_code", "swe_smith"), default="agentcl_code")
    ap.add_argument("--eval-dataset", default="/data/user_data/qixinx/swe_smith/heldout_v4/heldout_registry.jsonl",
                    help="--task swe_smith: instance registry (CODEBASE_EVAL_DATASET)")
    ap.add_argument("--n-samples", type=int, default=1, help="episodes per row (4 = avg@4 stateless)")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
