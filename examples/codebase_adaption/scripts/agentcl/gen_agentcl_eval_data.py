#!/usr/bin/env python3
"""Build miles eval datasets for the AgentCL coding protocol (agentcl_code task).

The protocol has a sequential dependency, so it runs as two eval launches:

stage1  (no dependency)
  pass1     one long episode per (stream, run): the whole stream in official
            order, memory mode (ACT + WRITE after every problem).
  baseline  one single-problem episode per (problem, sample): empty memory,
            no WRITE -> the memoryless score B used by PG and GG.

stage2  (needs the stage-1 trajectories: the final memory of every pass-1 run)
  pass2     single-problem episodes over the same stream, memory frozen at the
            final pass-1 memory, no WRITE -> S used by SG. Frozen memory and a
            per-problem ACT context make this identical to a sequential
            read-only second pass, so the episodes run in parallel.
  heldout   single-problem episodes over the held-out stream with the same
            frozen memory, no WRITE -> H used by GG.

Usage:
  python gen_agentcl_eval_data.py stage1 --out data/agentcl/stage1.jsonl
  python gen_agentcl_eval_data.py stage2 --stage1-traj <traj/eval/heldout/rollout_0> \
      --out data/agentcl/stage2.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

CLBENCH_ROOT = Path(os.environ.get("CLBENCH_ROOT", "/home/qixinx/continual-learning-bench"))
DATA_DIR = CLBENCH_ROOT / "data" / "agentcl_code"
STREAM_FILES = {
    "bcb_compositional": "bigcodebench_lite_pro.compositional.json",
    "bcb_naive": "bigcodebench_lite_pro.naive.json",
    "humaneval_heldout": "humaneval_pro.naive.json",
    "humaneval_compositional": "humaneval_pro.compositional.json",
}


def _family(fname: str) -> str:
    return "humaneval_pro" if fname.startswith("humaneval") else "bigcodebench_lite_pro"


def stream_items(stream: str) -> list[tuple[str, str]]:
    """[(instance_id, role)] in official stream order (ids match the clbench task)."""
    fname = STREAM_FILES[stream]
    records = json.loads((DATA_DIR / fname).read_text())
    return [
        (f"{_family(fname)}:{r['category']}:{r['id']}", "subtask" if r["category"] == "raw" else "complex")
        for r in records
    ]


def row(text: str, *, ids: list[str], roles: list[str], agentcl: dict, **extra) -> dict:
    meta = {
        "split": "agentcl",
        "task_family": "agentcl_code",
        "instance_ids": ids,
        "stage_labels": roles,
        "agentcl": agentcl,
        **extra,
    }
    return {"prompt": [{"role": "user", "content": [{"type": "text", "text": text}]}], "metadata": meta}


def build_stage1(args) -> list[dict]:
    rows: list[dict] = []
    # Long sequential episodes first so they start before the short ones.
    for stream in args.streams:
        items = stream_items(stream)
        for run in range(args.runs):
            rows.append(
                row(
                    f"AgentCL pass1 stream={stream} run={run}",
                    ids=[i for i, _ in items],
                    roles=[r for _, r in items],
                    agentcl={"phase": "pass1", "stream": stream, "run": run},
                )
            )
    seen: dict[str, str] = {}
    # Held-out baselines only feed GG (stage 2, off by default); opt in with --heldout-baseline.
    for stream in [*args.streams, *([args.heldout] if args.heldout_baseline else [])]:
        for iid, role in stream_items(stream):
            if role == "subtask" and not args.baseline_subtasks:
                continue
            seen.setdefault(iid, role)
    for iid, role in seen.items():
        for k in range(args.baseline_samples):
            rows.append(
                row(
                    f"AgentCL baseline {iid} sample={k}",
                    ids=[iid],
                    roles=[role],
                    agentcl={"phase": "baseline", "sample": k},
                    skip_write=True,
                )
            )
    return rows


def load_pass1_memories(traj_dirs: list[str]) -> dict[tuple[str, int], dict]:
    """(stream, run) -> {"memory", "n_done", "n_total", "path"} from pass-1 trajectories."""
    out: dict[tuple[str, int], dict] = {}
    for d in traj_dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            rec = json.load(open(path))
            tag = rec.get("agentcl") or {}
            if tag.get("phase") != "pass1":
                continue
            key = (tag["stream"], int(tag["run"]))
            summaries = rec.get("summaries") or []
            out[key] = {
                "memory": summaries[-1] if summaries else "",
                "n_done": len(rec.get("outcomes") or []),
                "n_total": len(rec.get("instance_ids") or []),
                "path": path,
            }
    return out


def build_stage2(args) -> list[dict]:
    memories = load_pass1_memories(args.stage1_traj)
    if not memories:
        sys.exit(f"no pass1 trajectories found under {args.stage1_traj}")
    rows: list[dict] = []
    for (stream, run), m in sorted(memories.items()):
        if m["n_done"] < m["n_total"]:
            print(
                f"WARNING pass1 {stream} run={run} finished {m['n_done']}/{m['n_total']} problems "
                f"(episode aborted?); its frozen memory is the last one written. {m['path']}",
                file=sys.stderr,
            )
        if not m["memory"].strip():
            print(f"WARNING pass1 {stream} run={run} ended with an empty memory", file=sys.stderr)
        for iid, role in stream_items(stream):
            if role == "subtask" and not args.pass2_subtasks:
                continue
            rows.append(
                row(
                    f"AgentCL pass2 {iid} stream={stream} run={run}",
                    ids=[iid],
                    roles=[role],
                    agentcl={"phase": "pass2", "stream": stream, "run": run},
                    skip_write=True,
                    initial_memory=m["memory"],
                )
            )
        for iid, role in stream_items(args.heldout):
            rows.append(
                row(
                    f"AgentCL heldout {iid} after stream={stream} run={run}",
                    ids=[iid],
                    roles=[role],
                    agentcl={"phase": "heldout", "stream": stream, "run": run, "heldout_stream": args.heldout},
                    skip_write=True,
                    initial_memory=m["memory"],
                )
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("stage1", "stage2"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--streams", nargs="+", default=["bcb_compositional", "bcb_naive"])
        p.add_argument("--heldout", default="humaneval_heldout")
    s1 = sub.choices["stage1"]
    s1.add_argument("--runs", type=int, default=3, help="independent pass-1 runs per stream")
    s1.add_argument("--baseline-samples", type=int, default=3, help="memoryless samples per problem")
    s1.add_argument(
        "--heldout-baseline", action="store_true",
        help="also score the held-out stream memorylessly (needed only for GG / stage 2)",
    )
    s1.add_argument(
        "--baseline-subtasks", action="store_true", help="also score subtasks memorylessly (not used by PG/SG/GG)"
    )
    s2 = sub.choices["stage2"]
    s2.add_argument("--stage1-traj", nargs="+", required=True, help="stage-1 eval trajectory dir(s)")
    s2.add_argument("--pass2-subtasks", action="store_true", help="also re-run subtasks in pass 2")
    args = ap.parse_args()

    rows = build_stage1(args) if args.stage == "stage1" else build_stage2(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    phases: dict[str, int] = {}
    for r in rows:
        ph = r["metadata"]["agentcl"]["phase"]
        phases[ph] = phases.get(ph, 0) + 1
    print(f"wrote {len(rows)} episodes to {args.out}: {phases}")


if __name__ == "__main__":
    main()
