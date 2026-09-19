#!/usr/bin/env python3
"""Replay one problem of an AgentCL trajectory and show why it passed/failed.

Re-executes the agent's commands for trial T of an episode trajectory in a fresh
sandbox workspace (same SIF, same task code), then grades the resulting
/testbed/solution.py against the hidden tests and prints the FULL grader output
(the trajectory itself only stores the eval status).

Run inside the miles SIF (needs clbench + apptainer on PATH):
  python replay_grade.py <episode.json> <trial_index> [--show-code] [--show-test]
  python replay_grade.py <episode.json> --list      # trials with status

Replay is faithful for deterministic commands; commands that depend on time or
randomness without a seed may differ from the original run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.environ.get("CLBENCH_ROOT", "/home/qixinx/continual-learning-bench"))
from src.tasks.agentcl_code.task import (  # noqa: E402
    SOLUTION_FILE,
    SUBMIT_SENTINEL,
    Sandbox,
    build_test_program,
    load_by_ids,
    solution_stub,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("trial", nargs="?", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show-code", action="store_true", help="print the final solution.py")
    ap.add_argument("--show-test", action="store_true", help="print the hidden test program")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()
    rec = json.load(open(args.episode))
    trials = rec["trials"]
    if args.list or args.trial is None:
        for i, t in enumerate(trials):
            o = t["outcome"]
            print(f"{i:3d} {o['instance_id']:50s} success={o.get('success')} status={o.get('eval_status')} turns={o.get('turns')}")
        return
    t = trials[args.trial]
    iid = t["outcome"]["instance_id"]
    problem = load_by_ids([iid], Path(os.environ.get("CLBENCH_ROOT", "/home/qixinx/continual-learning-bench")) / "data" / "agentcl_code")[0]
    sb = Sandbox()
    ws = tempfile.mkdtemp(prefix="replay_ws_")
    Path(ws, SOLUTION_FILE).write_text(solution_stub(problem.prompt))
    print(f"== {iid}  original status={t['outcome'].get('eval_status')} success={t['outcome'].get('success')}")
    for k, turn in enumerate(t["turns"]):
        cmd = ((turn.get("action") or {}).get("command") or "").strip()
        if not cmd:
            print(f"-- turn {k}: <empty command>")
            continue
        if cmd == f"echo {SUBMIT_SENTINEL}":
            print(f"-- turn {k}: SUBMIT")
            break
        rc, out = sb.bash(ws, cmd, 60.0)
        print(f"-- turn {k}: rc={rc} cmd={cmd.splitlines()[0][:100]!r}")
    code = Path(ws, SOLUTION_FILE).read_text(errors="replace")
    if args.show_code:
        print("==== final solution.py ====\n" + code)
    prog = build_test_program(problem.record, code)
    if args.show_test:
        print("==== hidden test program (solution + tests) ====\n" + prog)
    gd = tempfile.mkdtemp(prefix="replay_grade_")
    Path(gd, "grade.py").write_text(prog)
    rc, out = sb.python_file(gd, "grade.py", args.timeout)
    print(f"==== grade rc={rc} (0 = pass) ====\n{out[-6000:]}")


if __name__ == "__main__":
    main()
