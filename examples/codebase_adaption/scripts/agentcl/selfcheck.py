#!/usr/bin/env python3
"""Environment self-check for the AgentCL coding task (run once on a new server).

Checks, in order:
  1. every problem's initial /testbed/solution.py stub parses;
  2. the sandbox is isolated (no /home, /project, /mnt visible) and starts in /testbed;
  3. the dataset reference solutions, submitted through the real multi-turn task
     (write solution.py with a heredoc, then the submit sentinel), pass grading.

Expected result (known dataset defects, see README):
  bcb_compositional  95/96  (raw/269 broken reference)
  humaneval_heldout 119/120 (new/41 unsolvable)
Anything else means the environment is not set up correctly.

Run inside the miles SIF (needs clbench on sys.path and apptainer on PATH):
  python3 selfcheck.py [--streams bcb_compositional humaneval_heldout]
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import time

sys.path.insert(0, os.environ.get("CLBENCH_ROOT", "/home/qixinx/continual-learning-bench"))
from src.interface import Response  # noqa: E402
from src.tasks.agentcl_code.task import (  # noqa: E402
    DEFAULT_DATA_DIR,
    STREAM_FILES,
    AgentCLCodeTask,
    load_stream,
    solution_stub,
)
from src.tasks.codebase_adaptation.task import BashCommandResponse as Act  # noqa: E402

EXPECTED = {"bcb_compositional": (95, 96), "humaneval_heldout": (119, 120)}


def reference_module(p) -> str:
    """Reference solution as a complete module (what an agent would write)."""
    if re.search(r"^def\s", p.record["problem"], re.M):
        return p.prompt + p.record["solution"]
    m = re.search(r"^def\s+[A-Za-z_]\w*\s*\(.*$", p.record["solution"], re.M)
    return p.prompt + p.record["solution"][m.end() + 1:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+", default=["bcb_compositional", "humaneval_heldout"])
    args = ap.parse_args()
    ok = True

    bad = []
    for s in STREAM_FILES:
        for p in load_stream(s, DEFAULT_DATA_DIR):
            try:
                ast.parse(solution_stub(p.prompt))
            except SyntaxError as e:
                bad.append((s, p.instance_id, str(e)[:60]))
    print(f"[1] stub syntax errors: {len(bad)} {bad[:3]}")
    ok &= not bad

    t = AgentCLCodeTask(stream="humaneval_heldout", num_instances=1)
    t.reset()
    r = t.step(Response(action=Act(thought="", command="pwd; ls -d /home/qixinx /project /mnt 2>&1 | head -3")))
    obs = r.observation.content
    isolated = "/testbed" in obs and "No such file" in obs and "/home/qixinx\n" not in obs
    print(f"[2] sandbox isolated and cwd=/testbed: {isolated}")
    ok &= isolated

    for stream in args.streams:
        t = AgentCLCodeTask(stream=stream)
        t.reset()
        t0, done, fails = time.time(), False, []
        while not done:
            p = t.instances[t.current_issue_idx]
            t.step(Response(action=Act(thought="", command=f"cat <<'AGENTCL_EOF' > solution.py\n{reference_module(p)}\nAGENTCL_EOF")))
            r = t.step(Response(action=Act(thought="", command="echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")))
            if not r.instance_outcome.success:
                fails.append(r.instance_outcome.instance_id)
            done = r.done
        n = len(t.instances)
        passed = n - len(fails)
        exp = EXPECTED.get(stream)
        good = exp is None or (passed, n) == exp
        print(f"[3] {stream}: {passed}/{n} reference solutions pass ({time.time() - t0:.0f}s)"
              f"{'' if good else f'  <-- expected {exp[0]}/{exp[1]}'}  failed: {fails}")
        ok &= good
    print("SELFCHECK", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
