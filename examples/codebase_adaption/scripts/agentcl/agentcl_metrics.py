#!/usr/bin/env python3
"""Compute AgentCL metrics (PG / SG / GG) from miles eval trajectories.

Definitions (AgentCL, arXiv 2606.02461, Sec. 3.3), per problem i, in
percentage points, computed on complex tasks only (subtasks exist to seed the
memory and are never scored):

  B_i  memoryless accuracy: mean success over the baseline samples of i
  F_i  first pass: success of i inside the pass-1 episode (memory read/write)
  S_i  second pass: success of i with memory frozen at the end of pass 1
  H_j  held-out: success of held-out problem j with that same frozen memory

  PG = mean_i (F_i - B_i)     plasticity gain
  SG = mean_i (S_i - F_i)     stability gain
  GG = mean_j (H_j - B_j)     generalization gain

Each (stream, run) pass-1 episode yields one PG/SG/GG; the table reports the
mean and population std over runs. Accuracies (F, S, H, B) are reported
alongside. PG/SG/GG use the task's pass/fail.

clbench-style counterparts on the same complex tasks, using the
step-discounted reward (1 - regret/40) instead of pass/fail:

  Cum Reward = sum_i r^F_i                      (pass-1 reward)
  Cum Gain   = sum_i (r^F_i - mean_k r^B_{i,k}) (minus the memoryless reward)

and the same two sums on the binary (pass/fail) score, so both scorings are
reported side by side (Cum Success Gain = PG * n / 100):

  Cum Success      = sum_i F_i                  (# complex tasks solved in pass 1)
  Cum Success Gain = sum_i (F_i - B_i)

--per-problem writes one CSV row per (stream, run, complex task) with the
binary F, the memoryless samples' binary outcomes and their mean B, and the
matching rewards.

Default protocol runs only pass 1 + baseline (stage 1), so SG/GG print as n/a
unless stage-2 trajectories are passed in.

Usage:
  python agentcl_metrics.py --traj <stage1 traj dir> <stage2 traj dir> [--json out.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict


def load(traj_dirs: list[str]) -> list[dict]:
    recs = []
    for d in traj_dirs:
        for path in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
            try:
                rec = json.load(open(path))
            except Exception:
                continue
            if rec.get("task_family") == "agentcl_code" and rec.get("agentcl"):
                recs.append(rec)
    return recs


def role_of(iid: str) -> str:
    return "subtask" if ":raw:" in iid else "complex"


def outcomes(rec: dict) -> dict[str, bool]:
    """instance_id -> success for every problem the episode finished."""
    return {o["instance_id"]: bool(o.get("success")) for o in rec.get("outcomes") or [] if "instance_id" in o}


def rewards(rec: dict) -> dict[str, float]:
    """instance_id -> step-discounted reward for every problem the episode finished."""
    return {o["instance_id"]: float(o.get("reward") or 0.0) for o in rec.get("outcomes") or [] if "instance_id" in o}


def pct(x: float) -> float:
    return round(100.0 * x, 1)


def mean_std(xs: list[float], nd: int = 1, sign: bool = True) -> str:
    if not xs:
        return "n/a"
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    head = f"{m:+.{nd}f}" if sign else f"{m:.{nd}f}"
    return f"{head} ± {s:.{nd}f}" if len(xs) > 1 else head


def compute(recs: list[dict]) -> dict:
    base: dict[str, list[bool]] = defaultdict(list)
    base_r: dict[str, list[float]] = defaultdict(list)
    pass1: dict[tuple[str, int], dict] = {}
    pass2: dict[tuple[str, int], dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    held: dict[tuple[str, int], dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for rec in recs:
        tag = rec["agentcl"]
        ph = tag["phase"]
        oc = outcomes(rec)
        if ph == "baseline":
            for iid, ok in oc.items():
                base[iid].append(ok)
            for iid, rw in rewards(rec).items():
                base_r[iid].append(rw)
        elif ph == "pass1":
            pass1[(tag["stream"], int(tag["run"]))] = {
                "outcomes": oc,
                "rewards": rewards(rec),
                "instance_ids": rec.get("instance_ids") or [],
            }
        elif ph == "pass2":
            for iid, ok in oc.items():
                pass2[(tag["stream"], int(tag["run"]))][iid].append(ok)
        elif ph == "heldout":
            for iid, ok in oc.items():
                held[(tag["stream"], int(tag["run"]))][iid].append(ok)

    B = {iid: sum(v) / len(v) for iid, v in base.items()}
    Br = {iid: sum(v) / len(v) for iid, v in base_r.items()}
    per_run = []
    per_problem = []
    for (stream, run), p1 in sorted(pass1.items()):
        complex_ids = [i for i in p1["instance_ids"] if role_of(i) == "complex"]
        sub_ids = [i for i in p1["instance_ids"] if role_of(i) == "subtask"]
        F = {i: float(p1["outcomes"][i]) for i in complex_ids if i in p1["outcomes"]}
        S = {i: sum(v) / len(v) for i, v in pass2.get((stream, run), {}).items() if role_of(i) == "complex"}
        H = {j: sum(v) / len(v) for j, v in held.get((stream, run), {}).items()}
        pg_ids = [i for i in F if i in B]
        sg_ids = [i for i in F if i in S]
        gg_ids = [j for j in H if j in B]
        r = {
            "stream": stream,
            "run": run,
            "pass1_done": len(p1["outcomes"]),
            "pass1_total": len(p1["instance_ids"]),
            "n_complex": len(complex_ids),
            "F_acc": pct(statistics.mean(F.values())) if F else None,
            "subtask_acc": (
                pct(statistics.mean(float(p1["outcomes"][i]) for i in sub_ids if i in p1["outcomes"]))
                if any(i in p1["outcomes"] for i in sub_ids)
                else None
            ),
            "B_acc": pct(statistics.mean(B[i] for i in pg_ids)) if pg_ids else None,
            "S_acc": pct(statistics.mean(S[i] for i in sg_ids)) if sg_ids else None,
            "H_acc": pct(statistics.mean(H.values())) if H else None,
            "B_heldout_acc": pct(statistics.mean(B[j] for j in gg_ids)) if gg_ids else None,
            "PG": pct(statistics.mean(F[i] - B[i] for i in pg_ids)) if pg_ids else None,
            "SG": pct(statistics.mean(S[i] - F[i] for i in sg_ids)) if sg_ids else None,
            "GG": pct(statistics.mean(H[j] - B[j] for j in gg_ids)) if gg_ids else None,
            "cum_reward": round(sum(p1["rewards"][i] for i in pg_ids), 3) if pg_ids else None,
            "cum_baseline_reward": round(sum(Br[i] for i in pg_ids), 3) if pg_ids else None,
            "cum_gain": round(sum(p1["rewards"][i] - Br[i] for i in pg_ids), 3) if pg_ids else None,
            "cum_success": int(sum(F[i] for i in pg_ids)) if pg_ids else None,
            "cum_success_gain": round(sum(F[i] - B[i] for i in pg_ids), 3) if pg_ids else None,
            "n_PG": len(pg_ids),
            "n_SG": len(sg_ids),
            "n_GG": len(gg_ids),
        }
        per_run.append(r)
        for pos, i in enumerate(p1["instance_ids"]):
            if role_of(i) != "complex":
                continue
            per_problem.append({
                "stream": stream,
                "run": run,
                "stream_pos": pos,
                "instance_id": i,
                "F": int(F[i]) if i in F else "",
                "B": round(B[i], 4) if i in B else "",
                "B_samples": "".join(str(int(x)) for x in base.get(i, [])),
                "F_minus_B": round(F[i] - B[i], 4) if (i in F and i in B) else "",
                "reward_F": p1["rewards"].get(i, ""),
                "reward_B": round(Br[i], 4) if i in Br else "",
            })

    by_stream: dict[str, dict] = {}
    for stream in sorted({r["stream"] for r in per_run}):
        rs = [r for r in per_run if r["stream"] == stream]
        agg = {}
        for k in ("F_acc", "S_acc", "B_acc", "H_acc", "B_heldout_acc", "subtask_acc", "PG", "SG", "GG",
                  "cum_reward", "cum_baseline_reward", "cum_gain", "cum_success", "cum_success_gain"):
            vals = [r[k] for r in rs if r[k] is not None]
            if k in ("PG", "SG", "GG"):
                agg[k] = mean_std(vals)
            elif k.startswith("cum_"):
                agg[k] = mean_std(vals, nd=2, sign=k.endswith("gain"))
            else:
                agg[k] = f"{statistics.mean(vals):.1f}" if vals else "n/a"
        agg["runs"] = len(rs)
        by_stream[stream] = agg
    return {"per_run": per_run, "by_stream": by_stream, "per_problem": per_problem, "n_baseline_problems": len(B)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", nargs="+", required=True, help="eval trajectory dir(s) of stage 1 and stage 2")
    ap.add_argument("--json", help="write the full result as JSON here")
    ap.add_argument("--per-problem", help="write per-problem binary/reward scores as CSV here")
    args = ap.parse_args()
    res = compute(load(args.traj))

    print(f"baseline problems scored: {res['n_baseline_problems']}")
    print("\nPer stream (mean ± std over runs; PG/SG/GG in pp, accuracies in %):")
    cols = ["runs", "F_acc", "B_acc", "PG", "cum_success", "cum_success_gain",
            "cum_reward", "cum_baseline_reward", "cum_gain", "subtask_acc"]
    if any(r["S_acc"] is not None or r["H_acc"] is not None for r in res["per_run"]):
        cols += ["S_acc", "SG", "H_acc", "B_heldout_acc", "GG"]
    print("| stream | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for stream, agg in res["by_stream"].items():
        print(f"| {stream} | " + " | ".join(str(agg[c]) for c in cols) + " |")
    print("\nPer run:")
    for r in res["per_run"]:
        print(
            f"  {r['stream']} run={r['run']}: pass1 {r['pass1_done']}/{r['pass1_total']}  "
            f"F={r['F_acc']} S={r['S_acc']} B={r['B_acc']} H={r['H_acc']}  "
            f"PG={r['PG']} (n={r['n_PG']}) CumSuccess={r['cum_success']} CumSuccessGain={r['cum_success_gain']} "
            f"CumReward={r['cum_reward']} CumGain={r['cum_gain']}  "
            f"SG={r['SG']} (n={r['n_SG']}) GG={r['GG']} (n={r['n_GG']})"
        )
    if args.per_problem:
        import csv

        rows = res["per_problem"]
        with open(args.per_problem, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["instance_id"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} per-problem rows to {args.per_problem}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
