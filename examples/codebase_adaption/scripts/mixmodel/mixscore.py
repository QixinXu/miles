#!/usr/bin/env python3
"""混合模型实验出数: cum reward / cum gain, 与 clbench 同口径。
   cum reward = Σᵢ mean_reward_by_index[i]
   cum gain   = Σᵢ (mean_reward_by_index[i] - baseline[该位置的题])
   baseline = **同一个 ACT 模型**的 stateless 跑分(逐题均值)。
   用法: mixscore.py <replace_run_名> <stateless_run_名>"""
import json, glob, sys, statistics
import os
R = os.environ.get("EVAL_TRAJ_ROOT", os.path.expanduser("~/eval_traj"))
rep, base_run = sys.argv[1], sys.argv[2]

def collect(run):
    d = {}
    for f in glob.glob(f"{R}/{run}/traj/eval/heldout/rollout_0/ep_*.json"):
        for o in json.load(open(f)).get("outcomes", []):
            d.setdefault(o["instance_id"], []).append(float(o.get("reward", 0)))
    return d

rw, bs = collect(rep), collect(base_run)
if not rw: sys.exit(f"没有轨迹: {rep}")
order = sorted(rw)
mr = [statistics.mean(rw[i]) for i in order]
mg = [statistics.mean(rw[i]) - (statistics.mean(bs[i]) if i in bs else 0.0) for i in order]
neps = len(glob.glob(f"{R}/{rep}/traj/eval/heldout/rollout_0/ep_*.json"))
print(f"{rep}")
print(f"  episodes={neps}/5   题数={len(order)}   每题 run 数={sorted({len(rw[i]) for i in order})}")
print(f"  cum reward = {sum(mr):+.4f}")
print(f"  cum gain   = {sum(mg):+.4f}   (基线: {base_run})")
