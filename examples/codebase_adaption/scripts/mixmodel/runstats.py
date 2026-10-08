import json,glob,sys,statistics
# 用法: runstats.py <run 名>  (run 目录在 $EVAL_TRAJ_ROOT, 默认 ~/eval_traj); 对每个 episode 给 total/解题/轮数, 再给均值与样本 std (ddof=1)
import os
base=f"{os.environ.get('EVAL_TRAJ_ROOT', os.path.expanduser('~/eval_traj'))}/{sys.argv[1]}/traj/eval"
eps=sorted(glob.glob(f"{base}/**/ep_*.json",recursive=True))
tot=[];solved=[];turns=[];fail=[]
for f in eps:
    d=json.load(open(f)); oc=d["outcomes"]
    tot.append(sum(o["reward"] for o in oc)); solved.append(sum(o["success"] for o in oc))
    turns.extend(o["turns"] for o in oc); fail.extend(o["eval_status"] not in ("tests_passed","success") and o["reward"]==0 and o["eval_status"]!="tests_failed" for o in oc)
    st={}
    for o in oc: st[o["eval_status"]]=st.get(o["eval_status"],0)+1
    print(f"  {f.split('/')[-1]}: total={tot[-1]:.3f} solved={solved[-1]} status={st}")
n=len(tot)
print(f"episodes={n} totals={[round(t,3) for t in tot]}")
print(f"mean={statistics.mean(tot):.4f} std(ddof=1)={statistics.stdev(tot) if n>1 else 0:.3f} solved_mean={statistics.mean(solved):.2f} turns/issue={statistics.mean(turns):.1f} non-tests_failed-non-pass(fail%)={100*sum(fail)/len(fail):.1f}%")
