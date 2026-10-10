#!/usr/bin/env python3
"""Build the two data-ablation training sets from the 6+6 v3 episodes (2026-10-09).

  --mode 3p3:     each 6+6 episode is split into two 3+3 episodes. Within each repo
                  segment (already sorted easiest-first), issues of the same difficulty
                  bucket are dealt alternately to the two halves, so a mixed segment
                  (2 easy/2 mid/2 hard) yields easy+mid+hard in both halves. Both halves
                  keep the original easiest-first order; repoA half i pairs with repoB half i.
  --mode shuffle: same 6+6 episodes, issue order shuffled within each repo segment
                  (repo blocks and repo order unchanged); a draw equal to the original
                  order is re-drawn.
"""
import argparse, json, random, sys
sys.path.insert(0, "/home/qixinx/continual-learning-bench/scripts/swesmith")
from gen_episodes_v2 import bucket_of

PER_ISSUE = ("instance_ids", "stage_labels", "strategies")


def segments(md):
    out = []
    for lo, hi, repo in ((0, 6, md["repoA"]), (6, 12, md["repoB"])):
        assert all(l == repo for l in md["stage_labels"][lo:hi]), md["episode_index"]
        out.append([{k: md[k][i] for k in PER_ISSUE} for i in range(lo, hi)])
    return out


def split_half(seg):
    halves, turn = ([], []), {}
    for x in seg:  # seg is easiest-first; deal each bucket alternately, alternating start per bucket
        b = bucket_of(x["strategies"])
        k = turn.setdefault(b, len(turn) % 2)
        halves[k % 2].append(x)
        turn[b] = k + 1
    return halves


def episode(md, issues, idx, tier_suffix, text):
    new = {k: v for k, v in md.items() if k not in PER_ISSUE}
    for k in PER_ISSUE:
        new[k] = [x[k] for x in issues]
    new["episode_index"] = idx
    new["source_episode_index"] = md["episode_index"]
    new["difficulty_tier"] = md["difficulty_tier"] + tier_suffix
    return {"prompt": [{"role": "user", "content": [{"type": "text", "text": text}]}], "metadata": new}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("3p3", "shuffle"), required=True)
    ap.add_argument("--seed", type=int, default=20261009)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = [json.loads(l) for l in open(a.src)]
    out = []
    for r in rows:
        md = r["metadata"]
        sa, sb = segments(md)
        if a.mode == "3p3":
            (a0, a1), (b0, b1) = split_half(sa), split_half(sb)
            for h, (x, y) in enumerate(((a0, b0), (a1, b1))):
                assert len(x) == 3 and len(y) == 3
                out.append(episode(md, x + y, len(out), "_3p3",
                                   f"SWE-smith 3p3 v3-{md['difficulty_tier']} episode {md['episode_index']}.{h}: {md['repoA']} -> {md['repoB']}."))
        else:
            segs = []
            for s in (sa, sb):
                p = s[:]
                while [x["instance_ids"] for x in p] == [x["instance_ids"] for x in s]:
                    rng.shuffle(p)
                segs.append(p)
            out.append(episode(md, segs[0] + segs[1], len(out), "_shuf",
                               f"SWE-smith 6p6 v3-{md['difficulty_tier']} episode {md['episode_index']} (within-repo shuffled): {md['repoA']} -> {md['repoB']}."))
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"{a.mode}: {len(rows)} source episodes -> {len(out)} episodes -> {a.out}")


if __name__ == "__main__":
    main()
