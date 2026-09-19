# ACT-only Run: Late-Training Policy Collapse Analysis

**Run**: `smith-4b-v3curr-actonly-v3think` (Qwen3.5-4B, 120 rollouts, completed 2026-07-28)
**Setup**: v3 three-tier curriculum consumed sequentially (63 pure-easy → 138 easy-mid → 181 mixed episodes); only ACT samples receive gradient (`CODEBASE_TRAIN_ACT_ONLY=1`); WRITE (memory rewriting, v3 prompt + thinking) is still generated and consumed every task but never trained. Official step-discount reward, no zero-std group filtering.
**Companion runs**: grace12 (same data, ACT+WRITE trained, 12-step grace reward) and the babel v3curr gated run.
**Case viewer**: 12 full trajectories across all phases (early prior → compression → healthy plateau → verbosity episodes → collapse) — see the published trajectory viewer.

---

## 1. Summary

The run collapsed late in training, and the collapse is a **policy/generation failure, not a data-difficulty artifact**:

- Overall success: 30–50% (mid-run) → 2–11% (r105+); mean turns 8–15 → 22–35 (cap = 40).
- **Difficulty-controlled evidence**: `func_basic` (easy-tier) instances, same strategy family throughout, fell from **43.6% success / 10.5 turns (r60–70) to 16.2% / 26.9 turns (r105–118)**. Mixed-tier episodes contain 2 easy instances each; a healthy policy would keep solving them.
- The proximate mechanism is **generation-side format breakdown**: the model stops producing parseable bash code blocks, and trials burn their 40-turn budget in loops of empty turns.
- Memory (WRITE output) degrades severely in the same window, but paired analysis shows it is a **downstream symptom, not a cause** (§4).

The collapse unfolds in two distinct phases with a sharp transition (§3), which yields a concrete checkpoint recommendation (§6) and monitoring/reward lessons for the next run (§7).

## 2. What the extra turns actually contain

Full-population statistics over easy-tier ACT trials (mid r60–70, n=544 vs late r105–118, n=776):

| Metric | Mid r60–70 | Late r105–118 |
|---|---|---|
| Success | 43.6% | 16.2% |
| Mean turns | 10.5 | 26.9 |
| Mean *effective* commands per trial | 9.5 | **5.9** (fewer than mid-run) |
| Turns with no usable command | 6.3% | **61.8%** |
| Trials with ≥1 unusable turn | 34.9% | 83.5% |
| Mean assistant chars per turn | 845 | **4,234** |
| Turns containing degenerate repetition (a single token ≥50×) | 1.4% | **76.0%** |
| Trials ending in a deliberate submit | 92.3% | 46.0% |
| Trial outcome mix | completed 237 / tests_failed 258 / budget 35 | budget **419** / completed 126 / tests_failed 122 / empty_patch 107 |

Decomposition of the 16,869 unusable turns in the late window: **70.2% contain no bash code block at all**; 16.9% hit the 2,500-token response cap so the code fence never closes; 9.9% emit multiple bash blocks (violating the one-block protocol); 3.0% emit literal placeholders (`your_command_here`, `command`, `...`).

Ruled out as primary mechanisms: repeated identical commands (1.0% at the command level), sed-loops on one file (0.5%, unchanged from mid-run), over-exploration (actually *drops* 20.2% → 1.3%), and "fixed it but never submitted" (0 cases — unlike grace12). Editing ability itself is intact: ~1.8 edit actions per trial in both windows.

**The failure loop**: a response with no parseable block → environment replies "Empty command" → the next response's reasoning gets *longer and more frantic* → parse fails again. Each iteration burns a turn (but not a step), so a trial can consume the entire 40-turn budget while performing 5 real actions. Onset within a trial is immediate, not gradual: the median first bad turn is **turn 1** (29% of late trials are broken from turn 0) — the sampling distribution is globally poisoned, not trapped mid-task.

Typical late-stage forms (all in the viewer):
1. **"OK!" repetition**: reasoning tails degenerate into hundreds of `OK! OK! OK!` until truncation (r111 sqlfluff; 40 turns, 5 effective commands). Notably, the model sometimes *comments on its own failure* mid-trial ("my output exceeded the 2500-token limit! I must shorten!") and then produces a 7k-character response on the next turn.
2. **Parser probing**: between empty turns the model emits `echo ok` / `echo hello` / `mycommand` as if testing the harness, then submits an empty patch (r108 astroid).
3. **Silent exhaustion**: 36 consecutive empty turns after 4 normal ones (r114 patsy).
4. **Residual capability**: occasionally the same weights recover mid-trial and solve the task (r107 typeguard, r105 parso) — the capability is not erased; the output distribution is contaminated.

## 3. Collapse timeline: slow precursor, sharp transition

Per-rollout scan of ACT-side degeneration vs WRITE-side degeneration:

| r | ACT repetition% | ACT chars/turn | ACT turns >3k chars | WRITE OK-spam% | WRITE truncated% | Success% |
|---|---|---|---|---|---|---|
| 66–72 | 0 | 918–1,560 | 8–16% | 0 | 12–16% | 33–37 |
| 75–81 | 0.5 | 2,082–2,535 | 24–29% | 0 | 10–13% | 11→45 (episode, recovered) |
| 84–87 | 0 | 1,431–1,768 | 13–17% | 0 | 12–18% | 22–42 |
| 90–99 | 0–3.6 | **3,188→5,584** | 36→70% | 0 | 22→49% | 29→19 |
| **102** | **74.5** | 4,472 | 60% | 4.2 | 32% | **3.6** |
| 105–117 | 63→92 | 4,500–5,100 | 65–83% | 9→78% | 40→85% | 2–11 |

Three readings:

1. **Phase 1 (r90–99): runaway reasoning length.** Chars/turn quadruple while repetition is still ~0. Success declines gently. An earlier, smaller verbosity episode at r75–81 (2.5k chars/turn, success dip to 11.5%) **self-recovered** by r84–87 — the same pattern as the transient template-echo episode in smith-v2 (r84–104) that also self-healed. The r90 episode did not recover.
2. **Phase 2 (r99→r102): discrete jump into the repetition attractor.** ACT repetition goes 3.6% → 74.5% within one 3-rollout interval; success drops 18.8% → 3.6% simultaneously. This is a phase transition in the sampling distribution, not a gradual drift.
3. **Verbosity is a leading indicator.** Length inflation precedes the crash by 10+ rollouts and is visible long before reward moves decisively. It is a better early-warning signal than reward or success rate.

## 4. Memory degradation is downstream, not causal

In this run WRITE receives no gradient, so its output can only change via (a) weight drift caused by ACT training (shared weights) and (b) its input — the ACT transcript it summarizes. Both paths point ACT → WRITE, and the data confirms the ordering and the (non-)effect:

- **WRITE content quality**: length 2.1k → 4.3k chars (median 4.9k late); truncation 1% → 14% → **61%**; OK-spam contamination 0% → 0% → **50.6%**; WRITE thinking blocks go from 99% present-and-closed to **60% absent**. Structure (headers/bullets) survives at ~92% throughout — the degradation fills the structure with junk rather than destroying it.
- **Strict lag**: at r102, ACT repetition is already 74.5% while WRITE OK-spam is 4.2%; WRITE climbs to 78% over the next 15 rollouts. Late memories frequently *quote* the ACT transcript's own OK-spam ("ALL my responses were full of 'OK OK OK…'") — WRITE is faithfully summarizing an already-broken ACT stream.
- **Near-zero causal effect on ACT**: pairing trials by (rollout × instance × position), clean-memory minus contaminated-memory success difference is **+0.007** (n=239 pairs). And at trial position 0 — where memory input is empty — success has already fallen from ~0.46–0.53 to 0.207 with turns at 27.7. The weights themselves are broken before any memory enters the context.
- **The self-diagnosis exhibit**: several late memories are well-structured *instructions for curing the exact pathology* ("NEVER output excessive reasoning… short reasoning + ONE CODE BLOCK… the response must contain the bash block"). The knowledge layer can articulate the fix; the sampling distribution cannot execute it. This is the cleanest single demonstration that context-level knowledge does not repair a policy-level collapse.

## 5. Cross-run comparison: shared collapse, different flavors

grace12 (same data, ACT+WRITE trained, grace reward) collapses in the same window with the same core signature — easy-tier success 38.6% → **6.7%**, turns 19.4 → 38.1, deliberate submissions 80.5% → **10.2%**, unusable turns 5.5% → 53.2% — plus two flavors of its own: "fixed-and-verified but never submits" loops (self-congratulatory `echo "Fix verified"` variants for 20–30 turns), and prompt-template parroting ("Your reasoning and analysis here."). actonly's flavor is OK-spam; it does not show the never-submit pathology (its late submits are real, just often empty patches).

Since the two runs differ in reward shape and in whether WRITE is trained, but share the data schedule and the collapse window, the driver is **common late-stage training dynamics on the v3 curriculum tail**, not the grace mechanism and not WRITE training. Consistent with the SNR view (RAGEN-2): the mixed-tier tail is nearly unsolvable for this policy (multiple all-zero-reward rollouts from r111 on), reward variance vanishes, the task gradient goes quiet, and degenerate high-prior attractors (repetition, template echo) absorb probability mass. Verbosity growth compounds this: nothing in the reward penalizes long reasoning or unparseable turns — an empty turn burns budget but produces no differential learning signal, and a failing trial scores 0 regardless of how it fails.

## 5b. Proposed mechanism: why the policy suddenly got worse

Causal chain (each link labeled as measured fact vs. inference):

**Step 1 — A think-aloud register expands through the shared weights, unchecked because WRITE is untrained (revised 2026-07-29; cross-run evidence).** What inflates is not generic length but a specific *register*: think-aloud markers ("wait" ×7.5, "hmm" ×31, exclamation marks ×13 per turn vs. baseline) flooding the ACT reasoning — a channel that has no thinking mode. The trajectory is prune → rebound → acceleration: early RL first *suppressed* the pretrained think-aloud habit (r6–18 at 0.03 "wait"/turn, 3× below the base model), it rebounded from r22 as data hardened, and accelerated sharply from r46. Deep reading shows the register initially bought real reasoning quality (line-by-line tracing and explicit self-correction replacing fluent confabulation; success 28%→39%), which is why the drift was reward-compatible. Nothing prices response length or register (the step discount counts turns, not characters), so there was no restoring force.
   The decisive cross-run contrast: **grace12 — same data, same difficulty curve, but with WRITE trained — never developed the register at all** (ACT wait+hmm flat at ~1–2/1k words for the whole run vs. actonly's climb to ~7/1k; memory body essentially zero at 0.0–0.24/1k vs. actonly's 0.04→3.7/1k). Data difficulty alone therefore cannot be the driver; the discriminating variable is whether WRITE receives gradient. Two non-exclusive mechanisms fit: **(A) contamination loop** — the untrained WRITE leaks think-aloud into memory (register bleed plus unclosed-thinking spills), contaminated memory sits in every subsequent ACT context, rollouts sampled in such contexts lean slightly further into the register (measured per-trial effect is small, +0–9%, but it compounds through the training loop), and the reinforced rollouts push the weights further; **(B) style anchor** — WRITE's own gradient continuously trains the shared weights to produce clean structured notes, acting as an implicit text-style regularizer whose benefit spills over to ACT. Present data cannot separate A from B (the zero-memory position-0 trials also drift, which is consistent with both once the effect has reached the weights); the babel v3curr run (WRITE trained, official reward) is the natural discriminating control. Either way, the practical reading is the same: **training the WRITE channel doubles as a cheap anti-drift anchor for the shared policy, and ACT-only training removed it.**

**Step 2 — Length hits the 2,500-token response cap; quantitative creep becomes qualitative breakage (measured).** The cap is a cliff, not a damper: overflow → truncation → unclosed code fence → the turn yields no command. Oversized turns went 36% → 70% across r90–99, and wasted turns appeared at scale. First feedback loop: wasted turns → longer trials → step-discount losses → higher failure rate.

**Step 3 — Tier switch removes the positive anchors; negative-gradient updates squeeze probability mass into degenerate modes (timing measured; mechanism inferred).** Verified boundary: r99 = lines 198/199 (last pure easy_mid), **r100 = lines 200/201 (half mixed)**, r101+ = fully mixed. Reward halves exactly at r100 (0.145 → 0.075) and the repetition explosion lands within 1–2 rollouts of the boundary.
   The mechanics of the squeeze: a negative-advantage update pushes down the probability of *every token* in a failed trajectory — including its perfectly healthy actions (grep, cat, well-formed bash blocks); negative gradient is indiscriminate. Softmax normalization conserves probability, so whatever is pushed down is redistributed to *unsampled* tokens in proportion to the model's prior — and the fattest nearby prior mode of an LM is verbatim repetition. Every update on failures therefore quietly subsidizes repetition/template modes that have never been sampled and hence never punished. In healthy phases this leak is repaired by the positive gradient flowing through successful trajectories (30–45% success mid-run — which is why the r75–81 episode self-recovered). The mixed tier cut success to ~4%: within a group of 8, either all fail (zero advantage, no signal) or one lucky success carries all positive signal while seven failures spread negative signal over mostly-normal behavior. Demolition without construction: probability drains from healthy behavior into the repetition prior until repetition becomes sampleable — at which point in-context self-excitation (repetition begets repetition conditioned on its own output) floods the trajectory. That is the observed discrete jump.

**Step 4 — Junk is reinforced with positive sign (measured).** Late successful trials themselves contain ~50% wasted/repetitive turns (see the "Residual capability" case). GRPO credits the entire trajectory, so every surviving success also directly teaches "repetition is compatible with success." Repetition is not merely leaked into — it is actively fed.

**Step 5 — KL cannot rescue it (inferred).** Beyond the small coefficient (0.01), KL is per-token and conditioned on the sampled prefix: the reference model, shown a context full of "OK! OK! OK!", also predicts "OK!" with high probability. KL mildly penalizes *entering* the repetition basin and almost nothing for *staying* in it.

Why a phase transition rather than a drift: feedback loops from Steps 2–4 run concurrently, and the r100 tier boundary is an exogenous step input that removed the restoring force overnight — the system was already hollowed out by the r90–99 length inflation, and the switch pushed it over the critical point with essentially no delay.

**Counterfactual check**: smith-v2 (v2 randomly-mixed data) had the same kind of degeneration episode (r84–104) but *terminally recovered* — random mixing keeps solvable instances flowing, so positive anchors never vanish. Testable prediction: relaunching actonly from ~iter_95 with ~30% easy data interleaved into the tail would avoid the collapse. This is the single strongest piece of evidence that the curriculum's hard-tail concentration is the enabling condition, with the response-cap cliff and the repetition attractor as amplifiers and length inflation as the slow fuse.

## 6. Checkpoint recommendation for evaluation

- **Do not use iter_119** (fully collapsed) — and **not iter_95 either**: r93–96 is already deep in the runaway-verbosity phase (4.9–5k chars/turn, 56–61% oversized turns).
- Pre-eval reasoning favored iter_87 (last checkpoint with a healthy generation profile: 1,431 chars/turn, 12.6% oversized, zero repetition) with iter_79 as control.
- **19q eval verdict (2026-07-29): iter_63 wins for memory-mode use; the generation-profile heuristic was insufficient.**

| | replace (19q×5, memory) | stateless (avg@4) | memory contribution |
|---|---|---|---|
| iter_63 | 26.3%, cumReward 3.715 | 19.7% | **+6.6pp** |
| iter_87 | 17.9%, cumReward 2.990 | 22.4% | **−4.5pp** |

  Between r63 and r87, bare capability still rose (+2.7pp stateless) while memory-conditioned performance collapsed (−8.4pp replace) and the memory contribution flipped sign — replicating the smith-v2 flip (+4.8pp@47 → −3.9pp@119) in a run whose WRITE was never trained. The flip coincides with the register bleed into memory (§6b) and the model's own abandonment of in-text memory citations. Practical rule: a clean generation profile does not guarantee the memory-conditioned regime is intact; evaluate both modes before picking a checkpoint.

## 6b. Pre-collapse drift profile (r40–70, the KL-rise window)

Full-population scans plus 90 deep-read trajectories/memories over r10–25 vs r40–55 vs r56–70 characterize what the steadily rising train/kl_loss was actually measuring — three concurrent strands:

1. **Think-aloud register growth (the main KL driver, initially benign).** Fine-grained onset: pruned to below-prior levels during early compression (r6–18), rebound from r22, acceleration from r46. Not templating: n-gram concentration and cross-turn similarity *fall*; the text becomes longer and more diverse. Reading confirms genuine quality gains mid-window (evidence-based tracing, explicit self-correction) — the drift was paid for by reward. The register also seeps across channel boundaries: into ACT text (which has no thinking mode) and, in actonly only, into the memory body (wait+hmm 0.04→3.7/1k; memory becomes nearly as ruminative as the thinking channel itself, 3.5 vs 4.0–4.6/1k by r50–66). In-context trigger imitation was tested and rejected as the transmission path (ACT trials with wait-containing memory show the same output density as trials with clean memory; zero-memory trials drift too) — transmission is through the weights, per §5b Step 1.
2. **Behavioral canalization under step-cost pressure.** Localization degrades from search to full-file reading (grep as first action 24%→0.4%); editing collapses from precise sed lines (98.5%→22.8%) to a universal heredoc-rewrite hammer (13%→83%); verification, briefly self-taught in r40–55 (3.1%→16.4%, pytest first appears), is pruned again by r56–70 (10.2%) as the three-step combo VIEW→EDIT→SUBMIT re-concentrates (top-3 pattern coverage 28%→53%). Fallback diversity disappears (same-tool 9-hit retry streaks replace strategy switches). A persistent "fake-success belief" runs through all windows — sed exit-0 as proof, later the model's own script printing "Fixed!" read as external confirmation — and lessons about it are repeatedly *written to memory without changing behavior* (the knowledge–behavior dissociation of an untrained WRITE channel; ACT-side memory-citation language simultaneously drops to zero).
3. **Output-form decay, the strand continuous with the collapse.** Empty-command rate accelerates monotonically 0.90%→4.52%→7.70%, including multi-turn "plan written in THINK, command field empty" paralysis inside otherwise healthy trials; WRITE-side truncation grows 1.6%→14% with a bimodal length distribution (median stable, p90 doubling — a fattening bad tail, later the main distribution); late-window instability markers appear at low frequency (fabricated names, cross-lingual tokens, failure-rationalizing lessons).

Composite reading: the KL rise priced a mixture of genuine capability (strand 1's tracing), quiet capability narrowing (strand 2), and the earliest form of the collapse pathology (strand 3) — masked throughout by a rising easy-tier success plateau. Alarm-worthy leading indicators, all 20+ rollouts ahead of the crash: empty-command rate, wait/hmm density, chars/turn, WRITE truncation rate, first-sentence dedup rate.

## 7. Implications for the next run (v5)

1. **Monitor chars/turn as a first-class metric.** It led the crash by 10+ rollouts. An alert threshold around 2× the healthy baseline (~1.7k chars/turn for this model) would have fired at r90.
2. **Make format validity part of the reward.** Unparseable responses currently cost a turn but carry no penalty; a small explicit penalty (or making turns-without-commands count against the step budget) removes the free-fall region.
3. **Length regularization on reasoning** (cap or penalize response length directly) — the 2,500-token response cap currently acts as a cliff (truncation → broken fence → wasted turn) rather than a deterrent.
4. **Fix the curriculum tail.** From ~r111 the mixed tier produced multiple all-zero rollouts — no learning signal, pure drift exposure. Either dilute the tail with solvable instances (maintain reward variance / SNR), stop early on a heldout trigger, or interleave tiers instead of hard-concatenating them.
5. **Transients warn before terminal collapse.** Both this run (r75–81) and smith-v2 (r84–104) had self-recovering degeneration episodes before/without terminal collapse. A second episode that does not recede within ~10 rollouts is the signature to stop or intervene on.
6. **Keep gradient on the WRITE channel even when ACT is the target.** The grace12 contrast (§5b Step 1) indicates WRITE training functions as a cheap style/drift anchor for the shared weights — dropping it opened actonly's path to the verbosity collapse. If WRITE competence itself is not wanted, a small-weight WRITE loss (or any auxiliary supervised text objective on the shared weights) may buy the anchoring without distorting ACT training.
