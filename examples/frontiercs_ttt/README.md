## FrontierSmith group dataset

The same complete-group rollout and official Frontier-CS judge can train on
the curated FrontierSmith subset. The dataset contains 38 groups of five tasks
(190 task instances; ten exact Frontier-CS parity duplicates are excluded).
The prepared task package must be present under
`$FRONTIERCS_ROOT/qwen_eval/frontiersmith_dataset`; it contains the statements,
official checkers, hidden tests, curated group manifest, and helper scripts.

Generate the Miles-compatible group JSONL and expose its official checkers and
hidden tests to the ordinary Frontier-CS judge tree:

```bash
python3 "$FRONTIERCS_ROOT/qwen_eval/frontiersmith_dataset/build_miles_groups.py"
python3 "$FRONTIERCS_ROOT/qwen_eval/frontiersmith_dataset/install_training_problems.py" "$FRONTIERCS_ROOT"
```

The installer creates symlinks only for the 190 selected tasks under
`$FRONTIERCS_ROOT/algorithmic/problems`; it refuses to replace any existing
directory or conflicting symlink. Each task provides `statement.txt`,
`config.yaml`, `chk.cc`, and the original `testdata` directory. Consequently,
both the rollout's statement loader and the existing HTTP judge use the
official FrontierSmith evaluator without a second judge implementation.

Select the dataset when launching the standard complete-episode training
entrypoint:

```bash
export FRONTIERCS_PROMPT_DATA="$FRONTIERCS_ROOT/qwen_eval/frontiersmith_dataset/miles_problem_groups_38.jsonl"
export FRONTIERCS_GROUP_SIZE=5
export FRONTIERCS_RUN_ID=frontiersmith-g38
bash "$MILES_ROOT/examples/frontiercs_ttt/run_frontiercs_ttt_episode_qwen3.5_4B.sh"
```

All other rollout, advantage, optimizer, and judge settings remain configurable
through the existing Frontier-CS variables. The rollout now accepts the
non-numeric `frontiersmith_N` task IDs used by these groups. The group's
`shared_thread` annotation is retained in prompt metadata for traceability;
the task statements themselves remain the solver's problem inputs.

# Frontier-CS group-memory training in Miles

This directory contains the Frontier-CS rollout and training integration for
Miles. It is intentionally separate from the existing codebase-adaptation task.

The normal setup uses two Git repositories:

- the Miles branch contains the training code and the problem-group JSONL;
- the Frontier-CS branch contains problem statements, hidden test data, and the
  judge implementation.

There is no separate dataset checkout step. Clone both repositories at the
revisions used by the experiment.

## 1. Clone the two repositories

Clone the two published `frontiercs-ttt` branches:

```bash
git clone --branch frontiercs-ttt https://github.com/Racktic/miles.git \
  /home/your-user/miles
git clone --branch frontiercs-ttt https://github.com/Racktic/Frontier-CS.git \
  /home/your-user/Frontier-CS

export MILES_ROOT=/home/your-user/miles
export FRONTIERCS_ROOT=/home/your-user/Frontier-CS
```

For an exact commit rather than a branch, clone the repository normally and
then run `git checkout <commit>` in that repository.

## 2. Training data

Both required data inputs are already supplied by the two clones:

| Input | Location | Purpose |
|---|---|---|
| Group dataset | `${MILES_ROOT}/examples/frontiercs_ttt/data/problem_groups_30.jsonl` | Thirty groups, three problem IDs per group |
| Benchmark data | `${FRONTIERCS_ROOT}/algorithmic/problems` | Statements, configs, checkers, and hidden test cases |

The group JSONL is small and is committed to the Miles branch. The benchmark
data is committed to the Frontier-CS branch. No Hugging Face dataset download
is required for this task.

Verify both inputs before starting a GPU allocation:

```bash
test -f "${MILES_ROOT}/examples/frontiercs_ttt/data/problem_groups_30.jsonl"
test -d "${FRONTIERCS_ROOT}/algorithmic/problems"
```

Use `FRONTIERCS_PROMPT_DATA` only when intentionally selecting a different
group file, such as the smoke-test dataset.

## 3. Install Miles and prepare the model

Use the normal Miles environment or container described by the top-level Miles
documentation. For a source installation:

```bash
cd "${MILES_ROOT}"
pip install -r requirements.txt
pip install -e .
```

The launcher requires two copies of the model:

1. the Hugging Face checkpoint used by the rollout engine;
2. the Megatron `torch_dist` checkpoint used by the trainable actor and
   reference model.

For Qwen3.5-4B:

```bash
export FRONTIERCS_MODEL_ROOT=/home/your-user/models
export MEGATRON_LM_PATH=/home/your-user/Megatron-LM

hf download Qwen/Qwen3.5-4B \
  --local-dir "${FRONTIERCS_MODEL_ROOT}/Qwen3.5-4B"

cd "${MILES_ROOT}"
source scripts/models/qwen3.5-4B.sh
PYTHONPATH="${MEGATRON_LM_PATH}" python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${FRONTIERCS_MODEL_ROOT}/Qwen3.5-4B" \
  --save "${FRONTIERCS_MODEL_ROOT}/Qwen3.5-4B_torch_dist"
```

Do not commit model weights to either repository.

For Qwen3.6-27B, use the corresponding Miles model configuration during
conversion:

```bash
export FRONTIERCS_MODEL_ROOT=/home/your-user/models
export MEGATRON_LM_PATH=/home/your-user/Megatron-LM

hf download Qwen/Qwen3.6-27B \
  --local-dir "${FRONTIERCS_MODEL_ROOT}/Qwen3.6-27B"

cd "${MILES_ROOT}"
source scripts/models/qwen3.6-27B.sh
PYTHONPATH="${MEGATRON_LM_PATH}" python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${FRONTIERCS_MODEL_ROOT}/Qwen3.6-27B" \
  --save "${FRONTIERCS_MODEL_ROOT}/Qwen3.6-27B_torch_dist"
```

## 4. Configure the judge

Training communicates with the judge over HTTP. The only value required by the
rollout workers is `FRONTIERCS_JUDGE_URL`.

### Recommended: independent Docker judge

Run the judge on a host that supports privileged Docker:

```bash
cd "${FRONTIERCS_ROOT}/algorithmic"
docker compose up --build -d
curl -fsS http://127.0.0.1:8081/health
```

Then set `FRONTIERCS_JUDGE_URL` to the address reachable from every Ray worker.
If the GPU cluster does not allow privileged containers, run this service on a
separate machine and expose its port only to the training cluster.

The Docker image installs Node.js, C++ tooling, and the pinned go-judge runtime.
It mounts `${FRONTIERCS_ROOT}/algorithmic/problems`, so it evaluates the test
data from the checked-out Frontier-CS revision.

### Fallback: host judge on the training node

Use this only when the compute node supports the cgroup/namespace operations
required by go-judge. Install Node.js 20, C++17, go-judge, and go-judge-init,
then install the Node dependencies:

```bash
cd "${FRONTIERCS_ROOT}/algorithmic"
npm ci
```

Set `FRONTIERCS_AUTO_START_JUDGE=1`. If the executables are not on `PATH`, also
set `FRONTIERCS_GOJUDGE_BIN`, `FRONTIERCS_GOJUDGE_INIT`, and
`FRONTIERCS_NODE_BIN`. The launcher uses `host_judge.py` to start and stop an
isolated local service; it does not modify the Frontier-CS checkout.

## 5. Configure the run

Copy the environment template to a private machine-specific file and edit it:

```bash
cp "${MILES_ROOT}/examples/frontiercs_ttt/frontiercs_env.example" \
  /home/your-user/frontiercs.env
```

The required paths are:

| Variable | Meaning |
|---|---|
| `FRONTIERCS_ROOT` | Frontier-CS repository clone |
| `FRONTIERCS_HF_CHECKPOINT` | Hugging Face model directory |
| `FRONTIERCS_TORCH_DIST` | Converted Megatron checkpoint directory |
| `FRONTIERCS_OUTPUT_ROOT` | Rollout traces and environment state |
| `FRONTIERCS_SAVE_DIR` | Training checkpoints |
| `FRONTIERCS_JUDGE_URL` | Healthy judge HTTP endpoint |

The current full experiment settings are:

| Variable | Default | How to choose it |
|---|---:|---|
| `FRONTIERCS_GROUP_SIZE` | `3` | Must match the number of problems in every JSONL group |
| `FRONTIERCS_GROUPS_PER_UPDATE` | `2` | Complete group episodes collected before one optimizer step |
| `FRONTIERCS_MEMORY_ROUNDS` | `4` | Number of solve/evaluate rounds in each episode |
| `FRONTIERCS_CANDIDATES_PER_PROBLEM` | `4` | Independent candidates per problem per round |
| `FRONTIERCS_ACT_ADVANTAGE_MODE` | `group_relative` | Round-local GRPO over the four identical-prompt candidates |
| `FRONTIERCS_TRAIN_WRITE` | `1` | Set to `0` to generate/use memory but exclude WRITE responses from optimization |
| `FRONTIERCS_WRITE_CANDIDATES` | `4` | Sibling memories sampled from each identical WRITE prompt |
| `FRONTIERCS_WRITE_ADVANTAGE_MODE` | `grpo` | GRPO over sibling memories using equal-weight per-problem raw-score percentiles |
| `FRONTIERCS_ACT_MAX_NEW_TOKENS` | `25600` | Maximum problem-solving response length |
| `FRONTIERCS_WRITE_MAX_NEW_TOKENS` | `25600` | Maximum memory response length |
| `FRONTIERCS_SEQ_LENGTH` | `32768` | Maximum trainable prompt-plus-response sequence |
| `FRONTIERCS_KL_MODE` | `loss` | Use `loss` for KL loss or `none` to disable it |
| `FRONTIERCS_KL_COEF` | `0.01` | KL-loss coefficient |
| `FRONTIERCS_KL_TYPE` | `k1` | Miles KL estimator: `k1`, `k2`, `k3`, or `low_var_kl` |
| `FRONTIERCS_LR` | `1e-6` | Full-parameter actor learning rate |
| `FRONTIERCS_ACT_EXPLORE_BETA` | `0` | Weight of the normalized memory-delta exploration advantage; positive enables the external judge |

Keep `FRONTIERCS_GROUP_SIZE=3` for the supplied 30-group dataset. The default
`FRONTIERCS_GROUPS_PER_UPDATE=2` means that one epoch over 30 groups contains 15
optimizer steps. Increase this value only when rollout time and memory permit.

Hardware parameters are cluster-specific:

| Variable | Default |
|---|---|
| `FRONTIERCS_NGPU` | Number of visible GPUs |
| `FRONTIERCS_TP` | `FRONTIERCS_NGPU` |
| `FRONTIERCS_PP` | `1` |
| `FRONTIERCS_CP` | `1` |
| `FRONTIERCS_ACTOR_NUM_NODES` | `1` |
| `FRONTIERCS_ACTOR_GPUS_PER_NODE` | `FRONTIERCS_NGPU` |
| `FRONTIERCS_ROLLOUT_NUM_GPUS` | Total actor GPUs |
| `FRONTIERCS_ROLLOUT_GPUS_PER_ENGINE` | `1` |
| `FRONTIERCS_SGLANG_MEM_FRACTION` | `0.5` |

Before submitting training, the launcher waits until Ray contains the requested
number of nodes and GPUs. An incomplete allocation therefore fails with an
explicit timeout instead of hanging later while Miles creates its placement
group. Set `FRONTIERCS_WAIT_FOR_RAY_CLUSTER=0` only when readiness is validated
externally. By default it also verifies the repositories, data, model inputs,
and output directories on every eligible GPU node.

W&B is optional. Set `WANDB_PROJECT=miles-frontier-cs` and provide the API key
through the process environment or `FRONTIERCS_WANDB_ENV_FILE`. Never commit the
key. Set `FRONTIERCS_USE_WANDB=0` to disable logging.

### Memory-delta exploration reward

Set `FRONTIERCS_ACT_EXPLORE_BETA` to a positive value to enable exploration
shaping. The default is zero, so existing runs do not make external judge
calls. Configure the judge with:

```bash
export FRONTIERCS_ACT_EXPLORE_BETA=0.3
export FRONTIERCS_EXPLORE_JUDGE_API_KEY=...
export FRONTIERCS_EXPLORE_JUDGE_API_BASE=https://api.openai.com/v1
export FRONTIERCS_EXPLORE_JUDGE_MODEL=gpt-5-mini
```

After every ACT round, including the final round, the rollout generates an
updated memory. The exploration judge receives exactly the previous memory and
the updated memory; it does not receive problem statements, candidate code,
task scores, stderr, or other diagnostics. It assigns integer scores in
`{0, 1, 2}` for new discoveries, error correction, actionable knowledge, and
high-level abstraction. Their sum divided by eight is the raw exploration
score in `[0, 1]`.

The active judge is V4. Its memory-pair serialization matches the existing
codebase-adaptation judge (`Previous memory M_(k-1)` followed by `Updated memory
M_k`). An untested proposed action may earn credit, but it must be specific and
testable. The prompt version and full prompt hash are persisted in every episode
manifest and exploration result.

The score for a group round is attached to every ACT candidate generated in
that round. Advantages are then standardized separately for each problem in
one complete episode across its `S*K` candidates. Candidates from different
problems in the same episode and candidates from different episodes are never
mixed for exploration normalization. The terminal
memory exists only to score the final ACT round: it is saved in the trace but
is not returned as a WRITE training sample and has no WRITE loss.

## 6. Launch

Enter the GPU allocation, activate the Miles environment, source the private
configuration, and run the complete-episode launcher:

```bash
source /home/your-user/frontiercs.env
cd "${MILES_ROOT}"
bash examples/frontiercs_ttt/run_frontiercs_ttt_episode_qwen3.5_4B.sh
```

For Qwen3.6-27B, set its two checkpoint paths and use the model-specific
entrypoint. It defaults to `TP=4`; PP, CP, and the node counts remain explicit
hardware parameters:

```bash
export FRONTIERCS_HF_CHECKPOINT=/home/your-user/models/Qwen3.6-27B
export FRONTIERCS_TORCH_DIST=/home/your-user/models/Qwen3.6-27B_torch_dist
bash /home/your-user/miles/examples/frontiercs_ttt/run_frontiercs_ttt_episode_qwen3.6_27B.sh
```

The launcher verifies the group dataset, model directories, visible GPUs, and
judge health before submitting the Ray job. Slurm directives, container mounts,
and cluster module commands belong in a small cluster-specific wrapper rather
than in the shared launcher.

### Multi-node launch

`run_frontiercs_ttt_multinode.sh` is the scheduler-independent multi-node
entrypoint. Every process uses the same head address. Set the total number of
nodes and per-node GPUs on every node before starting it:

```bash
export FRONTIERCS_ACTOR_NUM_NODES=4
export FRONTIERCS_ACTOR_GPUS_PER_NODE=8
export FRONTIERCS_NGPU=8
export FRONTIERCS_TP=8
export FRONTIERCS_PP=2
export FRONTIERCS_CP=1
```

Start the head role on the designated head node:

```bash
source /home/your-user/frontiercs.env
bash /home/your-user/miles/examples/frontiercs_ttt/run_frontiercs_ttt_multinode.sh \
  head 10.0.0.10
```

The Qwen3.6-27B entrypoint accepts the same role and address, so it can be used
directly on the head:

```bash
bash /home/your-user/miles/examples/frontiercs_ttt/run_frontiercs_ttt_episode_qwen3.6_27B.sh \
  head 10.0.0.10
```

Start one worker role on each of the other nodes:

```bash
source /home/your-user/frontiercs.env
bash /home/your-user/miles/examples/frontiercs_ttt/run_frontiercs_ttt_multinode.sh \
  worker 10.0.0.10
```

Use the matching Qwen3.6-27B entrypoint on every worker when training that
model:

```bash
bash /home/your-user/miles/examples/frontiercs_ttt/run_frontiercs_ttt_episode_qwen3.6_27B.sh \
  worker 10.0.0.10
```

Workers wait for the Ray head and monitor it after joining. The head waits until
all four nodes advertise eight GPUs each and is the only process that submits
the training job. When that process finishes or fails, it stops the Ray head;
workers detect the closed head and clean up their local Ray services. Miles
receives 32 colocated actor and rollout GPUs. The rollout manager is pinned to
the Ray head so the Frontier-CS episode state and a head-local judge have a
single, deterministic owner.

For a Ray cluster that the scheduler has already created, invoke the normal
launcher with `FRONTIERCS_START_RAY=0`, set the dashboard endpoint in
`FRONTIERCS_RAY_ADDRESS`, and set the GCS endpoint in
`FRONTIERCS_RAY_CLUSTER_ADDRESS`. The same resource-readiness check runs before
job submission.

The launcher also verifies that the samples in one optimizer update are exactly
divisible across data-parallel ranks. Each default `G=3`, `K_act=4`, `S=4`,
`K_write=4` episode contains 60 trainable samples, so the check is
`FRONTIERCS_GROUPS_PER_UPDATE * 60` divisible by
`DP = total_GPUs / (TP * PP * CP)`. It fails before allocating model actors
rather than letting Miles silently trim complete-episode samples. TP, PP, or
the number of groups per update can be adjusted to satisfy this invariant.

## 7. Training semantics

One optimizer unit is a batch of complete group episodes. Model weights remain
frozen while one group passes through all memory rounds:

1. In each round, generate `G*K_act` solutions from the current shared memory.
2. Submit every extracted C++ solution to the Frontier-CS judge.
3. Sample `K_write` memories from one prompt containing the previous memory,
   candidate code, and evaluator feedback.
4. Evaluate every memory with next-round ACT, train all sibling WRITE samples,
   and keep only the highest-scoring memory branch. Non-winning downstream ACT
   samples are evaluator-only and do not enter ACT loss.
5. After all `S` rounds finish, return the `G*K_act*S` main-path ACT samples
   and `K_write*(S-1)` WRITE samples, then update the model.

With the default `G=3`, `K_act=4`, `S=4`, and `K_write=4`, one episode
contains 48 main-path problem-solving samples and 12 memory samples. With two
episodes per update, an optimizer step receives 120 samples.

For a solution-only optimization ablation, set `FRONTIERCS_TRAIN_WRITE=0`,
`FRONTIERCS_WRITE_ADVANTAGE_MODE=direct`, and
`FRONTIERCS_WRITE_CANDIDATES=1`. The pipeline still generates one nonterminal
memory and supplies it to later rounds; it simply returns no WRITE samples to
Miles. With `G=3`, `K_act=4`, and `S=4`, each episode then contains 48
trainable solution samples. If exploration reward is enabled, the terminal
memory is still generated and all four solution rounds receive memory-delta
exploration scores.

Problem-solving advantages default to `group_relative`. Each
`(episode, problem, round)` forms an independent GRPO group containing exactly
its four identical-prompt ACT candidates. Memory samples are never normalized
together with problem-solving samples.

### K-way memory GRPO

The default sets `FRONTIERCS_WRITE_ADVANTAGE_MODE=grpo` and
`FRONTIERCS_WRITE_CANDIDATES=4`. After each
nonterminal solution round, the rollout samples `K_write` memories from the
same prompt. Every memory conditions a paired-seed evaluation of the next
round's `G*K_act` solutions. For each problem, the rollout averages that
branch's raw scorer values across its `K_act` solutions, orients them so higher
is always better, and ranks the `K_write` sibling branches. The rank is mapped
to `[0,1]`; the group reward is the mean of these per-problem percentiles, so
every problem contributes equally regardless of its scorer's units. Invalid
submissions rank below valid ones; ties receive their average rank.

The `K_write` rewards from the same episode and writer round are standardized
with sample standard deviation for the WRITE loss. All memory candidates enter
the memory loss, but only the highest group reward memory and its downstream
solution branch continue on the main path. Non-winning downstream solutions
are evaluator-only and do not enter the solution loss. Ties select the lower
memory-candidate index for deterministic replay.

ACT advantages use each task's raw `score_unbounded` (falling back to the
bounded judge score only when absent), with minimize objectives sign-flipped;
the `K_act` candidates for the same problem and round are then GRPO-normalized.
The bounded 0--100 score remains in traces and score metrics for readability,
not as the ACT advantage signal.

Exploration reward still judges only the selected main-path memory delta. After
the final solution round, exactly one terminal memory is generated for its
exploration reward and is not returned as a memory training sample. Thus:

```text
trainable samples / episode = G*K_act*S + K_write*(S-1)
ACT evaluations / episode  = G*K_act*(1 + K_write*(S-1))
```

For the default `G=3`, `K_act=4`, `S=4`, and `K_write=4`, this is 60
trainable samples (48 main-path solution and 12 memory samples) and 156
solution evaluations. The 108 ACT samples behind non-winning memories are
evaluator-only and do not enter the solution loss. The model is updated only
after the complete branching episode finishes.

## 8. Outputs and recovery

All inspectable rollout state is stored under
`${FRONTIERCS_OUTPUT_ROOT}/${FRONTIERCS_RUN_ID}`. Model checkpoints are stored
under `FRONTIERCS_SAVE_DIR`.

Each episode records its prompts, raw generations, extracted reasoning, C++
code, evaluator feedback, memory, packed training samples, and an atomic episode
commit. A completed episode is replayed from disk after restart instead of being
sampled and judged again. Do not reuse a run ID with a different group, round,
or candidate configuration.

The W&B dashboard logs numeric score, execution, failure, response-length,
memory-length, memory-change, and training-signal metrics. The `diagnostics/`
section distinguishes static stderr usage (`stderr_print_frac` and
`stderr_print_count_mean`) from actually observed non-empty judge diagnostics
(`observed_nonempty_frac`). Under
`exploration_reward/` it logs the four rubric-dimension batch means, raw
exploration reward mean, same-problem group zero-std fraction, and mean group
standard deviation. It does not upload prompts, code, diagnostics, or memory
text.

## ALE-Bench backend

The rollout can use ALE-Bench's official public evaluator instead of the
Frontier-CS checker. This code path supports batch and reactive tasks through
the official `Session.public_eval` runner. It is a local integration and does
not change the default Frontier-CS backend.

On the judge node, install the official ALE-Bench package, ensure Docker is
available, and point `ALE_BENCH_DATA` to a folder containing the official
problem ZIPs and ID lists. Then start the authenticated service:

```bash
export PYTHONPATH="${FRONTIERCS_ROOT}:${PYTHONPATH}"
export ALE_BENCH_DATA="${FRONTIERCS_ROOT}/qwen_eval/ale_bench_dataset/upstream"
export ALE_BENCH_JUDGE_TOKEN="$(openssl rand -hex 32)"
export ALE_BENCH_JUDGE_HOST=127.0.0.1
python "${MILES_ROOT}/examples/frontiercs_ttt/frontiercs_ale_judge.py" --serve
```

The service binds to localhost by default. For a separate judge node, explicitly
bind it to a cluster-private interface and use the same `ALE_BENCH_JUDGE_TOKEN`
only in trusted Miles workers; firewall the port to the training nodes. The
service executes submitted programs, so do not expose it to the public network.
Limit judge load with `ALE_BENCH_MAX_CONCURRENT_EVALUATIONS` (default 4) and
`ALE_BENCH_NUM_WORKERS` (default 4 workers per evaluation).

On every Miles worker, set:

```bash
export FRONTIERCS_ALE_JUDGE_URL="http://<judge-private-address>:8082"
export ALE_BENCH_JUDGE_TOKEN="<same-secret>"
export ALE_BENCH_CATALOG="${FRONTIERCS_ROOT}/qwen_eval/ale_bench_dataset/ale_bench_task_catalog_40.jsonl"
export FRONTIERCS_PROMPT_DATA="${FRONTIERCS_ROOT}/qwen_eval/ale_bench_dataset/miles_problem_groups_8x5.jsonl"
export FRONTIERCS_GROUP_SIZE=5
export FRONTIERCS_RUN_ID=ale-bench-g8
```

ALE-Bench uses the full public seeds for each of its 40 tasks (50 per task in
this release). The judge returns a log-scaled, direction-aware 0–100 training
utility and separately records the raw public score; this transform is not the
official leaderboard score. `ALE_BENCH_SCORE_REFERENCE` defaults to 1e9 and
should be sanity-checked against pilot results before large runs. Private seeds
and `private_eval` are not used. Image input is optional and disabled by
default. Set `FRONTIERCS_USE_STATEMENT_IMAGES=1` to include extracted statement
figures in the model prompt; use a model/processor with vision support and make
the catalog's `extracted/<task>/images` assets available on every rollout
worker. Ensure `qwen-vl-utils` is installed in the Miles environment. The flag
has no effect on Frontier-CS or FrontierSmith statements.

The ALE judge still records a global log-transformed 0--100 score for readable
traces and score metrics, but the training advantages now use raw public scorer
values: direction-corrected raw scores for ACT, and per-problem sibling ranks
before forming WRITE group rewards. This avoids comparing raw score units
across unrelated problems. It is still not ALE-Bench's official final metric:
official evaluation uses private-test totals to rank against contest
participants, maps each task rank to performance, and reports mean performance
across tasks. Private scores and standings are not used during training.

## Training from multiple data sources in one run

The episode rollout routes each problem to its own judge from row metadata, so
one Miles prompt stream can contain Frontier-CS, FrontierSmith, and ALE-Bench
groups. The launcher accepts a semicolon-separated list of `DATASET=JSONL`
entries through `FRONTIERCS_PROMPT_DATA_SOURCES`; it merges them into a
run-specific JSONL, round-robin interleaves source rows, annotates each row's
`metadata.dataset`, and namespaces `group_id` to avoid collisions. Each source
file's rows are included exactly once, so source proportions follow the number
of groups supplied by each file. Existing `problem_datasets` metadata is
preserved for intentionally mixed-source groups.

All groups in one run must have the same number of problems because
`FRONTIERCS_GROUP_SIZE` is a single rollout parameter. The currently prepared
FrontierSmith and ALE-Bench group files both use five problems per group; use a
Frontier-CS group file with the same group size for a combined run.

```bash
export FRONTIERCS_PROMPT_DATA_SOURCES="frontier_cs=/path/to/fcs_groups.jsonl;frontiersmith=${FRONTIERCS_ROOT}/qwen_eval/frontiersmith_dataset/miles_problem_groups_38.jsonl;ale_bench=${FRONTIERCS_ROOT}/qwen_eval/ale_bench_dataset/miles_problem_groups_8x5.jsonl"
export FRONTIERCS_GROUP_SIZE=5
export FRONTIERCS_ALE_JUDGE_URL="http://<judge-private-address>:8082"
export ALE_BENCH_JUDGE_TOKEN="<same-secret>"
export ALE_BENCH_CATALOG="${FRONTIERCS_ROOT}/qwen_eval/ale_bench_dataset/ale_bench_task_catalog_40.jsonl"
# Optional: include ALE statement figures for vision-capable model runs.
export FRONTIERCS_USE_STATEMENT_IMAGES=1
bash "${MILES_ROOT}/examples/frontiercs_ttt/run_frontiercs_ttt_episode_qwen3.5_4B.sh"
```

Alternatively, create the merged file explicitly with
`python3 ${MILES_ROOT}/examples/frontiercs_ttt/merge_group_datasets.py` and
repeated `--source DATASET=JSONL` arguments, then set
`FRONTIERCS_PROMPT_DATA` to the result. Set either `FRONTIERCS_PROMPT_DATA` or
`FRONTIERCS_PROMPT_DATA_SOURCES`, not both. The source labels accepted are
`frontier_cs`, `frontiersmith`, and `ale_bench` (with common aliases).
