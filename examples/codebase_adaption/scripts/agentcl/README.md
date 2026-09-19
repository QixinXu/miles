# AgentCL 代码任务（agentcl_code）使用说明

本目录用于在我们的 ACT/WRITE 框架下，跑 AgentCL（arXiv 2606.02461）的 CodeEval-Pro 代码任务流。
评测对象有两类：一类是 miles 训练出来的 ckpt，另一类是 OpenAI 等 API 模型。

本文依次说明：

1. 代码分布在哪里
2. 在新服务器上怎么把环境准备好
3. 怎么跑评测
4. 各项指标和参数的含义
5. 已知的数据问题和常见坑

---

## 1. 代码分布

实现分散在两个仓库，另外依赖两个镜像。

| 位置 | 文件 | 作用 |
|---|---|---|
| continual-learning-bench（clbench） | `src/tasks/agentcl_code/task.py` | 任务本体：加载题目、沙箱执行 bash、判分、reward 计算 |
| clbench | `src/tasks/agentcl_code/prompts.py` | ACT 的 instance 模板，由 clbench codebase_adaptation 的模板改写而来 |
| clbench | `data/agentcl_code/*.json` | 题目数据，来自 HF `osunlp/AgentCL` 的 codeeval-pro 子集 |
| miles | `examples/codebase_adaption/codebase_rollout.py` | rollout 接入。episode 的 metadata 里 `task_family: agentcl_code` 就走这个任务；另外支持 `skip_write`、`initial_memory` 两个评测专用字段 |
| miles | `examples/codebase_adaption/prompts.py` | WRITE prompt。AgentCL 用单独一套文本（`AGENTCL_*`），按 `task_family` 选择 |
| miles | `scripts/agentcl/gen_agentcl_eval_data.py` | 生成评测用的 episode 数据（jsonl） |
| miles | `scripts/agentcl/eval_agentcl_orchard.sh` | 用 miles 评测 ckpt 的启动脚本 |
| miles | `scripts/agentcl/run_api_agentcl.py` | 用 API 模型评测的驱动，复用 miles 的 ACT/WRITE 构造 |
| miles | `scripts/agentcl/agentcl_metrics.py` | 计算 PG/SG/GG 和各项累计指标 |
| miles | `scripts/agentcl/selfcheck.py` | 新环境自检 |
| miles | `scripts/agentcl/replay_grade.py` | 调试工具：重放某道题的全部命令，打印完整的判分报错 |
| miles | `scripts/agentcl/rebuild_clean_sif.sh` | 把评测镜像里自带的题库删掉（必须做，见第 2 节） |
| 镜像 | `bigcodebench-evaluate-v0.2.4-clean.sif` | 沙箱镜像：智能体的命令和判分都在这里跑（python 3.10，pandas 2.0.3，sklearn 1.3.1 等） |
| 镜像 | `miles_dev-*.sif` | miles 训练和推理镜像；自检和 API 驱动也在它里面运行 |

---

## 2. 新服务器环境准备

以下路径都是 orchard 上的实际路径，换服务器时按需替换。

### 2.1 代码和数据

- **两个仓库**：clone miles 和 continual-learning-bench。`CLBENCH_ROOT` 默认是 `/home/qixinx/continual-learning-bench`。
- **题目数据**：已经放在 clbench 的 `data/agentcl_code/` 里。如果要从头下载，去 HF 数据集 `osunlp/AgentCL`，把 `codeeval-pro/` 下面这四个文件放进 `data/agentcl_code/`：
  - `bigcodebench_lite_pro.{compositional,naive}.json`
  - `humaneval_pro.{compositional,naive}.json`

### 2.2 apptainer

- **安装**：集群没有系统级的 apptainer，要装用户态版本。我们用官方 `install-unprivileged.sh` 装在 `/home/qixinx/apps/apptainer`，用之前先 `source /home/qixinx/apps/apptainer/env.sh`。
- **嵌套调用**：沙箱是在 miles 容器里面再起一层 apptainer，需要让容器里也能找到 apptainer：
  ```bash
  export APPTAINERENV_PREPEND_PATH=/home/qixinx/apps/apptainer/bin
  ```

### 2.3 沙箱镜像（必须用去数据版）

```bash
source /home/qixinx/apps/apptainer/env.sh
# 临时目录一定放本地盘；放在 NFS 上解包每分钟只有约 66MB，基本跑不完
export APPTAINER_TMPDIR=<本地盘目录> APPTAINER_CACHEDIR=<本地盘或 /project 目录>
apptainer build /project/flame/qixinx/agentcl/sif/bigcodebench-evaluate-v0.2.4.sif \
    docker://bigcodebench/bigcodebench-evaluate:v0.2.4
bash scripts/agentcl/rebuild_clean_sif.sh   # 生成 ...-clean.sif，先改脚本里的路径
```

**为什么一定要去数据**：官方镜像的 `/root/.cache/bigcodebench/BigCodeBench-v0.1.4.jsonl` 里有全部参考解答和测试，HuggingFace 的数据集缓存和 `/bigcodebench` 源码也在镜像里。智能体在沙箱里执行 `find /` 就能搜到它们。`rebuild_clean_sif.sh` 会删掉这三处。

任务默认使用 `DEFAULT_SIF`，也就是去数据版；也可以通过环境变量 `AGENTCL_SIF` 或任务参数 `sif_path` 指定别的镜像。

### 2.4 Python 依赖

clbench 的代码需要以下包：litellm、mini-swe-agent（提供输出反馈模板）、jinja2、pyyaml、pydantic、openai、huggingface_hub。

我们的做法是用训练时同一份依赖目录 `/home/qixinx/miles_pydeps/codebase_py312_clean`，运行时通过 `PYTHONPATH` 挂进 miles 容器。不要往共享的 `~/.local` 里 `pip install`，那会污染整个集群。

### 2.5 模型和密钥

- **评测 ckpt**：需要 Qwen3.5-4B 的 HF 权重，以及转换后的 torch_dist。`scripts/ensure_torch_dist_local.sh` 会在节点本地准备好。
- **评测 API 模型**：`/home/qixinx/miles/.env` 里要有 `OPENAI_API_KEY`。上传 ckpt 另外需要 `HF_TOKEN`。

### 2.6 自检（新环境必做）

```bash
source /home/qixinx/apps/apptainer/env.sh
export TMPDIR=<可写目录> APPTAINERENV_PREPEND_PATH=/home/qixinx/apps/apptainer/bin
apptainer exec --bind /home/qixinx,/project/flame,<TMPDIR 所在盘> \
  --env PYTHONPATH=/home/qixinx/miles_pydeps/codebase_py312_clean,TMPDIR=$TMPDIR \
  /project/flame/qixinx/images/miles_dev-202606081341.sif \
  python3 scripts/agentcl/selfcheck.py
```

自检分三步：

1. 所有题目的初始 `solution.py` 都能被 Python 解析。
2. 沙箱是隔离的：看不到 /home、/project、/mnt，并且命令从 /testbed 开始执行。
3. 参考解答经由真实的多轮流程提交，也就是先用 heredoc 写文件，再发送提交命令，结果要和下表一致。

| 任务流 | 预期通过数 | 没通过的题 |
|---|---|---|
| bcb_compositional | 95/96 | raw/269，参考解答本身是坏的，见第 6 节 |
| humaneval_heldout | 119/120 | new/41，无解 |

最后输出 `SELFCHECK PASSED` 才说明环境没问题。全程大约 4 分钟。

---

## 3. 任务是怎么交互的

- **工作目录**：每道题有一个全新的工作目录，挂载为 `/testbed`，里面只有一个 `solution.py`。它的内容是：
  - `from typing import *`
  - 原题面，也就是函数说明和签名
  - `raise NotImplementedError` 占位函数体
- **ACT 每一轮**：输出推理文字，再输出一个 ```bash 代码块，里面只能有一条命令。命令在沙箱镜像里执行。
  - 只有 `/testbed` 可写，而且内容会跨轮保留。
  - 当前目录、环境变量和 `/tmp` 每条命令都会重置。
  - 单条命令超时 60 秒。
- **提交**：执行 `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`。系统取出当前的 `solution.py`，在一个新沙箱里拼上隐藏测试运行，超时 120 秒。退出码为 0 就算通过。隐藏测试从不进入智能体的工作目录。
- **步数**：每题最多 40 步。空命令不算步数。
- **reward**：与 codebase_adaption 相同，`reward = 1 − regret/40`。做对时 `regret = 步数 − 1`，做错时 `regret = 40`。另外单独记录每题的对错，用来算 AgentCL 口径的正确率。
- **WRITE**：每做完一道题调用一次。输入包括这道题的完整 ACT 对话、最后的判分反馈（只有 PASSED 或 FAILED 这一档，不给报错细节）和上一份记忆，输出新的记忆，分两节：`### Function Library` 和 `### Lessons & Pitfalls`。记忆只在下一题的第一轮 ACT 时放在最前面；目前没有长度上限。
- **ACT prompt**：在 clbench codebase_adaptation 原文基础上只改了三处：“issue”改成“problem”，推荐流程改成写函数的步骤，Important Rules 新增第 3、4 两条。其余部分逐字保留，以贴近训练时的分布。完整文本见 clbench 的 `agentcl_code/prompts.py`。

---

## 4. 跑评测

### 4.1 评测协议和数据生成

| 阶段 | 内容 | 什么时候跑 |
|---|---|---|
| stage 1 | **F（第一遍）**：一条长 episode 按官方顺序做完整条任务流，边做边写记忆。**B（无记忆基线）**：每道难题单独成一个 episode，不带记忆也不写记忆 | 默认跑 |
| stage 2 | **S（第二遍）**：记忆冻结在 F 结束时的状态，重做难题。**H（留出集）**：用同一份冻结记忆做 HumanEval-Pro 的 120 道题 | 可选，默认不跑 |

```bash
python3 scripts/agentcl/gen_agentcl_eval_data.py stage1 --out <dir>/stage1.jsonl \
    [--runs 3] [--baseline-samples 3] [--streams bcb_compositional bcb_naive] \
    [--heldout-baseline] [--baseline-subtasks]
python3 scripts/agentcl/gen_agentcl_eval_data.py stage2 --stage1-traj <stage1 轨迹目录> --out <dir>/stage2.jsonl \
    [--pass2-subtasks]
```

| 参数 | 含义 | 默认 |
|---|---|---|
| `--runs` | 每条任务流跑几遍第一遍，每一遍是一条独立的长 episode | 3 |
| `--baseline-samples` | 每道题做几次无记忆基线；B 是这几次的通过率 | 3 |
| `--streams` | 用哪几条任务流，见第 5.1 节 | bcb_compositional、bcb_naive |
| `--heldout-baseline` | 同时给留出集生成无记忆基线；只有算 GG 时才需要 | 关 |
| `--baseline-subtasks` | 同时给简单题生成无记忆基线；PG 用不到 | 关 |
| `--pass2-subtasks` | 第二遍连简单题也重做；SG 用不到 | 关 |

默认的 stage 1 一共 150 个 episode：6 条第一遍，外加 48 道难题 × 3 次基线 = 144 条。

episode 的 metadata 字段含义：

- `task_family: agentcl_code`：走 AgentCL 任务。
- `instance_ids`：按顺序排列的题号。
- `skip_write`：不写记忆。
- `initial_memory`：episode 从这份记忆开始。
- `agentcl: {phase, stream, run, sample}`：打标签，供算指标时区分阶段。

### 4.2 评测 miles ckpt（需要 8 卡节点）

```bash
cd miles/examples/codebase_adaption
EVAL_CKPT_DIR=<含 latest_checkpointed_iteration.txt 的 ckpt 目录> EVAL_TAG=fmtonly-iter300 \
  bash scripts/agentcl/eval_agentcl_orchard.sh          # stage 1
# 可选：EVAL_STAGE=2 同样的命令再跑一次，就是 stage 2
python3 scripts/agentcl/agentcl_metrics.py --traj /tmp/qixinx/runs/agentcl-<tag>-stage1/traj \
    --per-problem agentcl-<tag>-per_problem.csv
```

| 环境变量 | 含义 | 默认 |
|---|---|---|
| `EVAL_STAGE` | 1 或 2 | 1 |
| `EVAL_CKPT_DIR`、`EVAL_TAG` | 被测 ckpt 和本次评测的名字 | 必填，tag 默认取 iter 号 |
| `AGENTCL_RUNS`、`AGENTCL_BASELINE_SAMPLES`、`AGENTCL_STREAMS` | 对应第 4.1 节的数据生成参数 | 3、3、两条 BCB 流 |
| `AGENTCL_HELDOUT_BASELINE=1` | 同时生成留出集基线，要算 GG 时打开 | 关 |
| `CODEBASE_ACT_THINKING`、`CODEBASE_WRITE_PROMPT_V3`、`CODEBASE_WRITE_THINKING`、`CODEBASE_MULTIBLOCK_FEEDBACK` | 和被测 ckpt 训练时的开关保持一致 | 都是 1，与 thinking 类训练组一致 |
| `CODEBASE_SINGLE_ISSUE_PROMPT=1` | 评测 single-issue 那组 ckpt 时打开，使题目标题格式与它训练时一致 | 关 |
| `AGENTCL_SIF`、`AGENTCL_WORK_ROOT` | 沙箱镜像；每道题工作目录所在的根目录 | 去数据版；节点本地 `/tmp/qixinx/agentcl_ws` |

引擎的设置和 `eval_ckpt_19q_orchard.sh` 一样：只加载权重，训练前先做评测，评测完还会跑一个无害的空训练步。

### 4.3 评测 API 模型

在 miles 容器里运行；可以在登录机上跑，不需要 GPU。

```bash
set -a; source /home/qixinx/miles/.env; set +a
source /home/qixinx/apps/apptainer/env.sh
export TMPDIR=<可写目录> APPTAINERENV_PREPEND_PATH=/home/qixinx/apps/apptainer/bin
export APPTAINERENV_OPENAI_API_KEY="$OPENAI_API_KEY"
apptainer exec --bind /home/qixinx,/project/flame,<TMPDIR 所在盘> \
  --env PYTHONPATH=/home/qixinx/miles_pydeps/codebase_py312_clean,TMPDIR=$TMPDIR \
  /project/flame/qixinx/images/miles_dev-202606081341.sif \
  python3 examples/codebase_adaption/scripts/agentcl/run_api_agentcl.py \
    --data <stage1.jsonl> --out <轨迹目录> --model gpt-5.5 --concurrency 12 --work-root <可写目录>
```

实际可运行的例子：`/home/qixinx/agentcl_runs/gpt-5.5-newprompt/run_stage1.sh`。

| 参数 | 含义 | 默认 |
|---|---|---|
| `--model`、`--reasoning-effort` | 模型名和推理强度；不设推理强度就用 API 的默认值 | gpt-5.5，不设 |
| `--concurrency` | 同时跑几个 episode | 12 |
| `--max-steps` | 每题步数上限 | 40 |
| `--max-act-tokens`、`--max-write-tokens` | 单次调用的输出上限，包含隐藏推理 | 32768、16384 |
| `--no-write-v3` | 改用非 v3 结构的 WRITE | 关 |
| `--limit N` | 只跑前 N 个 episode，用于小规模试跑 | 0，即全部 |

- **断点续跑**：已经写出轨迹文件的 episode 会被跳过，所以中断后原命令重跑即可。
- **API 带来的差别**：模型的隐藏推理不会进入上下文，只有可见输出会。
- **轨迹位置**：放在 `/home/qixinx/agentcl_runs/` 下。注意 /home 有约 22G 的配额。

### 4.4 调试单道题

```bash
python3 scripts/agentcl/replay_grade.py <episode.json> --list
python3 scripts/agentcl/replay_grade.py <episode.json> <trial序号> --show-code --show-test
```

轨迹里只存了判分状态。这个工具会在新沙箱里重放这道题的所有命令，还原最终代码，再判一次，打印完整的报错。

---

## 5. 任务流和指标

### 5.1 任务流（`stream`）

| 名字 | 内容 | 题数 |
|---|---|---|
| `bcb_compositional` | 先做 48 道 BigCodeBench 简单题，再做 48 道难题。每道难题都会调用某道简单题的函数 | 96 |
| `bcb_naive` | 只有上面那 48 道难题，顺序相同 | 48 |
| `humaneval_heldout` | 120 道 HumanEval-Pro 难题，作为留出集 | 120 |
| `humaneval_compositional` | HumanEval-Pro 的简单题加难题，论文没有用到 | 240 |

任务流的顺序是固定的，任务代码不会打乱它。

题号格式是 `<family>:<raw|new>:<id>`：`raw` 是简单题，`new` 是难题。

### 5.2 指标

四个基础量，都只在难题上统计：

- **B_i**：第 i 题无记忆基线的通过率。
- **F_i**：第 i 题在第一遍里的对错。
- **S_i**：第 i 题在第二遍里的对错。
- **H_j**：留出集第 j 题带冻结记忆时的对错。

简单题只用来积累记忆，不计入任何指标。

| 指标 | 定义 | 含义 |
|---|---|---|
| PG | mean_i (F_i − B_i) × 100 | 边做边记时，记忆带来多少提升，单位是百分点。**默认主指标** |
| SG | mean_i (S_i − F_i) × 100 | 记忆冻结后重做，是变好还是变差（需要 stage 2） |
| GG | mean_j (H_j − B_j) × 100 | 记忆对没见过的新题有没有帮助（需要 stage 2 和留出集基线） |
| cum_success | Σ_i F_i | 第一遍做对的难题数 |
| cum_success_gain | Σ_i (F_i − B_i) | 按对错算的累计增益，等于 PG × 题数 / 100 |
| cum_reward | Σ_i r^F_i | 第一遍的 reward 总和，按步数打折，与 clbench 的 Cum Reward 同口径 |
| cum_baseline_reward | Σ_i mean_k r^B_{i,k} | 无记忆基线的 reward 总和 |
| cum_gain | cum_reward − cum_baseline_reward | 按 reward 算的累计增益，与 clbench 的 Cum Gain 同口径，但只算难题 |
| subtask_acc | 第一遍里简单题的正确率 | 仅供参考 |
| F_acc、B_acc、S_acc、H_acc | 对应基础量的平均正确率，单位 % | 仅供参考 |

- **怎么汇总**：每一遍第一遍单独算出一组指标，表里报告多遍的均值 ± 标准差（总体标准差）。
- **逐题明细**：`--per-problem` 会输出每道难题每一遍的 F、B、每次基线采样的对错（B_samples）、F−B 和两种 reward。
- **怎么读 PG**：
  - `bcb_naive` 的 PG 应该接近 0。这条流里题目之间没有可复用的内容，它是对照。
  - `bcb_compositional` 的 PG 衡量简单题积累的经验能不能用到难题上。

---

## 6. 已知的数据问题

这些都是数据集本身的问题，在自检和失败分析中核实过。

| 题 | 问题 | 处理 |
|---|---|---|
| HumanEval-Pro new/41 | 函数名和基础函数相同，测试会先用基础版本把它覆盖掉，所以无解 | 在 `KNOWN_BROKEN` 里标记，仍然保留在流里 |
| BigCodeBench raw/269 | 参考解答本身过不了测试 | 在 `KNOWN_BROKEN` 里标记；它是简单题，不影响 PG |
| HumanEval-Pro new/46、63、112 | 题面被截断，后半段和函数签名被切进了 solution 字段 | 加载时把被切走的部分补回题面 |
| HumanEval-Pro new/29 | 题面只有函数签名，描述被放进了 test_code | 加载时从 test_code 把描述补回来 |
| BigCodeBench new/692 | 题面里 `import pandas as pd` 被粘在注释行末尾，等于被注释掉了 | 在函数签名前补一行独立的导入 |
| HumanEval-Pro 0、4、6、9、29 | 签名里用了 `List` 或 `Tuple`，但题面没有导入 | 所有题的 stub 开头都加 `from typing import *`，和判分时的拼法一致 |
| BigCodeBench 25/48 道难题 | 导入语句被粘在描述注释的末尾 | **没有修**。模型基本都能自己看出来并补上 |
| BigCodeBench 47、54、280、734 等 | 题面和测试看起来矛盾 | **不要当成坏题删掉**。其中 516、42、198、897 在无记忆时从没做对过，却在 compositional 流里靠简单题的记忆做对了：测试遵循的是基础函数的约定。直接删掉，会把 GPT-5.5 的 PG 从 +7.6 砍到 +2.5 |
| HumanEval-Pro 约 16 道 | 测试和题面矛盾，或者参考答案有错（例如 83） | 只影响留出集和 GG；如果要算 GG，建议另报一个干净子集的结果 |

设计上的一点说明：难题依赖的基础函数，对智能体是**有意不可见的**。`bcb_naive` 不给，`bcb_compositional` 要靠记忆把它从简单题带过来。这正是 PG 想测的能力，所以**不要**把基础函数写进题面。

---

## 7. 常见坑

- **嵌套 apptainer 的挂载泄漏**：在 miles 容器里启动沙箱时，会继承外层导出的 `APPTAINER_BIND`，把 /home 和 /project 挂进沙箱，`--no-mount bind-paths` 挡不住。任务代码的 `_launcher_env` 会删掉 `*_BIND` 和 `*ENV_` 变量；不要绕过它。
- **沙箱启动失败**：高并发时，squashfuse 挂载偶尔超时，返回码 255，并报 `FATAL: container creation failed`。任务代码会自动重试 3 次。GPT-5.5 的一次评测里，5386 个命令轮次中出现过 13 次。
- **登录机的目录**：`/mnt/tmp` 的根目录不可写，要用 `/mnt/tmp/<uid>/`。登录机的 `/tmp` 是共享 tmpfs，经常被占满。
- **在训练节点上做大量 I/O**：ckpt 转换这类读写几十 GB 的任务产生的 page cache，会计入 slurm cgroup 的内存，严重时会导致训练被 OOM kill。要在训练节点上做，就用 `scripts/hf_backfill_nice.sh`：它会降低优先级、清理 page cache，内存超过水位线就暂停。并行版本用 `scripts/backfill_all_but_latest.sh <RUN> <LOG> <JOBS>`。
- **`pkill -f` 杀掉自己**：匹配模式写成 `[x]yz` 这种形式，否则会连执行它的 shell 一起杀掉。

---

## 8. 已有结果（GPT-5.5，stage 1，默认参数）

| 版本 | compositional PG | naive PG | 目录 |
|---|---|---|---|
| 旧 prompt（修 bug 措辞的 ACT，仓库导向的 WRITE） | +7.6 ± 2.9 | +0.0 ± 1.0 | `/home/qixinx/agentcl_runs/gpt-5.5/` |
| 新 prompt（当前版本） | +12.5 ± 2.0 | +2.8 ± 1.0 | `/home/qixinx/agentcl_runs/gpt-5.5-newprompt/` |

新 prompt 的提升里，大约一半来自无记忆基线的采样波动：同一批题同样的设置，基线从 144 次对 76 次降到了 73 次。3 遍、48 道题的规模下，差 1 到 2 道题，还不足以得出显著的结论。
