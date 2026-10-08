# ACT / WRITE 解耦的混合模型评测

`codebase_rollout.py` 里 ACT (执行 bash 命令的 agent) 和 WRITE (写记忆的模型) 共用同一个 sglang url。
这个目录把两者拆开: ACT 和 WRITE 各自可以是 **本地 ckpt / 独立 sglang server / OpenAI 兼容 API / Anthropic API**,
不改仓库里任何现有文件: 作业启动时用 Python 从 `codebase_rollout.py` (或 `codebase_rollout_icl.py`) **现场派生**一个副本
`codebase_rollout_mix_<jobid>.py` (替换两处 `_infer` 调用、追加 API 辅助函数), 通过 run 脚本的 `CODEBASE_ROLLOUT_MODULE`
开关选用, 跑完即删。派生步骤会校验每处替换恰好命中一次, 原文件结构变了会直接报错退出。

## 文件

| 文件 | 作用 |
|---|---|
| `mixmodel_eval.sbatch` | 评测作业本体 (派生副本、可选起 WRITE server、跑 19 题评测、watch 进度) |
| `mixscore.py <replace_run> <stateless_run>` | 出 cum reward / cum gain (clbench 口径: 按题取均值再求和; gain 基线 = **ACT 模型自己的 stateless**) |
| `runstats.py <run>` | 每个 episode 的 total / 解题数 / 轮数, 以及 5 个顺序的均值与样本 std (ddof=1) |
| `make_shim.sh <torch_dist> <shim>` | 造 ACT 走 API 时需要的 ckpt 占位目录 |

## 模式与环境变量

必填: `EVAL_MODE` (stateless | replace), `EVAL_CKPT_DIR` (ACT 的 Megatron ckpt; ACT 走 API 时给占位 shim), `EVAL_TAG` (结果目录名 `eval-<TAG>-<MODE>`)。

| 变量 | 含义 | 默认 |
|---|---|---|
| `ACT_API_MODEL` | 非空则 ACT 走 API (如 `gpt-5.4`, `claude-opus-4-7`); 空则用 `EVAL_CKPT_DIR` 的本地模型 | 空 |
| `ACT_API_PROVIDER` | `openai` (chat completions / Responses) 或 `anthropic` (Messages) | openai |
| `ACT_JSON` | 1 = 保留 clbench 官方 JSON 系统模板 + 结构化输出 (OpenAI 用 strict json_schema; Anthropic 按官方做法把 schema 指令追加到最后一条 user 文本后); 0 = 本仓库的文本 bash 块格式 | 0 |
| `ACT_NOBUDGET` | 1 = 去掉每轮 "[Budget] You have N of 40 turns remaining" 提示 (官方没有这一行) | 0 |
| `ACT_STATEFUL` | 1 = OpenAI Responses API 有状态 (store=true + previous_response_id, 每轮只发新消息); 仅 openai | 0 |
| `WRITE_MODE` | `local` (与 ACT 同一本地模型) / `server` (`WRITE_HF` 在 GPU 2 起独立 sglang) / `api` (`WRITE_API_MODEL`, OpenAI chat) | local |
| `WRITE_HF`, `WRITE_API_MODEL` | server 模式的 HF 权重目录; api 模式的模型名 | 空, gpt-5.5 |
| `API_MAX_TOKENS_ACT` / `API_MAX_TOKENS_WRITE` | 每次调用输出上限 (Anthropic 官方口径 128000) | 32768 / 16384 |
| `API_CONCURRENCY` | 同时在飞的 API 请求数 | 8 |
| `ROLLOUT_SRC` | 派生源文件: `codebase_rollout.py` (记忆替换评测) 或 `codebase_rollout_icl.py` (naive ICL, 19 题一条上下文, 需配 `CODEBASE_NO_MEMORY=1 CODEBASE_CONTEXT_MAX_TOKENS=1000000`) | codebase_rollout.py |
| `MILES_ROOT` | miles 仓库根 | ~/qixinx/miles |
| `MIX_ENV_FILE` | 存 `OPENAI_API_KEY=` / `ANTHROPIC_API_KEY=` 的文件 (运行时读取, 不写进任何文件或日志) | $MILES_ROOT/.env |
| `MIX_SIF` | WRITE server 用的 apptainer 镜像 | babel 上的 miles_dev 镜像 |
| `MIX_EVAL_SCRIPT` | 本集群的 19 题评测入口 | scripts/eval_ckpt_19q_babel2.sh |
| `MIX_INNER_SCRIPT` | 支持 `CODEBASE_ROLLOUT_MODULE` 的 run 脚本 | run_codebase_adaption_qwen3.5_4B_multiport_si3.sh |
| `EVAL_TRAJ_ROOT` (打分脚本) | 轨迹根目录 | ~/eval_traj |

训练开关 (`CODEBASE_ACT_THINKING=1 CODEBASE_WRITE_THINKING=1 CODEBASE_WRITE_PROMPT_V3=1 CODEBASE_MULTIBLOCK_FEEDBACK=1`) 在脚本里写死, 与训练一致; stateless 用 `data/baseline19_textfmt_eval_think.jsonl` (每题 4 次采样, 76 个 episode), replace 用 `replace_textfmt_eval.jsonl` (5 个顺序)。

## 提交 (babel 口径)

在 `examples/codebase_adaption` 目录下提交 (日志落 `logs/%x-%j.out`, watch 进度行依赖这个路径); GPU 数: ACT 本地或 API 都要 2 卡 (TP=2 把本地/占位模型拉起来), WRITE server 再加 1 卡。
babel 计算节点的 conda hook 超过 2 分钟会让 `--export` 显式列表的作业 "user env retrieval failed" 被 held, 所以一律用 `env -i` 清环境再 `--export=ALL`:

```bash
cd $MILES_ROOT/examples/codebase_adaption
S=scripts/mixmodel/mixmodel_eval.sbatch; P=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
B=/data/group_data/rl/yuxiaoq

# 1. ACT = 本地 ckpt, WRITE = 另一个模型的 sglang server (3 卡)
env -i USER=$USER HOME=$HOME PATH=$P EVAL_MODE=replace EVAL_CKPT_DIR=$B/eval_shims/gwfmt347 EVAL_TAG=mix-gwfmt347act-base4bwrite \
  WRITE_MODE=server WRITE_HF=$B/hf/Qwen3.5-4B \
  /usr/bin/sbatch -p general -q normal --constraint=VRAM_48GB --gres=gpu:3 --exclude=<坏节点列表> --export=ALL $S

# 2. ACT = 本地 ckpt, WRITE = GPT-5.5 API (2 卡)
env -i USER=$USER HOME=$HOME PATH=$P EVAL_MODE=replace EVAL_CKPT_DIR=$B/eval_shims/gwfmt347 EVAL_TAG=mix-gwfmt347act-gpt55write \
  WRITE_MODE=api WRITE_API_MODEL=gpt-5.5 \
  /usr/bin/sbatch -p general -q normal --constraint=VRAM_48GB --gres=gpu:2 --export=ALL $S

# 3. ACT = GPT-5.4 (对齐官方 clbench ICL: JSON + 无 Budget + Responses 有状态), naive ICL, 2 卡; EVAL_CKPT_DIR 给占位 shim
env -i USER=$USER HOME=$HOME PATH=$P EVAL_MODE=replace EVAL_CKPT_DIR=$B/eval_shims/base EVAL_TAG=mix-gpt54icl-json-nobudget-stateful \
  ACT_API_MODEL=gpt-5.4 ACT_JSON=1 ACT_NOBUDGET=1 ACT_STATEFUL=1 WRITE_MODE=api \
  ROLLOUT_SRC=codebase_rollout_icl.py CODEBASE_NO_MEMORY=1 CODEBASE_CONTEXT_MAX_TOKENS=1000000 \
  /usr/bin/sbatch -p general -q normal --constraint=VRAM_48GB --gres=gpu:2 --export=ALL $S

# 4. ACT = claude-opus-4-7 (对齐官方 Anthropic 路径), naive ICL
env -i USER=$USER HOME=$HOME PATH=$P EVAL_MODE=replace EVAL_CKPT_DIR=$B/eval_shims/base EVAL_TAG=mix-opus47icl-r1 \
  ACT_API_PROVIDER=anthropic ACT_API_MODEL=claude-opus-4-7 ACT_JSON=1 ACT_NOBUDGET=1 WRITE_MODE=api API_MAX_TOKENS_ACT=128000 API_CONCURRENCY=4 \
  ROLLOUT_SRC=codebase_rollout_icl.py CODEBASE_NO_MEMORY=1 CODEBASE_CONTEXT_MAX_TOKENS=1000000 \
  /usr/bin/sbatch -p general -q normal --constraint=VRAM_48GB --gres=gpu:2 --export=ALL $S
```

同一个混合配置要算 gain, 再跑一次 `EVAL_MODE=stateless` (其余变量相同), 然后:

```bash
python3 scripts/mixmodel/mixscore.py eval-<TAG>-replace eval-<TAG>-stateless
python3 scripts/mixmodel/runstats.py eval-<TAG>-replace
```

## 换集群要改什么

1. `MILES_ROOT`, `MIX_ENV_FILE`, `MIX_SIF` 按本地路径给。
2. `MIX_EVAL_SCRIPT` / `MIX_INNER_SCRIPT` 指向本集群能跑通的 19 题评测入口和 run 脚本; run 脚本必须支持 `CODEBASE_ROLLOUT_MODULE` (si3 脚本里已有)。
3. `make_shim.sh` 造一个 base 模型的占位 ckpt 目录 (ACT 走 API 时用)。
4. sbatch 头里没有分区/QOS/约束, 全在命令行给。

## 已知坑

- 派生副本按作业号命名, 两个作业同名副本会被先结束者的清理钩子删掉 (曾出过 ModuleNotFoundError)。
- WRITE server 的 url 必须写成字面量塞进副本: rollout 跑在 Ray worker 里, 自定义环境变量过不了 RUNTIME_ENV_JSON 白名单。
- ACT API 返回 HTTP 400 (多为上下文超限) 视为不可重试, 该 episode ABORTED, 与本地超长同口径; WRITE API 失败则本题记忆不更新, 不中止序列。
- OpenAI Responses 路径把多条 output 消息的文本拼接, 极少数情况下得到多个 JSON 拼接而解析为空命令 (每千轮约 1 次)。
- Anthropic 路径的 JSON 解析比官方 clbench 宽松 (官方对带真实换行的 command 字段会判失败并在 4 次重采样后给零分), 做"官方口径"对比时要注明。
- 官方 leaderboard 的 icl · claude-opus-4-7 (10.36) 跑的时候 tablib-534/540/547 容器缺 MarkupPy, 这 3 题全 0; babel 的镜像没有这个问题, 对比时要去掉这 3 题或注明。
