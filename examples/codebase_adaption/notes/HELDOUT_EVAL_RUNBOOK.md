# held-out 四条 sequence 评测:换账号跑的操作手册

*2026-09-23 · 面向"同一个 babel、另一个账号"。结果与分析见 clbench `notes/HELDOUT_LONGSEQ_EVAL_RESULTS.md`,数据构造见 `notes/HELDOUT_SEQUENCE_CONSTRUCTION.md`*

跑的是 4 条 held-out sequence(用训练 53 个 repo 之外的 12 个 repo 合成):

| 组 | 文件 | sequence | 每条题数 |
|---|---|---|---|
| 13+13 | `FINAL_2x13_26issues.jsonl` | ① bleach + charset_normalizer;② flake8 + line_profiler | 26 |
| 9+9+9 | `FINAL_3x9_27issues.jsonl` | ③ alive-progress + bleach + line_profiler;④ flake8 + sqlparse + soupsieve | 27 |

每条 sequence 两种口径:**replace**(带记忆,WRITE 每题覆写,5 个固定题序各跑一遍)和 **stateless**(单题无记忆,每题 4 次采样)。

---

## 0. 交接包(group 可读,不用读我的 /home)

```
/data/group_data/rl/yuxiaoq/qixinx/heldout_eval_kit/
├── README.md                  # 本文件
├── scripts/                   # 拷进你的 miles:examples/codebase_adaption/scripts/
│   ├── eval_heldout_longseq.sh          # 本地权重(Megatron 加载式)评测入口
│   ├── eval_heldout_api.sh              # API 模型(GPT-5.5 等)入口
│   └── run_heldout_eval_babel_v2.sh     # 驱动:逐组合跑、等 GPU 空闲、落盘后收尾
├── data/                      # 拷进你的 miles:examples/codebase_adaption/data/
│   ├── ho_2x13_26issues_replace.jsonl      # 10 行 = 2 sequence × 5 个固定题序
│   ├── ho_3x9_27issues_replace.jsonl       # 10 行
│   ├── ho_all97_stateless.jsonl            # 97 行(两组去重),配 4 次采样 = 388 条轨迹
│   ├── ho_2x13_26issues_stateless.jsonl    # 按组分开的 stateless(可选,旧口径)
│   └── ho_3x9_27issues_stateless.jsonl
├── clbench_scripts/           # 拷进你的 clbench:scripts/swesmith/
│   ├── gen_heldout_eval_data.py     # 由 FINAL_*.jsonl 重新生成上面的 ho_*.jsonl
│   ├── score_heldout_eval.py        # 细打分(按 sequence / 按位置 / 按 repo / 配对 gain)
│   ├── verify_heldout_grading.py    # 判分可用性验证(gold 过、no-op 不过)
│   ├── gen_episodes_v4.py           # 从题池合成新 sequence
│   └── pick_episodes.py
├── report_tables.py           # 固定汇报用的两张表
├── patches/
│   ├── 01-thinking-no-memory-guard.diff   # ★必须打,否则 stateless 跑不了
│   └── 02-cmd-stdout-cap.diff             # 建议打(babel 宿主 OOM 防护)
└── seq/                       # sequence 与题目 registry
    ├── FINAL_2x13_26issues.jsonl
    ├── FINAL_3x9_27issues.jsonl
    └── heldout_registry.jsonl   # 1,611 题;CODEBASE_EVAL_DATASET 指的就是它
```

同一份 registry / sequence 也在 `/data/group_data/rl/yuxiaoq/qixinx/swe_smith/heldout_v4/`。

---

## 1. 仓库和分支

```bash
git clone git@github.com:Racktic/miles.git ~/miles
cd ~/miles && git checkout -b heldout racktic/swecl-actthink-scaffold   # 这条分支上已有 swe_smith eval 分支 + --task swe_smith 的 API 驱动
git clone git@github.com:Racktic/continual-learning-bench.git ~/continual-learning-bench   # main 即可,含 swe_smith 任务本体
```

held-out 的评测脚本、`ho_*.jsonl` 评测数据、两个补丁都**已经在这条分支里**(`examples/codebase_adaption/{scripts,data,patches/heldout}`),clone 完只需要做第 2 步(打补丁)和第 3 节的改路径。
sequence 本体 + 题目 registry 在 clbench 仓库 `data/swe_smith/heldout_v4/`,题目镜像和权重见第 4 节。

---

## 2. 必须打的补丁(★ 不打则 stateless 全部失败)

分支上 `examples/codebase_adaption/codebase_rollout.py:781` 是:

```python
if act_think and no_memory:
    raise ValueError("CODEBASE_ACT_THINKING is not supported together with CODEBASE_NO_MEMORY")
```

我们的 stateless 是 **ACT thinking + 无记忆**跑的,会直接撞这个 raise。补丁把它改成**只拒绝多题 episode**:

```bash
cd ~/miles && git apply examples/codebase_adaption/patches/heldout/01-thinking-no-memory-guard.diff
```

为什么单题安全:这个守卫防的是 `_encode_prompt_with_fifo` 的上下文裁剪与 thinking 的 append-only token 缓冲冲突,而**只有多题 ICL episode 才会跨题累积上下文**。单题 episode 上限 40 轮,实测最大 `context_tokens` 46,568 对 240,000 预算,`context_truncation_count` 全程 0。

第二个补丁 `patches/heldout/02-cmd-stdout-cap.diff` 是 agent 命令 stdout 的有界捕获(head 1MB + tail 1MB,判分/setup 命令全量不受影响)。babel 上不打会有宿主 OOM 风险(某些题 `find /` 之类输出能打爆内存),判分结果不受影响。**建议打。**

---

## 3. 拷文件 + 改路径(代码里写死的是我的账号)

脚本和数据都在仓库里了,只有出表脚本 `report_tables.py` 在交接包里(`$K/report_tables.py`,`K=/data/group_data/rl/yuxiaoq/qixinx/heldout_eval_kit`)。

然后按下表改 6 处(都是写死的绝对路径,`YOU` = 你的用户名):

| 文件 | 行 | 原值 | 改成 |
|---|---|---|---|
| `scripts/run_heldout_eval_babel_v2.sh` | 15 | `SD=/home/qixinx/miles/examples/codebase_adaption` | 你的 miles 路径 |
| 同上 | 16 | `CKROOT=/data/group_data/rl/yuxiaoq/qixinx/agentcl/ckpt` | 被测 ckpt 的根目录(见第 4 节) |
| 同上 | 17 | `RUNROOT=/data/group_data/rl/.../codebase_adaption_runs` | 你自己的轨迹根目录(**要可写**) |
| `scripts/eval_heldout_longseq.sh` | 26–27 | `CODEBASE_HF_CKPT` / `CODEBASE_TORCH_DIST` 指向 `/data/user_data/qixinx/...` | 你能读的 Qwen3.5-4B / torch_dist(见第 4 节) |
| 同上 | 55 | `CODEBASE_EVAL_DATASET` 默认 `/data/user_data/qixinx/swe_smith/heldout_v4/heldout_registry.jsonl` | `/data/group_data/rl/yuxiaoq/qixinx/swe_smith/heldout_v4/heldout_registry.jsonl`(或直接用环境变量覆盖,不改文件) |
| 同上 | 79 / `eval_heldout_api.sh` 21 | `NODE_RUN_DIR` / `OUT` 里的 runs 根目录 | 同 `RUNROOT` |
| `scripts/eval_heldout_api.sh` | 55、57、60 | `/home/qixinx/miles`、读我的 `.env` | 你的 miles 路径、你自己的 `OPENAI_API_KEY` |
| `scripts/cluster_babel_env.sh` | 10、18、20 | miles SIF / HF ckpt / torch_dist 在 `/data/user_data/qixinx` | 你能读的副本(第 4 节) |
| `report_tables.py` | 8、10 | `R=`(runs 根目录)、`CKS`(模型 tag 列表) | 你的 `RUNROOT`、你要出表的 tag |

`CODEBASE_EVAL_DATASET`、`CODEBASE_SCR`、`CLBENCH_ROOT`、`CLBENCH_SIF_DIR`、`CODEBASE_PYDEPS`、`CODEBASE_MILES_SIF` 都是 `${VAR:-默认}`,**可以不改文件、在命令前面 export 覆盖**;上表里只有 SD/CKROOT/RUNROOT 和 `eval_heldout_longseq.sh` 的 26–27 行是硬写的。

---

## 4. 环境:哪些能直接用,哪些要你自己准备

group 可读(直接用):

| 东西 | 路径 |
|---|---|
| 题目 SIF 库(含 held-out 12 个 repo) | `/data/group_data/rl/yuxiaoq/qixinx/swesmith_sifs` |
| sequence + registry | `/data/group_data/rl/yuxiaoq/qixinx/swe_smith/heldout_v4/` |
| 交接包 | `/data/group_data/rl/yuxiaoq/qixinx/heldout_eval_kit/` |
| 被测 ckpt(torch_dist) | `/data/group_data/rl/yuxiaoq/qixinx/agentcl/ckpt/<tag>/ckpt`,现有 tag:`qwen35-4b-base`、`explore-fmtonly-scaffoldv2-ACTthinking-iter191`、`explore-gwin-think-iter251`、`actonly-think-iter239` |

在我的 `user_data` 里(**你读不到,要自己准备或找我拷**):

| 东西 | 我的路径 | 你要做什么 |
|---|---|---|
| miles 镜像 | `/data/user_data/qixinx/images/miles_dev-202606081341.sif` | 自己有一份就行;没有让我拷到 group_data |
| Python 依赖 | `/data/user_data/qixinx/miles_pydeps/codebase_py312_clean` | 同上(litellm、mini-swe-agent、jinja2、pyyaml、pydantic、openai、huggingface_hub) |
| Qwen3.5-4B HF 权重 + torch_dist | `/data/user_data/qixinx/Qwen3.5-4B{,_torch_dist}` | 自己下 HF 权重再转 torch_dist,或让我拷 |

另外:`CODEBASE_SCR` 默认 `/scratch/qixinx`,改成你自己的;`MSWEA_SINGULARITY_EXECUTABLE=apptainer` 已在脚本里设好。

---

## 5. 跑本地权重(需要 8 卡 allocation)

```bash
# 1) 先拿一个 8 卡 allocation(rl 分区),记下 JOBID
# 2) 在 allocation 里跑驱动(三个组合顺序跑):
cd ~/miles/examples/codebase_adaption
srun --overlap --jobid=<JOBID> -n1 --gres=gpu:8 --cpus-per-task=128 --mem=0 bash -c '
  cd ~/miles/examples/codebase_adaption &&
  MAX_MIN=300 POLL_SEC=60 bash scripts/run_heldout_eval_babel_v2.sh \
    <tag>:stateless:ho_all97_stateless.jsonl \
    <tag>:replace:ho_2x13_26issues_replace.jsonl \
    <tag>:replace:ho_3x9_27issues_replace.jsonl'
```

- `<tag>` 要先加进驱动第 18–23 行的 `CKPT` 映射表(`[tag]="<ckpt 目录>"`,该目录要有 `latest_checkpointed_iteration.txt`)。
- 轨迹落在 `$RUNROOT/hoeval-<tag>-<mode>-<数据>/traj/eval/heldout/rollout_0/ep_*.json`。
- 中断后原命令重跑即可:已达期望条数的组合会跳过(replace 期望 10 份,stateless 期望 388 份)。
- 实测耗时(4B 模型、8×A100):stateless 约 1h10m,每组 replace 约 1h45m–2h30m;未训练的 base 更快(三组共 3h24m)。

## 6. 跑 API 模型(不占 GPU)

```bash
EVAL_MODE=stateless EVAL_DATA=<abs>/data/ho_all97_stateless.jsonl EVAL_TAG=gpt55 \
  API_MODEL=gpt-5.5 API_CONCURRENCY=16 bash scripts/eval_heldout_api.sh
EVAL_MODE=replace  EVAL_DATA=<abs>/data/ho_2x13_26issues_replace.jsonl EVAL_TAG=gpt55 API_CONCURRENCY=10 bash scripts/eval_heldout_api.sh
EVAL_MODE=replace  EVAL_DATA=<abs>/data/ho_3x9_27issues_replace.jsonl  EVAL_TAG=gpt55 API_CONCURRENCY=10 bash scripts/eval_heldout_api.sh
```

- 三组可以并发,但**每组的 `CODEBASE_APPTAINER_RUNTIME_ROOT` 必须不同**(脚本里已按 RUN_ID 隔离)。
- 仍然要在 miles 容器里跑(脚本自己借 `launch_codebase_adaption_apptainer.sh` 进容器,容器内再执行自己),因为要嵌套起题目 SIF。
- 用量参考:4 条 sequence 全量 = 14,235 次调用、约 1.93 亿 prompt token(79% 命中缓存)、634 万输出 token,墙钟约 70 分钟。
- 与本地权重的唯一实质差别:API 模型的隐藏推理不进入多轮上下文,只有可见输出进。

## 7. 出结果

```bash
# 两张固定表(13+13 组 / 9+9+9 组;要在计算节点跑,login 节点没挂 /data)
python3 ~/heldout_prep/report_tables.py
# 细打分
python3 ~/continual-learning-bench/scripts/swesmith/score_heldout_eval.py --json scores.json
```

指标口径:`reward = 1 − (turns−1)/40`(未解出 = 0);Cum Reward / Cum Success 是一条 sequence 内逐题求和;replace 取 5 个 run 的均值 ± 总体标准差;stateless 每题恰好取 4 次采样;`Cum Gain = replace − stateless`。

## 8. 踩过的坑(照抄能省半天)

- **replace 的轨迹只在整条 sequence 跑完才落盘**(26/27 题串行)。判"是否在跑"要看组合日志里的 `gen throughput`,不要看轨迹数。
- **驱动的 `MAX_MIN` 单位**:`waited` 每轮加 `(POLL+59)/60` 分钟;老版本每轮加 2 分钟,导致 `MAX_MIN=300` 实际只有 150 分钟,把一条跑到第 26/27 题的 replace 误杀了。设的时候确认一下。
- **login 节点上长跑的 `srun` 客户端会被杀**(实测 4–5 小时,step 记 `CANCELLED 0:15`)。长评测要用守护循环反复拉起驱动(驱动会跳过已完成组合),别指望一个 srun 撑到底。
- **不要改正在跑的脚本**,连 `sed -i` / 换 inode 也不行:脚本在 NFS 上、bash 在计算节点时,旧 inode 被删,计算节点下次读脚本拿 ESTALE 直接静默退出(症状:最后一行日志没打印)。
- **组合之间要等 GPU 真正空闲**(驱动里的 `wait_gpu_clean`),否则上一批的 Ray/SGLang 没退干净,下一批抢不到卡直接崩。
- **`pkill -f <pattern>` 会杀掉自己所在的那条命令**;`pgrep -f` 同理会匹配到自身。
- **stateless 用的是原始题序(段内易→难)**,它的"前段/后段"差异只是难度梯度,不能当位置效应读;replace 侧才有块内打乱。
- **两组数据之间有 9 道题重叠**,打分脚本按 run 名里的数据集 tag 选映射表,别把两组合成一张表算。
