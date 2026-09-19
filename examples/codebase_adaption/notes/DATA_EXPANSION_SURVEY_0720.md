# SWE-CL 训练数据扩充调研与三轨方案(2026-07-20)

> 动机:五检验判别分析(见 [[V3THINK_MEMORY_AND_MEMORIZATION_0720]] §6)实证现有 240 题池被权重背穿
> (单题曝光中位 32 次/max 367),记忆需求被参数蚕食。需要新数据满足:**同 repo 多题可组序列、
> 可执行判分、题间知识可复用、总池大到权重背不动(低重复/序列内独占 repo)**。
> 三路调研:① SWE agentic RL 数据集普查(网络,已对抗性核查)② 现成 SWE CL 序列搜索(网络)
> ③ 内部 studybench 转换可行性(本地实测)。关键数字均经一手来源验证,截至 2026-07-20。

---

## 0. 总判与生态位

- **同构的现成替代不存在**:SWE-Bench-CL(我们的现数据源,arXiv 2507.00014)仍是唯一"Verified issue 按 repo 分组按时间排序"的资源(4 个 fork 全是无改动复制);
- **"时间序 + 可执行 + RL 规模"三者兼备的公开数据集不存在**——这是我们的生态位,也是论文卖点;
- 头部 RL 数据集恰是"少 repo 多题"型(担心的"repo 太散"只对长尾数据集成立),序列可构造性比预期好。

## 1. RL 训练数据集对照表(按 CL 序列适配度排序)

| 数据集 | 规模/repo | per-repo 分布(实测) | 判分 | 时间字段 | 镜像组织(SIF 成本) | 可得性 |
|---|---|---|---|---|---|---|
| **SWE-smith** ([HF](https://huggingface.co/datasets/SWE-bench/SWE-smith), arXiv 2504.21798) | py 50.9K/222 repo + 六语言 ~37K | **均值 381/repo**,max pandas 2277 | ✅ F2P/P2P+官方 harness | ❌ 单 commit 快照,无时间序 | ⭐ **每 repo 一镜像**(2.2-2.6GB/个) | MIT |
| **R2E-Gym-Subset** ([HF](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset), arXiv 2504.07164) | 4578/10 repo | **最小 repo 108 题**(pandas 1444/numpy 781/...) | ✅(DeepSWE/SkyRL 已跑通 RL) | ⚠️ commit_hash 可回查日期(已抽验) | per-instance(335MB-1.1GB) | Subset 公开未标 license;Full gated |
| **SWE-rebench V2** ([HF](https://huggingface.co/datasets/nebius/SWE-rebench-V2)) | 32K/3617 repo/20 语言 | 长尾,但 **73 repo ≥50 题**(top-100≈8.6K) | ✅ 双 pass 验证+每条镜像名 | ✅ created_at(2014-2025,题新) | per-instance(swerebenchv2/*) | CC-BY-4.0,月更 |
| SWE-Gym (HF, arXiv 2412.21139) | 2438/11 repo | pandas 737/MONAI 374/... | ✅ | ✅ created_at | per-instance ~2400 个 | MIT |
| SWE-Flow (arXiv 2506.09003) | 16K/62 项目 | ~260/项目 | ✅ TDD 单测 | ⭐ 自带依赖拓扑序(非时间序) | per-project | GitHub |
| SWE-bench-extra | 6376/1974 repo | 摊薄(111 repo ≥10) | ✅ 但无镜像 | ✅ | 自建 | CC-BY-4.0 |
| OpenSWE/daVinci | 45K/12.8K repo | 摊薄,不合格 | ✅ | 有 | per-instance | gated |

**坑清单**(核查修正):princeton 官方 train split 的 F2P/P2P **全空**,不可 RL,勿踩;Skywork-SWE/Kimi-Dev/SWE-RL(Meta)/Qwen3-Coder 训练数据均未公开(SWE-RL 是 difflib 相似度 reward 非执行判分);SWE-Gym "6TB 镜像"系误传(那是 R2E 的);SWE-smith 单镜像实测 2.2-2.6GB 非官方宣称 1GB;SWE-smith 主数据集 2025-12 停更,用 SWE-smith-py 等语言版;Nemotron 系是 SWE-Gym/R2E/rebench 的轨迹衍生品,非新题源。

## 2. 现成 CL 序列基准与相邻工作

**A. 可执行的"演化链"基准(结构不同但可用,见 §3 修正)**
- **SWE-EVO-LongChain-50**(HF: zhiyuanhucs/SWE-EVO-LongChain-50,arXiv 2512.18470):50 链×5-15 milestone(共493),每步真实 commit 过渡,F2P/P2P+docker(145 个/141GB);12 repo(9 Go+pytest/click/sqlfluff)。⚠️ pytest 与现池同仓库,需时间窗查污染;
- **SWE-Milestone / DeepCommit**(arXiv 2603.13428,MIT):7 repo 5 语言 98 milestone + 依赖 DAG(110 边)+ 每步 docker;孤立任务 >80% → 连续 ≤38%。⚠️ scikit-learn 同仓库需查;
- **SWE-Chain**(arXiv 2605.14415):12 条版本升级链/155 过渡,Build+Fix 判分,与 Verified 无重叠;
- **SWE Context Bench**(arXiv 2602.08316):1100 base+376 related,related 组由**真实 issue/PR 依赖引用边**挖掘——比纯时间序更保证"前题经验对后题有用",其挖掘方法值得借用。

**B. 通用 agent-CL 基准**
- CL-Bench(arXiv 2606.05661,含 SWE 域,引用了 SWE-Bench-CL):关键发现 naive ICL 胜过专门 memory 系统;
- **AgentCL(OSU,arXiv 2606.02461)的方法论警告**:纯时间序 naive 流可能**根本区分不出 memory 方法优劣**,需"刻意可复用"的 controlled compositional 流——我们构造序列时必须显式保证题间知识复用性(依赖序/共享文件重叠),不能只按时间排;
- 方法论论文:SWE-Exp(轨迹蒸馏经验库)、Subtask-Level Memory(记忆粒度对齐,Verified +4.7pp)。

## 3. 修正:"评测向"≠"不可训"(用户指正,2026-07-20)

早先表述"2026 新基准都是评测向不可训"是概念偷换。**能不能训只取决于:可执行判分 + license + 规模**。LongChain-50/SWE-Milestone/SWE-Chain 三者判分与镜像齐全、license 无碍,**完全可训**;只要我们不在其上报数,不存在污染。真实制约仅是**规模小**(493/98/155 < 现池 240,做主池会更快背穿)。正确用法:
1. **当高价值课程片段**:milestone 链是有状态依赖(后步建于前步产出),是"记忆必要性"的最强形态,混入大池每链只过 1-2 遍;
2. **复用其数据工厂**(更大的解锁):DeepCommit 的 milestone-DAG 重建 pipeline、SWE-EVO 的 release-notes 切链法均开源——在更多 repo 上自产任意规模依赖链;
3. 留一手:官方子集留出不训(如 50 链训 40 留 10),或只训自产链,保未来可比性。

## 4. studybench-v2-v6hard-synthouter 转换(内部数据,本地实测)

- 本地只有 test split(120 行/87 repo);**train.jsonl(464MB/960 行)需下载**;audit 显示全集 870 repo;
- 结构:repo 是知识单元(同 repo 多行 = 同 commit 同 docs,仅外层题不同);retrieval_qas 短答结构化事实(unique fact/repo median 47,≥12 fact 的 repo 77/87),**train/eval 改写对**(同 fact 两种问法)天然防题面背诵;inner_docs 数百 KB/repo,天然要求分块+记忆;
- **外层编码题短期判死**:hidden_tests 是自然语言测试规格(要先 LLM 合成真实测试),rubric 需 LLM judge(管线无 judge 通道),87 repo 无 SIF 且快照未验证(content_compatible_proxy);
- **retrieval_qas 是完美自动判分对象**:短答 exact match,零容器零 judge;
- ⚠️ matplotlib/matplotlib 与现池重叠需剔除;tablib/tenacity 不在其中(heldout 无污染);
- **推荐方案 (a)**:episode = 同 repo「6 题学习(带 doc 片段)→ 6 题考试(eval 改写问法,不给 doc)」——考试时**记忆是唯一知识通道**,~800 repo 序列内独占,权重背不动;工程量 ~3 个文件 1-2 天(转换脚本 + studybench_qa task 类(exact match evaluator)+ rollout 加 QA 分支——最后一项是唯一集成风险点);方案 (b)(纯阅读轮+考试)做 (a) 的 schedule 消融;方案 (c)(外层题容器判分)2-4 周起,不做或远期。

## 5. 三轨行动方案(推荐执行顺序)

| 轨 | 内容 | 定位 | 工程量 | 存储 |
|---|---|---|---|---|
| **1. studybench QA**(先行) | 方案 (a),~800 repo/5.9 万 fact | **最低成本验证核心假设**:数据重构后 WRITE→READ 能否自举(此处 D 检验拉不起来,SWE 上更没戏) | 1-2 天 | ~0.5GB |
| **2. SWE-smith top-20~30 repo**(主力) | 每 repo ≥400 题,序列按**模块依赖拓扑序**(题按 patch 文件聚类 + import 图排序;呼应 AgentCL 警告),pipeline 增发保证每题只见 1-2 次 | 大池防背题的主训练数据,判分与现管线同构 | 1-2 周 | ~60GB SIF(20 镜像,全场最低) |
| **3. R2E-Gym / SWE-rebench-V2 真实时间序**(评测侧) | R2E 10 repo 全≥108 题真实 commit 时间线;rebench 73 repo≥50 题自带 created_at 且题新 | "合成训练→真实序列评测"迁移实验(pandas/moto 三数据集共现,天然对照) | 中 | per-instance SIF ~1TB,**待存储解决** |
| (+) 演化链数据 | LongChain-50/SWE-Milestone 直用 + 复用其 pipeline 自产 | 题间强依赖的课程片段/专项消融 | 中 | 141GB(LongChain 全量,可只取 Python 链) |

设计红线(来自 AgentCL + 五检验教训):序列必须显式构造知识复用(依赖序/共享文件),不能 naive 时间序;每题曝光次数做硬预算(≤2);heldout 用时间/仓库双隔离;门控只保护它测量的维度——想要的性质要么进 reward 要么由数据结构内生必要。

## 6. 复现与来源

三份原始调研报告(含全部链接与核查过程)由 subagent 产出,关键来源:SWE-smith/R2E-Gym/SWE-rebench/SWE-Gym 的 HF 卡与论文、swesmith.com、Together DeepSWE blog、rLLM;SWE-EVO/SWE-Milestone/SWE-Chain/SWE-Context-Bench/CL-Bench/AgentCL/LifelongAgentBench/GitChameleon/SWE-Exp 的 arXiv;studybench 本地 test.jsonl+audit.json 实测(脚本可参 ~/probe/)。Nebius RL 配方参考:arXiv 2508.03501(7,249 题过滤法,72B 11.4%→39.0%);reward 沿用二值 F2P 是各家共识。
