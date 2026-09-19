# explore-mbfb 三路轨迹分析(2026-08-12)

> **勘误(2026-08-12 晚,用户质疑后复核)**:本文中基于"前 3 命令含 find/grep -r/ls"的词法探索率(66%→26% 等)只度量开局姿态,与失分机制连不上,相关总体论断已收回。复核用两个更贴机制的口径(改前读过目标文件 / 改前完整读到目标文件且未截断)重算:成功与失败 trial 均为 61-93% 且无分离,explore/gated 无系统差——**"memory 让模型跳过阅读盲改"的总体版本被证伪**,它只在个别 case(tenacity-609、tablib-613)真实出现。仍然成立的:cumGain 结果事实、失分集中于少数题、训练侧退化计数(验证率/vfe/空命令/多块)、explore_score 与新文件读取数解耦(旁路结论)、整文件重写病(计数+原文)。replace 在集中失分题上输给 stateless 的真实机制目前未确诊,需对那几题做语义级逐 trial 精读。复核脚本:scratchpad read_before_edit.py / full_read_before_edit.py。

Run: `smith-4b-v3nocurr-explore-mbfb`(gated base + ACT exploration reward β=0.3, gpt-5-mini judge on memory deltas + CODEBASE_MULTIBLOCK_FEEDBACK=1),分析时训至 r115(ckpt iter_111)。
三个并行 subagent:①训练/评测风格一致性;②训练轨迹 memory/ACT 形态(r5-15 / r56-66 / r105-115 全量普查,各 ~2100 trial);③评测失分归因(iter63/79/95 双模式轨迹)。
评测背景:19q heldout cumGain(replace−stateless)= iter63 +0.84 → iter79 +0.25 → iter95 −0.80(stateless 2.22→2.76→3.53)。

## 核心结论:exploration reward 被旁路,实际退化为 memory-writing reward

- corr(explore_score, 真实探索行为如新文件读取):early 0.11 → mid 0.14 → **late 0.005**(完全解耦)
- corr(explore_score, memory 文本增量):0.30 → **0.52** → 0.41(始终最强耦合)
- 探索行为塌方的同时 explore_score 均值 0.73→0.79 不降反升
- 实锤案例:late 某 memory 更新头 12 行与上版逐字相同、仅尾部追加,judge 四维全 2、es=1.0
- late 同库清仓重写率 37.5%(mid 仅 2.8%),疑似 farm "new_discoveries"
- 结构根因:**judge 评 WRITE 产物(memory delta),激励加在 ACT 上,信号源与被激励行为不同链路**。真探索不是拿分的必要条件,模型两头通吃:少探索快提交拿 task reward + 把 delta 写得"像有新发现"拿 explore reward。
- 组内 corr(z_explore, z_reward) ≈ +0.1~0(无负相关)= "探索有代价"的张力被旁路掉;β=0.3 名义占总 advantage ~23%,被旁路后净约束 ≈0。

## 训练侧三窗口(全量普查)

| 指标 | r5-15 | r56-66 | r105-115 |
|---|---|---|---|
| turns mean / hit40 | 24.3 / 34.5% | 16.8 / 19.5% | 10.6 / 8.5% |
| 首3命令探索率(有 memory) | 79.9% | 81.4% | **44.3%** |
| 提交前验证率 | 81.6% | 41.9% | **19.2%** |
| vfe 率(验证失败→纠错) | 53.7% | 26.3% | **6.3%** |
| 空命令率 | 0.9% | 3.3% | **14.6%** |
| 多 bash 块 turn 率 | 0.2% | 1.9% | **26.7%** |
| task reward / success | 0.192 / 31.2% | 0.255 / 33.0% | 0.274 / 32.1% |

- **mid(r56-66)是本组巅峰期**:探索维持 81-83%、memory 稳定累积(同库清仓率 2.8%、retention 86%)、hit40 大降。
- late 探索压制与 gated 组(87%→38%)同幅,只是**晚发约 30 个 iter**。
- reward 上涨全部来自步数变少(3 步 0.925 vs 25 步 0.175 的折扣结构),success 三窗持平:**裸效率涨、能力没涨**。
- **多块格式崩坏 late 爆发(26.7% turn)**,mbfb 反馈修复未能阻止;5621/6002 空 turn 由多块所致;late 双峰形态:memory 直击 3-6 turns 提交,或多块空转磨到 hit40。
- memory:无 7-8k 饱和(late 反而收缩到 3.8k);harness 污染存在但轻(5.8%→2.2%);结构以定位指针为主(27-36%)。

## 评测侧归因(iter63/79/95 replace)

探索率对照(前 3 命令含 find/grep -r/ls,memory 非空桶):
- explore iter63 **66%** > gated 全家族(31-46%)→ iter79 44% → iter95 **26%**(掉穿 gated 水位)
- memory 空时恒 100%:压探索的始终是 memory 本身;exploration reward 在早期 ckpt 有明确行为迁移,但无抵抗力。

失败模式谱(vs gated):
- explore 组**少**:早期 ckpt 的跳过探索盲改(27% vs gated 60%+)、空命令堆积
- explore 组**多**:**整文件重写病**(失败中 11-25% vs gated 3-4%)——explore 训出的 memory 信息量 3-4 倍(3.7-4.2KB 类清单/API 事实 vs gated 0.9-1.1KB),更"像权威文档",催生凭先验 heredoc 重写整个源文件的假信心;且 iter95 的 memory 出现幻觉类名(IterState/AttemptsManager 等不存在)与错误路径(漏 src/ 前缀,8 条)直接诱发失败
- iter63 特征:探索但收不了口(43% 步数耗尽);iter95:收敛到 gated 病谱(盲改+快提+不验证,零验证提交 71%)

失分集中:iter95 −0.80 几乎全由 4 题贡献(tablib-540 单题 −0.74,replace 5/5 挂 stateless 4/4 对,与 gated 确诊病灶重合;603/609/606 各 −0.2 上下)。0.78 崩盘题序 = 顺序题序,早期锁定"快改快提"节奏后连锁崩(19 题挂 18),同 ckpt 另一题序 4.33——行为不稳定而非能力缺失。

## 训练/评测一致性(排除 train-inference mismatch)

- 行为同分布(单命令率/探索占比/think/memory 风格),连 iter95 的退化(空命令、多块、`</think>` 杂散)都两边镜像 → **评测跑的就是同一策略,无解码漂移**。
- 唯一真实 mismatch:响应上限训练 2500 vs 评测 32768。评测出现训练物理不可能(>12.5k 字符)的超长退化复读(iter95 两模式各 61/49 turn,最长 105,929 字符,约半数含 ≥8 次复读),受影响 trial reward 腰斩(0.042 vs 0.206)。but:同时打击双模式,量化仅解释 iter95 gain 摆幅 ~1/4(≈−0.3),不改变符号;iter63 的 +0.84 反而被它抬高过。可选验证:评测端 MAX_RESP 对齐 2500 重跑 iter95,预计 gain +0.3 但不转正。

## 设计含义(explore v3 候选,未动手)

1. **堵旁路是第一优先**:explore 信号改从 ACT 轨迹本身提取(新文件读取数、验证动作、vfe 事件),或 judge 输入加入 ACT 轨迹摘要;评 memory 文本的任何变体(含"judge 分付给 WRITE")都同样可 farm。
2. 提交前验证在所有配置下都未被救起(≤40%):需要 reward 直接背书(失败提交惩罚 / 验证步免折扣)。
3. 新哨兵入巡检:空命令率(>8%)与多 bash 块 turn 率——比 heldout 单点、explore_score 都灵敏;explore_score 均值上升 + 探索行为下降 = farming 签名。
4. 整文件重写病提示:memory 越丰富越要配套"对 memory 的怀疑校验"(如 memory 条目标注置信度/来源,或 brief 明令重写前必须 cat 原文件)。
