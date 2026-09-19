# Recovered case-study reports (extracted from session transcript 2026-08-19 after node loss)



# ==== ACT_EVOLUTION_gwin (transcript line 45088, full 21905 chars) ====

# ACT 行为随训练的演化 — `smith-4b-v3nocurr-explore-gwin` trajectory case study

数据:train rollout 0..138(每个 rollout 16 episodes × 12 issues = 192 trials)+ 晚期 rollout 150/155/160。
所有 142 个 rollout 全量解析(`act_metrics.csv`,142 行);`act_extra.csv`/`act_format.csv` 为每 5 个 rollout 抽样(0,5,…,135,138,150,155,160)。
代码:`act_metrics.py`(主指标)、`act_extra.py`(edit 风格 / blind submit / group 多样性)、`act_format.py`(格式漂移)、`act_transcript.py`(transcript 压缩打印)、`act_index.py`(instance 索引)。原始摘录见 `act_excerpts.txt`。

**读数前提(重要 caveat)**
- 每个 rollout 只有 2 个 GRPO group(2 条 instance 序列,共 24 个不同 instance,分属 2~4 个 repo),所以**单 rollout 的 success/turns 主要反映当次抽到的题目难度**;下面凡是给"趋势"都用 20-rollout 窗口均值(3840 trials/窗口),末窗 150-160 只有 3 个 rollout(576 trials)且 150 是明显异常的一次(pandas+sqlglot+astroid+funcy 硬题组合)。
- 每个 episode 的 12 个 trial 是 2 个 repo × 6 题;**同 repo 内的 6 题按 mutation type 字母序排列**(combine_* → func_basic → func_pm_* → lm_rewrite → pr_*),所以 trial_pos 越靠后题越难,"位置效应"与"难度效应"是 confounded 的,不能直接拿 pos 5 vs pos 0 说 memory 的作用(第 1.4 节做了按 type 控制)。
- 命令分桶口径:`submit` = `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`;`edit` = 对 repo 文件写入(sed -i / python replace 脚本 / heredoc 覆盖 / cp,mv);`script` = 写 /tmp 或 test_/repro_ 类脚本文件(不算 repo 编辑);`test` = pytest / python 脚本 / python -c / heredoc 喂 python;`search` = grep/rg/find/ls;`view` = cat/head/tail/sed -n/nl;`empty` = harness 返回 "Empty command"。同一命令按 submit > edit/script > test > 首段命令类型 的优先级归一桶。
- reward 口径(从数据反推):成功时 reward = 1 − 0.025×(turns−1),失败为 0;所以 mean_reward 的上升既可来自成功率也可来自 turns 下降。
- "memory 引用" = assistant 文本匹配 /memory|as noted|previous issue|earlier|previously|my notes|…/i,含少量噪声(如代码里的 memory)。

---

## 1. 定量曲线

### 1.1 每 10 个 rollout 的关键指标(单 rollout,噪声大;完整表见 `act_metrics.csv`)

| rollout | succ | reward | turns | budget% | length% | assist chars | empty cmd% | memref% | view/search/edit/script/test/submit % | trials w/ test% | tested-before-submit% | blind-submit-after-edit% | py_replace style% | no-reasoning% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.35 | 0.16 | 31.6 | 55 | 0.0 | 551 | 1.2 | 1.4 | 28/22/11/10/26/2 | 96 | 99 | 2 | 17 | 10 |
| 10 | 0.38 | 0.22 | 25.1 | 34 | 0.0 | 529 | 1.6 | 1.8 | 26/21/15/7/27/3 | 92 | 90 | 8 | 12 | 7 |
| 20 | 0.21 | 0.13 | 20.7 | 23 | 0.1 | 534 | 0.5 | 3.2 | 29/28/15/2/21/4 | 57 | 47 | 32 | 14 | 3 |
| 30 | 0.16 | 0.11 | 20.3 | 24 | 0.1 | 488 | 1.4 | 6.3 | 33/28/14/2/16/4 | 48 | 35 | 50 | 16 | 7 |
| 40 | 0.17 | 0.13 | 16.7 | 17 | 0.1 | 610 | 0.9 | 3.9 | 34/26/16/5/14/5 | 41 | 31 | 48 | 19 | 8 |
| 50 | 0.30 | 0.21 | 21.2 | 29 | 0.4 | 698 | 1.0 | 2.7 | 33/22/14/4/23/3 | 54 | 38 | 31 | 27 | 15 |
| 60 | 0.33 | 0.22 | 24.6 | 38 | 0.2 | 645 | 1.5 | 2.8 | 33/21/8/14/19/3 | 80 | 77 | 7 | 39 | 11 |
| 70 | 0.27 | 0.18 | 23.3 | 37 | 0.5 | 1005 | 3.2 | 2.3 | 33/18/15/10/18/3 | 80 | 70 | 13 | 39 | 9 |
| 80 | 0.33 | 0.26 | 17.9 | 26 | 0.6 | 1036 | 2.0 | 2.9 | 30/18/14/14/18/4 | 68 | 60 | 38 | 40 | 7 |
| 90 | 0.39 | 0.31 | 17.7 | 24 | 0.6 | 822 | 1.3 | 1.4 | 31/22/13/9/19/4 | 62 | 55 | 35 | 52 | 12 |
| 100 | 0.34 | 0.27 | 12.3 | 14 | 0.7 | 915 | 2.8 | 2.4 | 35/19/14/11/9/7 | 22 | 15 | 85 | 69 | 16 |
| 110 | 0.41 | 0.33 | 13.0 | 15 | 0.6 | 838 | 1.1 | 1.8 | 42/18/14/8/10/7 | 18 | 7 | 89 | 69 | 39 |
| 120 | 0.43 | 0.37 | 12.0 | 15 | 0.6 | 932 | 1.6 | 4.6 | 34/23/13/7/16/7 | 28 | 15 | 88 | 54 | 23 |
| 130 | 0.38 | 0.32 | 13.3 | 19 | 0.5 | 871 | 1.8 | 4.0 | 39/20/10/7/17/6 | 26 | 11 | 88 | 59 | 32 |
| 135 | 0.19 | 0.17 | 19.4 | 33 | 0.4 | 790 | 1.0 | 3.0 | 35/20/8/17/14/3 | 37 | 15 | 86 | 57 | 37 |
| 138 | 0.41 | 0.32 | 19.9 | 31 | 0.3 | 682 | 0.6 | 1.7 | 37/15/13/10/21/4 | 56 | 38 | 62 | 55 | 56 |
| 150 | 0.27 | 0.20 | 26.8 | 50 | 0.1 | 505 | 0.2 | 1.1 | 30/13/5/35/15/2 | 55 | 47 | 64 | 55 | 76 |
| 155 | 0.58 | 0.49 | 11.2 | 10 | 0.6 | 734 | 1.9 | 4.0 | 32/18/21/7/11/8 | 24 | 17 | 76 | 56 | 21 |
| 160 | 0.46 | 0.37 | 16.2 | 24 | 1.2 | 740 | 8.9 | 5.2 | 28/21/14/13/13/5 | 44 | 27 | 78 | 42 | 6 |

(`tested-before-submit%` = 提交的 trial 中,submit 前跑过任何 test/run 命令的比例;`blind-submit-after-edit%` = 提交的 trial 中,最后一次 edit 之后既没 view 也没 test 就直接 submit 的比例;`no-reasoning%` = assistant 输出以 ``` 开头 / 代码块前没有任何文字的 turn 占比。)

### 1.2 20-rollout 窗口均值(全部 142 个 rollout;主体结论以此为准)

| window | succ | reward | turns | budget% | length% | assist chars | empty% | memref% | syntax-err% | nonzero-rc% | submitted% | trials w/ edit% | trials w/ test% | tested-before-submit% | first-edit turn | edits/trial | view/search/edit/script/test/submit % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 000-019 | 0.296 | 0.166 | 26.3 | 38.7 | 0.1 | 568 | 0.9 | 1.8 | 2.5 | 15.6 | 62.6 | 91.7 | 86.6 | 82.4 | 8.4 | 3.8 | 28/23/15/7/24/3 |
| 020-039 | 0.191 | 0.141 | 16.0 | 15.2 | 0.1 | 571 | 1.4 | 4.5 | 3.5 | 13.5 | 85.5 | 96.0 | 38.6 | 30.3 | 5.9 | 3.2 | 30/25/21/3/14/6 |
| 040-059 | 0.287 | 0.211 | 17.3 | 20.5 | 0.2 | 614 | 1.5 | 2.9 | 2.2 | 12.5 | 80.4 | 94.8 | 46.3 | 37.0 | 6.4 | 2.8 | 34/23/17/4/16/5 |
| 060-079 | 0.315 | 0.221 | 20.7 | 28.0 | 0.4 | 798 | 2.1 | 2.7 | 2.1 | 13.6 | 72.8 | 93.2 | 68.2 | 60.0 | 8.0 | 2.5 | 33/20/12/9/18/4 |
| 080-099 | 0.310 | 0.236 | 17.0 | 22.8 | 0.6 | 884 | 2.3 | 2.4 | 1.8 | 12.8 | 77.7 | 93.6 | 54.4 | 44.1 | 6.7 | 2.2 | 32/20/13/10/18/5 |
| 100-119 | 0.325 | 0.275 | 10.5 | 10.4 | 1.0 | 958 | 2.3 | 2.7 | 2.2 | 11.3 | 89.9 | 95.3 | 17.9 | 10.6 | 4.6 | 1.7 | 35/19/16/9/10/9 |
| 120-139 | 0.342 | 0.288 | 14.0 | 18.6 | 1.2 | 1018 | 2.1 | 4.6 | 2.6 | 11.9 | 81.6 | 90.4 | 28.5 | 15.6 | 5.8 | 1.6 | 35/21/13/10/13/7 |
| 150-160 | 0.436 | 0.355 | 18.1 | 28.1 | 0.6 | 660 | 3.7 | 3.4 | 3.7 | 12.5 | 72.6 | 88.7 | 41.1 | 30.4 | 6.9 | 2.0 | 30/17/14/19/13/5 |

读法:
- **success 基本平的(0.30 → 0.19 dip → 0.29~0.34),reward 从 0.17 涨到 0.29 几乎全靠 turns 从 26 降到 10~14**(reward = 1−0.025×(turns−1),20 个 turn 值 0.5)。150-160 的 0.44 只有 3 个 rollout,其中 155 是很容易的一批(sqlfluff/python-docx/radon),不作趋势看。
- turns 的下降不是单调的:0-19 → 20-39 骤降(26→16),60-79 回弹到 21,100-119 掉到 10.5(最激进),120 之后又回到 14~18。这条曲线和"验证习惯"曲线是镜像的(见下)。
- `frac_length`(act_finish=length)始终 <1.3%,不是主要问题;但 100+ 后逐渐上升(0.1%→1.2%),和 assistant 文本变长同步(568→1018 chars/turn)。
- `empty cmd`(harness 收到空命令)从 0.9% 升到 2~4%,rollout 160 突增到 8.9%(见 §3 格式漂移)。
- memory 引用整体很低(1.8% → 4.6% 的 turn),主要集中在 T1(见 §1.4)。
- 命令分布:view 28%→35%、search 23%→17~21%、test 24%→10~13%、submit 3%→7~9%;edit 占比稳定 13~16%,但 edits/trial 从 3.8 降到 1.6 —— 即**一次 edit 就提交**成为常态。

### 1.3 验证/提交习惯(act_extra.csv,抽样 rollout)

| rollout | 提交前跑过测试% | 最后一次 edit 后 blind submit% | edit 后先 view 再提交% | 首次 submit 的中位 turn | edit 风格 sed -i% | edit 风格 python replace 脚本% | heredoc 覆盖% | group 内 T1 命令去重率 | group 内前 4 步 pattern 去重率 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 97 | 2 | 1 | 21 | 54 | 17 | 21 | 0.87 | 0.65 |
| 20 | 37 | 32 | 24 | 12 | 79 | 14 | 2 | 0.98 | 0.73 |
| 40 | 13 | 48 | 28 | 9 | 63 | 19 | 10 | 0.95 | 0.73 |
| 60 | 74 | 7 | 18 | 14 | 39 | 39 | 5 | 0.72 | 0.59 |
| 80 | 58 | 38 | 4 | 8 | 13 | 40 | 34 | 0.66 | 0.63 |
| 100 | 10 | 85 | 4 | 5 | 2 | 69 | 23 | 0.49 | 0.47 |
| 120 | 5 | 88 | 1 | 5 | 1 | 54 | 34 | 0.71 | 0.64 |
| 130 | 8 | 88 | 3 | 5 | 2 | 59 | 22 | 0.71 | 0.54 |
| 138 | 34 | 62 | 4 | 7 | 5 | 55 | 16 | 0.66 | 0.48 |
| 150 | 35 | 64 | 0 | 11 | 0 | 55 | 28 | 0.86 | 0.64 |
| 155 | 10 | 76 | 11 | 5 | 10 | 56 | 5 | 0.85 | 0.59 |
| 160 | 16 | 78 | 5 | 5 | 12 | 42 | 9 | 0.92 | 0.75 |

- **"改完就交"是最显著的行为漂移**:blind-submit-after-edit 从 2% → 85~91%(rollout 100-135),晚期 62~78%。首次 submit 的中位 turn 从 21 降到 5。
- edit 工具从 `sed -i`(早期 54~79%)整体迁移到 "写一个 /tmp/fix_x.py 做 `content.replace(old,new)` 再 `python /tmp/fix_x.py`"(rollout 100+ 占 55~69%)。这些脚本在 100+ 后 ~90% 带 `if old in content ... else print('not found')` 之类的守卫,可见失败率约 9~16%;**看到失败后模型不会直接提交**(FAILED→submit = 0%,会 view/edit 重试),但**看到 "Done" 后 44~56% 下一步就是 submit**(early 0~1%)。
- GRPO group 内多样性在 rollout 90-100 附近最低(8 个 episode 的 T1 命令去重率 0.49,前 4 步 pattern 去重率 0.47),即同一题 8 条 rollout 高度同构(见 case 2/3);120 后又略回升。

### 1.4 按 trial_pos 的成功率(memory 是否让"后面的题"更受益?)

20-rollout 窗口均值,pos 0-5 是 repo A、6-11 是 repo B(memory 跨全部 12 题累积):

| window | pos0 | pos1 | pos2 | pos3 | pos4 | pos5 | pos6 | pos7 | pos8 | pos9 | pos10 | pos11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 000-019 | 0.49 | 0.36 | 0.31 | 0.17 | 0.22 | 0.21 | 0.46 | 0.37 | 0.31 | 0.22 | 0.26 | 0.18 |
| 040-059 | 0.44 | 0.38 | 0.31 | 0.17 | 0.10 | 0.14 | 0.48 | 0.44 | 0.35 | 0.21 | 0.19 | 0.23 |
| 080-099 | 0.57 | 0.44 | 0.28 | 0.26 | 0.13 | 0.20 | 0.50 | 0.41 | 0.26 | 0.25 | 0.20 | 0.23 |
| 100-119 | 0.65 | 0.46 | 0.32 | 0.21 | 0.27 | 0.12 | 0.49 | 0.38 | 0.30 | 0.24 | 0.19 | 0.27 |
| 120-139 | 0.46 | 0.49 | 0.37 | 0.26 | 0.29 | 0.21 | 0.58 | 0.49 | 0.36 | 0.23 | 0.16 | 0.20 |

表面上是"越靠前越涨、越靠后不涨",但如上所述 pos 与 mutation type 绑定。**按 type 控制后**(rollout ≤20 vs ≥120,同 type 同 repo 内位置):

| type × 位置 | early succ / turns (n) | late succ / turns (n) |
|---|---|---|
| func_basic, repo 首题 | 0.45 / 23.9 (264) | 0.42 / 12.0 (206) |
| func_basic, 第 2-3 题 | 0.40 / 22.6 (704) | **0.53** / 10.4 (700) |
| func_pm_ctrl_shuffle, 首题 | 0.46 / 24.7 (336) | 0.61 / 9.0 (384) |
| func_pm_ctrl_shuffle, 第 2-3 题 | 0.36 / 23.1 (152) | 0.60 / 9.9 (290) |
| combine_file, 第 4-6 题 | 0.25 / 24.2 (336) | 0.15 / 11.6 (279) |
| lm_rewrite, 第 4-6 题 | 0.16 / 31.0 (480) | 0.13 / 22.5 (406) |
| func_pm_remove_cond, 第 4-6 题 | 0.20 / 32.8 (176) | 0.18 / 18.0 (144) |

结论:**同 type 下"非首题"的提升(func_basic +13pt、ctrl_shuffle +24pt)略大于"首题"(−3pt / +15pt),与 memory 有一点正向作用相容,但幅度小且噪声大;真正一致的是所有 type 的 turns 都腰斩,而 hard type(lm_rewrite / combine_file / remove_cond)成功率持平或下降。** memory 的可见使用形式主要是 T1 直接 `cat`/`grep` 到 memory 里记的文件(T1 跳过 search 直接 view/edit 的比例 8% → 35~47%;T1 文本显式提 "Based on memory" 的比例 7% → 21~31%),而不是复用验证/测试方法。

### 1.5 按 mutation type 的 success / turns(窗口均值,n=trials)

| window | func_basic | func_pm_ctrl_shuffle | combine_file | lm_rewrite | func_pm_ctrl_invert_if | func_pm_remove_assign | combine_module | func_pm_remove_cond | pr |
|---|---|---|---|---|---|---|---|---|---|
| 000-019 | 0.42/23.2 (960) | 0.43/24.4 (496) | 0.20/25.8 (496) | 0.17/32.6 (424) | 0.25/30.3 (192) | 0.13/26.0 (168) | 0.10/25.1 (200) | 0.19/33.9 (160) | 0.15/32.0 (136) |
| 040-059 | 0.41/14.2 (960) | 0.42/14.4 (600) | 0.13/15.3 (504) | 0.18/27.6 (336) | 0.25/20.4 (216) | 0.06/22.4 (152) | 0.04/17.4 (160) | 0.25/21.4 (136) | 0.28/15.7 (104) |
| 080-099 | 0.43/15.7 (848) | 0.56/10.8 (528) | 0.11/14.9 (384) | 0.14/27.6 (432) | 0.33/19.0 (184) | 0.33/18.0 (224) | 0.11/14.2 (192) | 0.17/20.6 (192) | 0.25/20.7 (120) |
| 100-119 | 0.47/10.6 (832) | 0.48/ 7.1 (632) | 0.15/ 8.8 (344) | 0.14/15.3 (352) | 0.26/ 8.5 (176) | 0.25/ 9.9 (232) | 0.01/11.7 (128) | 0.10/18.5 (152) | 0.17/15.5 (88) |
| 120-139 | 0.47/11.0 (852) | 0.58/ 9.2 (570) | 0.09/11.0 (350) | 0.14/21.8 (374) | 0.44/16.7 (135) | 0.20/15.8 (104) | 0.10/13.0 (174) | 0.14/16.4 (120) | 0.14/16.6 (136) |

→ 提升集中在单点小改(func_basic、ctrl_shuffle、invert_if);多文件/重写类(combine_*、lm_rewrite)不涨甚至跌,但 turns 从 25~33 砍到 9~22 —— 对这些题,"快速放弃/快速误交"替代了"40 步耗尽"。

---

## 2. Case study(同 repo、同 mutation type;instance 在训练中不重复,找不到完全相同的 instance_id)

摘录格式:`Tn [桶] 命令 → 观察一行`。完整版见 `act_excerpts.txt`。

### Case 1 — cantools `func_basic`:rollout 0 `func_basic__fnvf3wq5`(SYM `_dump_signals`) vs rollout 124 `func_basic__xqus9738`(C header guard + fuzzer makefile)

**早期(rollout 0,8 条中 3 条成功,平均 38.4 turns;下面是一条 40 turn 后才 PASS 的)**
```
T01 [search] ls -la /testbed                                   → 目录列表
T02 [search] find /testbed -name "sym.py"                       → ./src/cantools/database/can/formats/sym.py
T03 [view  ] cat …/sym.py                                       → output too long
T04 [view  ] sed -n '/def _dump_signals/,/^def /p' … | head -100 → 看到函数
T05 [test  ] python3 -c "import cantools; db=load_file(...)"     → rc=1 Traceback(文件不存在)
T06 [search] find /testbed -name "*.sym" | head                 → 找到 tests/files/sym/*.sym
T07 [test  ] python3 -c "db=load_file('tests/files/sym/send-6.0.sym'); print(dump)" → 复现
… 33 more turns(反复 sed -i / 重跑脚本 / pytest -k sym),T40 submit → PASSED (reward 0.025)
```
另一条(失败):`sed -i '855,873s/.*/…'` 触发 bash 语法错(rc=2 unexpected EOF),换成 /tmp/fix_sym.py 脚本,之后在 pytest 反复中耗尽 40 步。

**晚期(rollout 124,8 条全部失败,4~8 turns)** — 8 条几乎同构:
```
T01 [search] find . -name "*.py" | xargs grep -l "c_source\|generate\|header" | head -5 → c_source.py 等
T02 [view  ] cat -n ./src/cantools/database/can/c_source.py     → **output too long(根本没看到代码)**
T03 [edit  ] cat <<'EOF' > /tmp/fix_c_source.py … content.replace(
             "    include_guard = f'{database_name.lower()}_h'  # Changed to lowercase.", …) … EOF && python /tmp/fix_c_source.py
                                                                → rc=0 (no output)
T04 [submit] echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT         → Submission FAILED (tests_failed)
```
点评:
- 早期是"过度探索 + 反复验证"型:T5 就开始写复现脚本,test 命令占 26%,但工具使用笨拙(sed 多行替换语法错、`cat` 整文件被截断),40 步花完是主要失败模式(rollout 0 有 55% 的 trial 撞 budget)。
- 晚期是"盲改 + 秒交"型:T2 输出被截断、**模型并未看到源码**,T3 却在 replace 脚本里写出带 `# Changed to lowercase.` 注释的"旧代码"(是对 SWE-smith 注入注释风格的猜测/幻觉),`str.replace` 无守卫、静默不匹配,T4 直接提交。整个 group 8/8 失败 → 该题 GRPO advantage 全 0,没有梯度告诉它这样不对。
- 同一题 8 条 rollout 的前 3 步命令类型完全一致(search→view→py_replace→submit),多样性坍缩。

### Case 2 — sqlglot `func_pm_ctrl_shuffle`:rollout 0 `rq4xvhqf`(`_add_replace_columns` UnboundLocalError) vs rollout 150 `48dpisag`(`eliminate_qualify` 里的 `alias_or_name`)

**早期(rollout 0,pos 6,memory_in 2.1k chars;成功,20 turns,reward 0.525)**
```
T01 [search] find . -name "*.py" | xargs grep -l "REPLACE" | head -20      → 若干文件
T02 [search] grep -n "def qualify_columns" sqlglot/optimizer/*.py           → qualify_columns.py:19
T03 [view  ] cat sqlglot/optimizer/qualify_columns.py                       → output too long
T04 [search] grep -n "_add_replace_columns" …                               → 640/646/744
T05 [view  ] sed -n '744,800p' …                                            → 看到函数
T06 [script] cat <<'EOF' > /testbed/test_replace_bug.py …                   → (写复现)
T07 [test  ] python test_replace_bug.py                                     → rc=1 UnboundLocalError(复现成功)
T08 [view  ] sed -n '744,760p' …                                            → 再看一遍
… T9-T19: sed -i 修改 → 复现脚本转绿 → 边界用例脚本 → pytest;T20 rm 测试文件 && submit → PASSED
```
memory 有 2k 字符,但 T1 仍从 `find | xargs grep` 起手,memory 没有改变探索路径。

**晚期(rollout 150,pos 6,memory_in 1~3.4k chars;8 条中 8 条成功,3 turns,reward 0.95)** — 8 条 T1 命令逐字相同:
```
T01 [view  ] cat -n sqlglot/transforms.py | grep -A 100 "def eliminate_qualify"   → 直接看到函数(文件名来自 issue 文本)
T02 [edit  ] cat <<'EOF' > /tmp/fix_qualify.py … old_func='''def eliminate_qualify…''' new_func=… content.replace … EOF && python /tmp/fix_qualify.py → Done
T03 [submit] echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT                          → PASSED
```
点评:
- 这是 policy 学到的"理想路径":定位(1 步,直接 view 而非 search)→ 整函数替换(1 步)→ 提交,不复现、不跑测试。对 ctrl_shuffle 这种"看一眼就知道语句顺序错了"的题,reward 从 0.5 涨到 0.95,完全来自 turns。
- 代价是没有任何验证环节;整个 group 8 条 T1~T3 命令类型完全一样,只有 replace 脚本里 new_func 的文本略有不同(有的把 `return` 移位、有的重写 helper),GRPO 的 within-group 方差只剩"这次的整函数重写对不对"。

### Case 3 — astroid `lm_rewrite`:rollout 6 `08v06c41`(typing.NamedTuple brain 注册) vs rollout 125 `p9cqd5n6`(`util.safe_infer` 对 Uninferable 返回 None)

**早期(rollout 6,pos 5,memory_in 2~4.5k;8/8 失败,全部 40 turns)**
```
T01 [search] find /testbed -name "brain_namedtuple*" && grep -r "typing.NamedTuple" … → brain_namedtuple_enum.py
T02 [view  ] cat …/brain_namedtuple_enum.py                → output too long
T03 [view  ] sed -n '280,350p' …                           → 片段
T04 [view  ] sed -n '440,490p' …                           → 片段
T05 [view  ] sed -n '680,750p' …                           → 片段
T06 [view  ] sed -n '350,430p' …  T07 sed -n '358,410p'  T08 sed -n '420,480p' …
… 32 more turns,最后一条仍是 [view] sed -n '590,610p' → Max steps (40) reached
```
另一条:T1 `find /astroid …`(路径幻觉,rc=1),T2 `find / && find / -name …`(输出爆炸被截断),T3 才用 `$PWD` 找到文件;然后同样陷入 sed -n 翻页循环。**"只读不改"的 budget 耗尽**是早期 lm_rewrite 的典型死法(0-19 窗口 fail_budget_never_edited 5.7%,但 lm_rewrite 单类里 40-turn 失败占多数)。

**晚期(rollout 125,pos 11,memory_in 0.7~2.5k;8 条中 7 条成功,3~5 turns)**
```
T01 [view  ] cat -n astroid/util.py | grep -A 20 "def safe_infer"       → 看到函数(路径来自 issue)
T02 [edit  ] cat <<'EOF' > /tmp/fix_safe_infer.py … content.replace(old_code,new_code) … EOF && python /tmp/fix_safe_infer.py → Done
T03 [submit] echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT               → PASSED
```
失败的那条:T1 `find … -exec grep -l "def safe_infer\|Uninferable"` → T2 view → T3 replace 脚本(改法不同:只改 isinstance 分支返回值)→ T4 submit → tests_failed。另一条 T1 把 `cat <<'EOF' > /tmp/test.py && cat /tmp/fix.py` 拼在 heredoc 头上 → harness 判为 "Empty command"(浪费 1 步),之后照常 3 步搞定。
点评:这题(单函数、issue 已点名文件)恰好落在晚期 policy 的舒适区;而与 rollout 6 那种"要读懂 700 行 brain 文件里多个 transform 的注册关系"的 lm_rewrite 相比,晚期 policy 对后者的处理是"随便改一处然后交"(120-139 窗口 lm_rewrite 0.14/21.8 turns,与早期 0.17/32.6 相比成功率没变、只是死得快)。

### 补充:中期样本(rollout 65,pdfminer `func_basic`,成功,10~12 turns)
```
T01 [search] find . -name "*.py" -exec grep -l "get_widths" {} \;   → pdffont.py
T02 [view  ] cat -n ./pdfminer/pdffont.py | grep -A 50 "def get_widths"
T03 [script] cat <<'EOF' > test_issue.py …(复现)
T04 [test  ] python test_issue.py                                   → 复现:期望 {32:250,…} 实际 {33:250,…}
T05 [edit  ] cat <<'EOF' > fix.py … content.replace(...)             → (no output)
T06 [test  ] python test_issue.py                                    → 仍错(replace 未命中)
T07 [view  ] nl -ba pdffont.py | sed -n '65,78p'                     → 看真实文本
T08 [edit  ] sed -i …  T09 test → 绿  T10 submit → PASSED
```
60-79 窗口是"验证习惯"短暂回潮期(tested-before-submit 60%,blind submit 7~13%):有复现、有 edit 后回归,但已明显比 rollout 0 精炼(10 步 vs 20~40 步)。这段之后(100+)验证环节又被 reward 压掉。

---

## 3. 失败模式分类(每 trial 计一次;占该窗口全部 trial 的比例)

| window | 未提交耗尽 40 步 | 其中从未 edit 过 | 提交但 tests_failed | 提交 empty_patch | 提交时未 edit 过 | 提交前从未跑测试(所有提交) | ≤5 步就提交 | ≤5 步提交且失败 | 同一命令连续 ≥3 次 | ≥3 次 bash 语法错 | ≥1 次空命令 | ≥3 次空命令 | 出现 length 截断 | n_trials | 失败数 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 000-019 | 37.4% | 5.7% | 28.9% | 3.3% | 2.5% | 12.1% | 4.5% | 2.4% | 0.1% | 5.9% | 8.3% | 1.5% | 1.5% | 3840 | 2705 |
| 020-039 | 14.5% | 1.5% | 58.2% | 7.2% | 2.4% | 60.2% | 24.4% | 18.4% | 0.2% | 5.5% | 10.7% | 1.0% | 1.7% | 3840 | 3106 |
| 040-059 | 19.6% | 3.5% | 47.6% | 3.5% | 1.6% | 51.9% | 22.4% | 14.2% | 0.0% | 3.3% | 12.8% | 1.5% | 2.8% | 3840 | 2736 |
| 060-079 | 27.2% | 6.2% | 40.3% | 0.6% | 0.5% | 29.3% | 11.5% | 6.2% | 0.0% | 3.8% | 17.2% | 2.5% | 5.3% | 3840 | 2631 |
| 080-099 | 22.3% | 5.8% | 45.0% | 1.3% | 0.4% | 44.1% | 27.0% | 16.6% | 0.0% | 2.2% | 20.5% | 2.9% | 7.3% | 3840 | 2650 |
| 100-119 | 10.1% | 3.4% | 54.3% | 2.9% | 1.1% | 80.4% | 53.0% | 32.3% | 0.2% | 1.1% | 13.8% | 1.4% | 7.1% | 3840 | 2593 |
| 120-139 | 18.4% | 8.5% | 45.8% | 0.9% | 0.9% | 69.6% | 42.4% | 22.7% | 0.2% | 1.9% | 17.0% | 2.0% | 9.5% | 3612 | 2380 |
| 150-160 | 27.4% | 10.2% | 27.8% | 1.0% | 0.5% | 52.6% | 36.1% | 10.8% | 0.0% | 4.0% | 21.0% | 6.6% | 7.3% | 576 | 325 |

分期画像:
- **早期(0-19)**:失败主体是 **budget 耗尽(37%)** —— 复现脚本写错/API 用错反复 Traceback、`sed -i` 多行替换语法错(≥3 次语法错的 trial 5.9%)、`cat 整文件` 被截断后翻页;`<tool_call>`/`<|im_start|>` 格式残留(1.4% turn)导致空命令。提交的 trial 里 99% 提交前跑过测试。"重复同一命令 ≥3 次"的僵死循环几乎不存在(0.1%),模型是在忙碌地失败而不是卡死。
- **中期(20-59)**:第一次相变。turns 16、budget 耗尽降到 15%,但 **submit-tests_failed 升到 58%、≤5 步误交 18%**,提交前不跑测试 60%。40-59 稍回摆。60-79 出现验证回潮(untested 29%),80-99 又滑回 44%。
- **晚期(100-139)**:第二次相变、更极端:**80% 的提交没有跑过任何测试,53% 的 trial ≤5 步就交,32% 是 ≤5 步误交**;budget 耗尽只剩 10%。同时 "从未 edit 就耗尽"(只读循环)在 120-139 又回到 8.5%,length 截断 9.5%(reasoning 变长)、空命令 17%。
- **150-160**:150 是异常 rollout(pandas/sqlglot/astroid/funcy 硬题:50% budget 耗尽、35% 命令是写 /tmp/test*.py 复现脚本、76% 的 turn 没有 reasoning 直接给代码块);155/160 回到晚期常态,但 160 空命令 8.9%(≥3 次空命令的 trial 6.6%),原因见下。

格式漂移(`act_format.csv`,每 turn 比例):
- `<tool_call>` / `<|im_start|>` 残留:1.4%(r0)→ 基本消失(≤0.1%,r110+)—— 这是唯一被 RL 明确修掉的格式问题。
- **"Wait / Hmm / Actually," 自我纠正词**:3% → 15~24%(r80-130);assistant >3k chars 的 turn 1% → 5~12%。ACT 是 no-thinking 模式,但 reasoning 段逐渐长成 think 风格的自言自语。
- **无 reasoning、直接代码块**(违反 prompt 的 "Always reason first"):~8% → 30~56%(r110-138),r150 76%,r160 又回到 6%。
- **多个 ```bash 块 / 未闭合 fence**:r160 分别 5.8% / 4.5%(此前 <2%),并出现 `</think>` 泄漏(0.9%)—— r160 的空命令 8.9% 主要由此构成(harness 遇到多块/未闭合就判空)。晚期几个 checkpoint 之间格式表现波动很大(150 vs 155 vs 160 三个方向不同),不像稳定收敛。

---

## 4. 所见与结论

1. **ACT 学到的核心东西是"更短",不是"更对"。** 20-rollout 窗口的 success 从 0.30 → 0.19(dip)→ 0.29~0.34,基本平;mean turns 26 → 10~14;reward 0.17 → 0.29 的涨幅几乎全部可由 reward = 1−0.025×(turns−1) 的 turns 项解释。按 mutation type 看,func_basic +5pt、ctrl_shuffle +15pt、lm_rewrite/combine_file 持平或 −5~−10pt。
2. **验证环节被 reward 系统性压掉,而且是两次相变而不是缓慢漂移**:提交前跑过测试的比例 82%(0-19)→ 30%(20-39)→ 回潮到 60%(60-79)→ 11%(100-119)→ 16%(120-139);"最后一次 edit 后不看不测直接 submit" 2% → 85~91%(r100-135)。这与"≤5 步误交"从 2% 涨到 32% 是同一件事。turns 的曲线就是验证习惯曲线的镜像。
3. **工具风格从 `sed -i` 迁到"写 /tmp/fix.py 做 str.replace"(0-19: 15% → 100+: 55~69%)**,脚本大多带 `if old in content` 守卫,模型看到 "not found" 后不会提交(0%),会重试;但看到 "Done" 后 44~56% 直接提交。Case 1 显示极端情形:文件因 output-too-long 没看到,仍凭对 SWE-smith 注释风格的猜测写 old 串、无守卫 replace、静默不命中、提交失败,且 group 8/8 同构失败 → 无 advantage 信号。
4. **memory 的可见作用很小且形态单一**:显式引用 memory 的 turn 1.8% → 4.6%;主要体现在 T1 跳过 search 直接 view 已知文件(8% → 35~47%)。按 type 控制后,同 repo 内非首题的成功率提升(func_basic +13pt、ctrl_shuffle +24pt)略高于首题(−3pt/+15pt),方向与 memory 有益相容但幅度小、噪声大;没有看到 memory 让"验证方法/测试命令"被复用的迹象。
5. **GRPO group 内多样性在 r90-100 附近坍缩**(8 条 rollout 的 T1 命令去重率 0.87 → 0.49,前 4 步类型序列去重率 0.65 → 0.47),此时同一题 8 条 rollout 几乎都是 view→py_replace→submit,组内方差只剩"这次的整函数重写对不对";120 后略回升(0.7)。
6. **格式方面 RL 修掉了 `<tool_call>` 残留,但引入了新漂移**:reasoning 变长且出现 "Wait/Actually" 自言自语(3% → 15~24%)、无 reasoning 直接给代码块(8% → 30~56%,违反 prompt)、r160 出现多代码块/未闭合 fence/`</think>` 泄漏使空命令冲到 8.9%。晚期各 checkpoint 之间(138/150/155/160)在 turns、格式、验证比例上来回摆,不像已经稳定。
7. 诚实的噪声说明:单 rollout 只有 24 个 instance / 2 个 repo 组合,success 单点波动 ±20pt 是常态(r125 0.18 vs r120 0.43);上面所有"趋势"都只在 20-rollout 窗口(3840 trials)层面成立,150-160 只有 3 个 rollout、576 trials,其中 r150 明显是硬题组合,不要拿它当趋势终点。


# ==== WRITE_EVOLUTION_gwin (transcript line 45117, full 22369 chars) ====

# WRITE 输出与 write_reward 随训练的演化 — `smith-4b-v3nocurr-explore-gwin`

数据:train rollout 0..138 全量(每 rollout 16 episodes × 12 issues → 192 次 WRITE)+ 150/155/160(scratchpad 副本)。135/143/151/159 是 heldout eval(5 eps,不参与 write_reward 统计,只用于形态观察)。
每个 episode = 2 个 repo × 6 issues;`write_audit[k]` 是 issue k 之后的 memory 重写,`downstream_trial_pos=k+1`。
脚本:`write_extract.py`(抽取 27,318 条 WRITE 记录 → `write_records.jsonl`)、`write_analyze.py`(→ `write_metrics.csv`)、`write_content.py`(形态分类 → `write_content.csv`, `shape_examples.json`)、`write_cases.py`(→ `write_cases.txt`)、`write_reward_mech.py`、`write_grounding.py`。

先给结论,再给证据。

## 0. 结论速览

WRITE 输出经历了 5 个清晰的阶段,**没有出现"越写越长越空"的掏空**,而是先"越写越短、每次清仓"(r50–80),再"格式向 ACT 串味 + 占位句"(r90–155),最后在 r155→r159 之间**崩到 thinking 不闭合、raw thinking 直接漏进 memory**(r160 有 66% 的 memory 是 7–8k 字符的推理独白)。write_reward 从头到尾均值≈0,方差里一半以上来自 WRITE 无法控制的 prev-window(mean-reversion),与 memory 的任何内容特征相关系数 |r|<0.06;explore judge 分数一路**下滑**(0.75→0.45),没有向上漂移的 reward hacking。

## 1. 曲线(按 rollout;完整数字见 `write_metrics.csv`)

| rollout | ep_reward | mem chars(med) | resp_tokens | think 闭合率 | finish=length | format_ok | write_reward mean/std | ungated mean | near-copy(ratio>0.9) | heavy rewrite(ratio<0.5) | explore mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0   | 0.16 | 2451 | 977  | 1.00 | 0.005 | 0.84 | +0.039 / 0.23 | +0.085 | 0.02 | 0.60 | 0.73 |
| 20  | 0.13 | 1674 | 875  | 0.98 | 0.016 | 0.75 | +0.036 / 0.30 | +0.039 | 0.01 | 0.74 | 0.75 |
| 40  | 0.13 | 1597 | 1014 | 1.00 | 0.026 | 0.96 | +0.070 / 0.27 | +0.024 | 0.08 | 0.61 | 0.73 |
| 60  | 0.22 | 776  | 678  | 1.00 | 0     | 0.56 | +0.018 / 0.20 | -0.018 | 0.01 | 0.95 | 0.67 |
| 75  | 0.18 | 534  | 723  | 1.00 | 0.005 | **0.19** | -0.033 / 0.14 | -0.012 | 0.09 | 0.75 | 0.59 |
| 85  | 0.19 | 1354 | 1014 | 1.00 | 0.010 | 0.67 | -0.042 / 0.28 | -0.093 | **0.27** | 0.24 | 0.56 |
| 100 | 0.27 | 1535 | 1154 | 0.97 | 0.042 | 0.64 | +0.016 / 0.29 | -0.011 | 0 | 0.56 | 0.45 |
| 120 | 0.37 | 878  | 1095 | 0.98 | 0.036 | **0.17** | +0.002 / 0.12 | +0.096 | 0 | 0.87 | 0.51 |
| 138 | 0.32 | 1567 | 1055 | 1.00 | 0.010 | 0.77 | +0.025 / 0.30 | -0.017 | 0 | 0.61 | 0.41 |
| 150 | 0.20 | 2332 | 1330 | 0.96 | 0.099 | 0.82 | +0.022 / 0.27 | -0.015 | 0 | 0.60 | 0.56 |
| 155 | 0.49 | 2073 | 1423 | 0.87 | 0.208 | 0.77 | -0.033 / 0.35 | -0.010 | 0 | 0.63 | 0.57 |
| 160 | 0.37 | **7904** | **1855** | **0.34** | **0.72** | **0.25** | +0.032 / 0.20 | +0.067 | 0 | 0.94 | 0.59 |

(ungated = 不乘 format_ok 的 R = mean(r[k+1..k+3]) − mean(r[k−2..k]) + 0.1;heldout rollout 159:think 闭合 0.40、length 0.67、mem 7772 chars,与 160 一致。)

补充曲线(`write_content.csv`,每 5 个 rollout 抽样):

| rollout | clean 两节 | 前置 prose+两节 | prose+空```bash```+两节 | 无 header 纯 prose/带 bash 命令 | think 泄漏 | bullets/mem | RK / LP bullets | 上一版 bullet 保留率 | 占位句率 |
|---|---|---|---|---|---|---|---|---|---|
| 0   | 0.71 | 0.16 | 0.09 | 0.04 | 0    | 14.8 | 7.5/7.5 | 0.50 | 0.00 |
| 30  | 0.98 | 0.01 | 0.01 | 0.01 | 0    | 13.9 | 7.4/6.5 | 0.58 | 0.00 |
| 60  | 0.99 | 0.01 | 0    | 0    | 0    | 5.4  | 3.1/2.3 | **0.10** | 0.01 |
| 75  | 0.94 | 0    | 0    | 0    | 0.01 | **3.7** | 2.2/1.5 | 0.30 | 0.02 |
| 90  | 0.46 | 0.31 | 0.18 | 0.03 | 0    | 9.7  | 4.9/3.9 | 0.78 | 0.09 |
| 105 | 0    | 0.09 | 0.66 | **0.22** | 0.03 | 6.5 | 3.3/3.7 | 0.76 | 0.20 |
| 120 | 0    | 0    | 0.42 | **0.56** | 0.02 | 3.0 | 3.1/3.2 | 0.60 | 0.24 |
| 138 | 0    | 0.01 | **0.98** | 0.01 | 0    | 7.0 | 3.3/3.3 | 0.90 | 0.34 |
| 155 | 0    | 0.07 | 0.76 | 0.03 | 0.14 | 10.9 | 4.5/4.6 | 0.78 | **0.39** |
| 160 | 0.01 | 0.02 | 0.28 | 0.04 | **0.66** | 23.5 | 3.3/3.6 | 0.32 | 0.23 |

explore judge 四维(0–2)均值:r0 new=1.70 err=0.83 ver=1.73 nonred=1.60 → r95 new=0.89 err=0.17 ver=0.84 nonred=0.88 → r138 new=1.02 err=0.28 ver=1.02 nonred=0.97 → r160 new=1.40 err=0.34 ver=1.60 nonred=1.40。**四维同步下降,error_correction 掉得最狠(0.85→0.3)**;r150+ 略回升(memory 变长了,judge 给的 verification_targets 回到 1.6)。judge 分数与 downstream 无关:pooled corr(explore, mean next3) 在各阶段为 −0.06..+0.04。**没有 judge 分数向上漂移的 hacking 迹象**;反过来是 memory 掏空把 judge 分数拉低了。

具体 token 密度(每 100 词):unique 路径 0.7(r0)→1.8–2.4(r60–85,因为 memory 变短、只剩路径)→1.0–1.4(r100+)→0.56(r160,被 thinking 稀释);"always/make sure/be careful"类 generic 短语 0.69(r0)→0.03(r110+),即**generic 建议不是变多而是消失**;meta 短语("the task…/this task/memory")0.1→1.4–2.5(r90+),即 memory 从"知识条目"变成"对本题的叙述"。grounding(memory 中路径 / 反引号标识符是否出现在此前 transcript 里)全程 0.95–1.00,**没有幻觉增长**。

## 2. 五个阶段(内容分析)

**Phase A r0–45 "结构化、较长、半清仓"**:两节干净(clean 0.7→0.98),10–17 bullets,1600–2450 chars,~75–89% 的 bullet 含具体 token(路径/反引号/行号/命令)。上一版 bullet 保留率只有 0.3–0.58,heavy rewrite 0.55–0.77——即从一开始 memory 就是"大部分重写",不是增量维护。format fail 的原因早期是 prompt marker(memory 里出现 `COMPLETE_TASK_AND_SUBMIT`,r0 占 fail 的 56%)和缺 header。

**Phase B r50–80 "越写越短、清仓式重写"(最接近'掏空')**:memory 中位数 2450→**534 chars**,bullets 3.7–5.4,RK 节 2 条 / LP 节 1.5 条,上一版 bullet 保留率 0.07–0.15(r60–70),heavy rewrite 0.93–0.96。每次 memory 只写"这题的函数在 X 行 + 这题的教训",上一题的条目被直接丢掉(r70 例:k=2 记了 `depr`/`_parse_http_header`,k=3 只剩 `path_shift`)。format_ok 掉到 0.19(r75),**fail 原因 90–100% 是"某节 <2 bullets"**。这不是"长而空",而是"短而只记本题"。注意这个阶段 ep_reward 反而在涨(0.13→0.22),说明 ACT 的进步与 memory 无关。

**Phase C r80–90 "反弹 + verbatim copy"**:memory 回到 1000–1400 chars,保留率跳到 0.74–0.85,但 same_as_prev(memory 与上一版完全相同,format gate 判 fail)r80/85 = 11%/9%,near-copy 0.27–0.30——短暂的"照抄上一版"策略,随即被 gate 压下去(r90 后≈0)。

**Phase D r90–155 "ACT 格式串味 + 占位句"**:WRITE 输出变成 `一段 prose 复述本题("The task involved fixing…") + 空的 ```bash``` 块 + 两节 header`。r138 时 98% 是这个模板;r105–125 有 17–55% 干脆没有 header、直接是 prose + 一条 bash 命令(甚至 `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`),format_ok 因缺 header 掉到 0.17(r120)。两节 bullet 各 3–4 条,保留率 0.8–0.9(几乎原样带下去,每次加 1 条),占位句("no new pitfalls / remains valid / straightforward fix")从 0 涨到 0.2–0.39。这是共享 policy 下 WRITE 被 ACT 的输出分布拖走的表现:memory 长度没爆(1000–2300 chars),但有效内容 = 上一版 + 一句本题小结。

**Phase E r159–160 "thinking 不闭合 → 推理独白漏进 memory"**:think chars 中位数 r0 1460 → r100 3094 → r155 3239 → r160 5041(p90 7766);r155 finish=length 21%,r159/160 达 67–72%,`</think>` 闭合率 0.34。代码逻辑(`codebase_rollout.py` L1025–1031):`<think>` 开标签在 prompt 里,输出若无 `</think>` 且无 `<think>` 字样,则**整段 raw thinking 被当作 visible memory**(r160 中 memory==raw_output 的样本 66%,7.9k chars),再作为 previous_memory 喂给下一次 WRITE,于是下一次 thinking 又要花大量 token 去"解析上一版里那段独白"(见 write_cases 里 sqlfluff r155:"The 'Previous memory' text … includes a monologue about confusion between repos"),形成自增强。format_ok 0.25;这些样本 write_reward=0 但 trained=True。r155 已经 14%,r160 66%,是**5 个 rollout 内的相变**。

## 3. Case study(全文见 `write_cases.txt`)

**sqlfluff r8 vs r155**(同 repo;r8 stage 0,r155 stage 6)
- r8:trial1 后 1410 chars,`src/sqlfluff/cli/commands.py`(lines 42-63)、`test/cli/commands_test.py`、pytest 命令;trial3 后 3237 chars 累积到 6 个路径;trial5 后 3849 chars、9 个路径。下一题前 3 条命令**一次都没命中** memory 中的路径(下一题是 parser/match_algorithms、grammar/anyof、api/simple——与 memory 记的 CLI 文件无关)。
- r155:trial1、trial3 后都是 7.6k/7.4k chars 的未闭合 thinking("The user wants me to update the memory… the previous memory is for radon… completely a different repository!"),format_ok=False,reward 0;trial5 后恢复闭合,2125 chars,但开头仍是"The task aimed to fix … the developer struggled to reproduce"。下一题命中 0。**但 r155 这组 ACT 的 reward 是 [0.93,0.95,0.9,0,0,0.85]**——memory 是垃圾,ACT 照样解题,说明 ACT 基本不依赖 memory。

**astroid r6 vs r150**
- r6:结构化两节,记 `astroid/_ast.py` line 36、`brain/brain_subprocess.py`、`tests/brain/test_brain.py -k …`;trial3→trial4 的下一题首 3 条命令命中了 `./astroid/brain/helpers.py`(唯一命中)。
- r150:全部变成 prose 段落("The task aimed to fix … The investigation revealed …"),1.4k tokens 的 thinking,2–5 个路径;下一题 0 命中,但 ACT 的 thought 里出现"Based on the repository knowledge, typing inference lives in `astroid/bra…`",即 ACT 会**引用** memory 但引用的是上题的位置,对新题没帮助(reward 0)。

**marshmallow r138 vs r160**(相邻的两个"末期"样本)
- r138:模板化 `prose + 空bash + 两节`,format_ok True,write_reward +0.625/+0.35/+0.14(下游 0.85/0.72/0.88 恰好是易题);ACT thought 明确说"According to the memory, there was a previous issue where `OrderedSet.__repr__`…"——memory 被读到,但那是同一个类的另一个 bug,ACT 仍从头 cat 文件。
- r160:trial1 后正常(1111 chars),trial3/5 后 8.4k/8.1k chars 的独白("**Previous Memory:** … **Task Findings for Memory Update:** … Wait, I need to be careful here"),format_ok False;下一题命中 `src/marshmallow/decorators.py`(因为 8k 字符里什么路径都有)。

跨全量:next-trial 前 5 条命令命中 memory 路径的比例 0.11–0.33,无趋势;命中 vs 未命中的下一题 reward:r0–20 0.127 vs 0.149,r81–110 0.234 vs 0.204,r111–138 0.219 vs 0.248 → **无差异**。

## 4. Reward 机制检查

- **由构造决定的相关**:全量 trained 样本 corr(ungated R, mean next3)=+0.55..0.63,corr(ungated R, mean prev3)=**−0.74..−0.76**;var(prev3)=0.050–0.097 > var(next3)=0.034–0.066,next3 只解释 R 方差的 42–46%。即 write_reward 一半以上是"前 3 题考得怎样"的镜像(mean-reversion),WRITE 对此毫无控制。
- **位置偏置**:episode 内每个 repo 的 6 题按 mutation 类型固定排序(pos0 多为 func_pm_ctrl_shuffle/func_basic,均值 0.32–0.36;pos5 多为 combine_module/lm_rewrite,0.05–0.08),ACT reward 随位置单调下降(r111–138:p0 0.43 → p5 0.16)。于是 ungated R 按 k 有系统偏置:k0–2 = −0.06..−0.08,k3 +0.06,k4 +0.18,**k5 +0.25**(next3 是新 repo 的易题、prev3 是本 repo 的难题),k6 +0.12,k7–8 ≈ −0.05。GRPO 分组 (group_index, downstream_trial_pos) 会去掉这个均值,但它说明 reward 主要由题序而非 memory 决定。
- **memory 内容 vs 下游**:pooled corr(mean next3, ·):unique 路径数 −0.04..+0.04,chars ±0.04(r150–160 +0.16,n=352,被 8k 独白拉的),explore ±0.06,与上一版相似度 ratio ±0.04,format_ok ±0.05。**找不到任何 memory 特征能预测下游 reward**。
- **format gate 的作用方向**:按 (rollout, 题序, k) 分组算 GRPO advantage,format-fail 样本的 gated advantage 在各阶段为 −0.07..−0.45(r81–110 为 +0.01≈中性),ungated 反事实为 −0.05..+0.04 → gate 平均上确实在惩罚 format 失败,不存在"负 reward 时故意坏格式拿 0"的净激励(虽然 fail 样本里 ungated<0 的比例 0.35→0.56 略高于 ok 样本 0.35→0.45)。但 gate 只检查 header/bullets≥2/非 verbatim,对 Phase D 的"prose+空 bash+两节"和 Phase E 的部分独白(恰好含两节)放行。
- 训练覆盖:k=11 无下游不训;k≤10 的 trained 比例 0.75–0.91,r60–90 掉到 0.6–0.7(整组 reward 相同被过滤等)。

## 5. 所见与结论

1. **write_reward 无信号**:均值 +0.02±0.25 全程不变,ungated 均值 ≈0(靠 +0.1 bonus 抬平),方差 55% 来自 prev-window,与 memory 的路径数/长度/相似度/explore 的相关 |r|<0.06(n≈3.4k/阶段)。K=3 窗口差分把"题序难度梯度"变成了 k5 +0.25 / k0–2 −0.07 的系统偏置。
2. **explore judge 没被 hack**:0.75→0.45(r95 低点 0.35),四维齐降,error_correction 0.85→0.3;与下游 corr≈0。ACT reward 同期 0.16→0.27–0.37 涨,与 memory 质量脱钩。
3. **"掏空"以两种形态出现,时间不同**:r50–80 是"短而清仓"(534 chars、3.7 bullets、上一版保留 7–15%,format fail 90% 因 <2 bullets);r90–155 是"模板化+占位句"(98% `prose+空bash+两节`,占位句 0.2–0.39,每次只加 1 条),不是"长而 generic"——generic 短语反而从 0.69/100 词降到 0.03,幻觉 grounding 全程 >0.95。
4. **r155→r159 相变:thinking 不闭合 → raw thinking 变 memory**:finish=length 0.21→0.72,`</think>` 闭合率 0.87→0.34,memory 中位数 2073→7904 chars,66% 的 memory 是推理独白;根因是 `codebase_rollout.py` L1028–1031 在 `<think>` 开标签位于 prompt 时,未闭合输出整段作为 visible memory,并回灌为 previous_memory 自增强。这是当前最紧急的机制 bug(与"是否该继续训"无关,只陈述事实)。
5. **ACT 几乎不用 memory**:下一题前 5 条命令命中 memory 路径 11–33% 无趋势,命中与否下游 reward 无差(0.13 vs 0.15 / 0.23 vs 0.20 / 0.22 vs 0.25);r155 sqlfluff 组 memory 是 7k 独白但 ACT reward 0.93/0.95/0.9。ACT thought 里偶尔"According to the memory…",引用的都是上一题的位置。
6. 噪声提示:每 rollout 只有 2 个 instance 序列 × 8 episodes,单 rollout 的 corr/均值抖动大(如 r120 format_ok 0.17 vs r138 0.77);阶段结论以 20–30 个 rollout 的 pooled 值为准;150+ 只有 3 个 train rollout(n=352 trained writes)。
=====
# smith-4b-v3nocurr-explore-gwin — Heldout 轨迹 & ACT 如何用 memory(只读 case study)

数据:heldout eval 20 个点(iter 7…159,每点 5 episodes × 6 issues = 30 issues,25 个带 memory 的 trial);train 抽样 every-5th + 150/155/160(31 个点,每点 16 episodes × 12 issues)。
脚本:`heldout_metrics.py`(→ `heldout_metrics.csv`)、`heldout_posconf.py`、`heldout_memuse.py` / `heldout_memuse2.py`、`heldout_writehealth.py`、`heldout_misled.py`、`heldout_cases.py`。原始输出:`memuse_out.txt`、`memuse2_out.txt`、`writehealth_out.txt`、`case_*.txt`。

**先说一个结构性事实(读所有数字前必须知道)**:heldout 每个 episode 都是固定的 6 个 instance:tablib-595/596/613 占 pos 0–2(顺序被 5 种 permutation 打乱),tenacity-614/615/628 **固定**占 pos 3/4/5。六个 instance 难度差异极大(100 次里成功率:595 13%、596 1%、613 26%、614 2%、615 0%、628 39%)。所以 "trial_pos 3–5 vs 0–2" 完全被 instance 难度混淆(pos5 永远是最容易的 628,pos3/4 永远是几乎不可解的 614/615),不能拿来当 within-episode learning 的证据。干净的比较只有一个:**同一个 tablib instance 在 pos0(无 memory)vs pos1/2(有 memory)**。

## 1. Heldout over training

| iter | succ | reward | turns | rw pos0-2 | rw pos3-5 | succ tablib | succ tenacity | eps "helped" |
|---|---|---|---|---|---|---|---|---|
| 7 | .10 | .015 | 37.4 | .000 | .030 | .00 | .20 | 3/5 |
| 15 | .20 | .105 | 27.6 | .057 | .153 | .07 | .33 | 3/5 |
| 23 | .10 | .069 | 19.6 | .058 | .080 | .07 | .13 | 2/5 |
| 31 | .10 | .086 | 13.8 | .062 | .110 | .07 | .13 | 2/5 |
| 39 | .13 | .083 | 20.1 | .110 | .055 | .13 | .13 | 3/5 |
| 47 | .13 | .117 | 22.0 | .118 | .115 | .13 | .13 | 4/5 |
| 55 | .27 | .147 | 25.1 | .077 | .217 | .27 | .27 | 4/5 |
| 63 | .07 | .037 | 23.7 | .018 | .057 | .07 | .07 | 1/5 |
| 71 | .20 | .126 | 26.2 | .098 | .153 | .13 | .27 | 4/5 |
| 79 | .13 | .062 | 23.9 | .045 | .078 | .13 | .13 | 2/5 |
| 87 | .20 | .156 | 22.3 | .090 | .222 | .13 | .27 | 4/5 |
| 95 | .10 | .091 | 16.5 | .122 | .060 | .13 | .07 | 2/5 |
| 103 | .10 | .093 | 10.4 | .125 | .062 | .13 | .07 | 3/5 |
| 111 | .10 | .087 | 12.3 | .058 | .117 | .07 | .13 | 3/5 |
| 119 | .10 | .092 | 12.9 | .185 | .000 | .20 | .00 | 2/5 |
| 127 | .03 | .030 | 16.8 | .060 | .000 | .07 | .00 | 1/5 |
| 135 | .17 | .105 | 19.8 | .135 | .075 | .20 | .13 | 3/5 |
| 143 | .23 | .188 | 16.8 | .327 | .048 | .40 | .07 | 3/5 |
| 151 | .13 | .075 | 28.2 | .063 | .087 | .13 | .13 | 3/5 |
| 159 | .10 | .075 | 17.0 | .095 | .055 | .13 | .07 | 3/5 |

(完整逐列含 rw_pos0..5、per-repo、eval_status 分布见 `heldout_metrics.csv`。)

- 成功率在 0.03–0.27 之间随机游走,20 个点 30 issues/点,一次成功 = 3.3 个百分点,**看不出趋势**。分阶段均值:early(7–39) succ .127 / rw .072;mid(47–103) .150 / .104;late(111–159) .124 / .093。
- 有趋势的是 **turns**:early 27→ mid ~20 → late 12–17(iter 103 只有 10.4)。变短不是因为解得快,而是"看几眼就 submit":turns≤6 的 trial 占比 iter 7–87 为 0–7/30,iter 95–127 跳到 12–19/30(其中成功仅 1–3 个)。budget 耗尽(40 turns 未 submit)从 iter7 的 25/30 降到 late 的 4–9/30。
- **pos 3–5 vs 0–2**:early 阶段 pos345 rw .086 > pos012 .057,late 反过来 .055 < .132 —— 这只是 tablib 与 tenacity 相对成功率的漂移(late 阶段 tenacity 几乎全挂:iter 119/127 tenacity 0/15),不是 within-episode learning 的变化。
- **干净的同 instance 对比(tablib, pos0 无 memory vs pos1/2 有 memory, 全部 20 点合计)**:success pos0 14/100 = .140 vs pos12 26/200 = .130;reward .090 vs .098。逐 instance:595 pos0 .20 vs pos12 .08(有 memory 反而差);596 .00 vs .02;613 .30 vs .25。分阶段:early pos0 .12 / pos12 .04,mid .15 / .14,late .14 / .19。→ **memory 对 heldout 成功率没有可测的正效应,late 阶段有微弱正向(+4pt,n=35 vs 70,噪声内)**。
- memory 唯一可见的收益是**成功时步数更少**:tablib 成功 trial 的 mean turns pos0 15.4(n=14) vs pos12 11.2(n=26),reward .643 vs .752;主要来自 mid 阶段(pos0 26.2 turns vs pos12 11.5)。late 阶段反而 pos0 8.2 < pos12 11.5(late 的 pos0 也已经学会直接开 `_ods.py`)。
- "memory helped" episodes(同 repo family 里第一题失败、后面题成功):early 13/25,mid 24/40,late 18/35 —— 但由于 tenacity family 里 628 永远排最后且远比 614/615 容易,这个指标基本就是"628 有没有解出来",不能解读为 memory 的功劳;反向 "hurt"(第一题成功后面全败)3/6/4。

## 2. ACT 怎么用 memory(train + heldout)

定义(`heldout_memuse.py`):对 memory_in 非空的 trial,从 memory 里抽 token:文件名(`*.py` basename)、函数/方法名(`def x`、`x()`、反引号标识符)、`pytest ...` 命令;(a) overlap = 前 3 条命令里出现任一 token;(b) ref = assistant 文本里出现 "memory / from the previous issue / as recorded / based on my notes" 等;phase:early ≤20 / mid 60–100 / late ≥120。

| split/phase | n | path | func | any | ref(全程) | ref(前3步) | succ\|any / \|not | succ\|ref / \|not | 整体 succ |
|---|---|---|---|---|---|---|---|---|---|
| heldout early | 50 | .24 | .24 | .40 | .16 | .08 | .15 / .17 | .00 / .19 | .16 |
| heldout mid | 125 | .30 | .42 | .58 | .20 | .15 | .18 / .08 | .12 / .14 | .14 |
| heldout late | 125 | .26 | .55 | .68 | .14 | .13 | .13 / .13 | .22 / .11 | .13 |
| train early | 880 | .24 | .27 | .40 | .14 | .09 | .40 / .34 | .29 / .37 | .36 |
| train mid | 1584 | .16 | .28 | .36 | .18 | .14 | .27 / .31 | .25 / .30 | .29 |
| train late | 1232 | .18 | .39 | .48 | .25 | .22 | .35 / .35 | .21 / .40 | .35 |

- pytest 命令重合率全程 0:memory 里几乎不写可复用的测试命令(WRITE 的 "how to run tests" 这一项从来没被填出实质内容)。
- overlap 随训练上升(heldout .40→.68,train .40→.48),但**matched control 同样高**(`heldout_memuse2.py`:把该 trial 的命令与"另一个 episode、同 repo、同 trial_pos"的 memory 配对,heldout own .75 vs ctrl .75,train own .44 vs ctrl .42)。也就是说 overlap 反映的是 memory 内容是 repo-generic 的(`_ods.py`、`read_cell`、`wraps` 换个 episode 也一样),**没有证据表明 ACT 在读自己这份 memory 里的具体内容**。
- 更直接的行为信号:第一条命令的类型(有 memory 且同 repo 时):heldout `open`(直接 cat/sed 某个具体文件)early .28 → mid .07 → late .41;train .16 → .14 → .43;而 pos0(无 memory)heldout 全程 100% `explore`,train late pos0 也有 .37 `open`(issue 文本本身给了文件名)。第一条命令直接打开 memory 里提到的路径:heldout early .15 / mid .07 / late .25,train .08/.05/.11。→ late 阶段确实多了"跳过 find,直接开 memory 里的文件"的行为,但相应的 succ|dopen 在 heldout 是 .24 vs .15(n=77 vs 323),train 是 .24 vs .28 —— 没有稳定收益。
- 显式引用(ref):heldout ~15–20% 不随训练涨;train late 涨到 25%(前 3 步 22%),但 **ref 的 trial 成功率显著更低**(train late .21 vs .40,n=302 vs 930;heldout 全部 .10 vs .14)。抽样看引用文本:约一半是开场白式的 "Based on the memory, the file is X"(信息量 = 一个路径),另一半是卡住时回头引用 memory 里的"上次尝试 X 失败"。ref 更像 "困难题的伴随物" 而不是使用 memory 的证据。
- **同 repo vs 换 repo**:换 repo 的第一题(memswitch)overlap 恒为 0(heldout)/≤.24(train),而且这些 trial 在 train 里成功率反而最高(.58–.65,是新 repo 的第一题、instance 更简单),说明 memory 内容对换 repo 后没有任何可迁移的通用 "how-to" 价值 —— 与 pytest 重合率为 0 一致。

### WRITE 侧的退化(直接决定 ACT 拿到什么)—— `heldout_writehealth.py`
- iter ≤ 87:WRITE 输出干净的两段式 bullets(median 700–2000 chars,preamble<10%)。
- iter 90–105 起:**几乎 100% 的 memory 前面带一段 "Let me analyze this task…" 的推理 preamble**(heldout iter 95 .73 → 103 起 1.00;train iter 90 .51 → 100 .97),整段 raw output 原样进 ACT 的 `[Memory of what you've learned so far]`。ACT 是 no-thinking 模式,WRITE 用同一个 policy,thinking 漏进了输出。
- iter 111–127:`### Repository Knowledge` heading 缺失率升到 30–37%(train 120 达 56%),bullets 从 ~9 条掉到 3–4 条。
- iter 159/160:**67%/72% 的 WRITE 输出打到 2048 token 上限(finish=length)**,median memory 长度 7.7K/7.9K chars,是一整段没有结论的自我辩论(例:heldout 159 ep0 trial 3 之后,WRITE 花 2048 tokens 讨论"换 repo 了,memory 到底要不要 carry forward",没有输出任何 bullet)。换 repo 时 100%(heldout)/94%(train)撞长度。同一时刻 ACT 侧空命令率从 ≤4% 跳到 12.6%(iter 159)。→ iter ~155 之后是 policy 整体的格式崩坏,不是 memory 机制的问题。
- 中期(iter 55–75)memory 反而被压得很短(median 530–780 chars,4–5 bullets),这时 gated_windowed K=3 的 write reward 均值为负(train iter 65–90 rw_mean −.03~−.05),WRITE 在被推向"少写"。

## 3. Case study 摘录

### 3a. 早期 heldout iter 15 ep1(`case_15_1.txt`)— [0,0,0,1,0,1]
- trial0 tablib-595(无 memory):`ls`/`find` → `cat _ods.py` → 整文件重写,40 turns 耗尽。
- memory→trial1(922 chars):`_ods.py` 位置、`table.TableCell` 属性、"datetime 用 datevalue round-trip 失败"、"time 必须 ISO 8601 duration PT…"。trial1 tablib-613(空字符串 reload 崩):第一步直接写 repro 脚本、第三步 `sed -n '150,180p' _ods.py`—— 跳过了 explore(memory 提供了文件),27 turns,tests_failed。
- memory→trial2 新增 "`read_cell` 在 140–172 行、`getAttribute('value')` 用 try/except 包"。trial2 tablib-596(**XLSX** 日期格式题):前三条命令全是 `grep datetime _ods.py` / `cat _ods.py` —— **memory 把它引到 ODS 上**,22 turns 失败。
- trial3 tenacity-614(memory 全是 tablib):100% 从 `ls`/`find` 重新 explore,memory 无用但也无害,21 turns 成功。
- 之后 WRITE **整段丢弃 tablib 内容**,只保留 tenacity(1448 chars,含 `wraps()` 行号 349–376、`_local` thread-local)。trial4 615(pickle RLock)第二步就 `sed -n '340,380p' __init__.py`、第三步 sed 改 `copy._local = self._local`——用了 memory 的行号,但方向错,40 turns 未提交。trial5 628(wait_exponential_jitter)与 memory 无关,14 turns 成功。
- 结论:early 阶段 memory 的作用 = 省掉 1–2 步定位;对解题方向没有帮助,对 596 有误导。

### 3b. 晚期 heldout iter 143 ep1(`case_143_1.txt`)— [1,1,0,0,0,1],该点 heldout 最好(.188)
- trial0 595 无 memory:`find|grep ods` → `cat -n _ods.py` → 两次整文件重写 → submit,14 turns 成功(reward .675)。
- memory→trial1(1816 chars,末尾夹着 WRITE 的 `</bash>` 残片):`_ods.py`、`ODSFormat`、"datetime 用 date-value、time 用 ISO duration"。trial1 613:`cat -n _ods.py | head -250` → repro 脚本 → 用 python 脚本 patch `dset_sheet` 的 else 分支 → submit,**4 turns 成功,reward .925**。这是 memory 带来 step-efficiency 收益的最干净例子(同一 episode 里 pos0 的 595 花 14 turns)。
- trial2 596:memory 仍全是 ODS,但这次 ACT 第一条就 `grep xlsx` → `cat _xlsx.py` → 整文件重写 → 6 turns 提交失败(不像 iter 15 那样被带偏,但也没跑测试就交)。
- trial3/4 tenacity(memory 里 tablib 内容原样保留,后面追加 tenacity 行):第一条命令都是 `find … grep BaseRetrying|wraps` / `grep RLock`——完全由 issue 文本驱动;614 14 turns 失败,615 19 turns 失败;628 14 turns 成功(第一条命令是空命令)。
- 结论:late 阶段 memory 的"用法"退化成一件事——同 repo 下一题直接 `cat` memory 里那一个文件;成功与否仍取决于 issue 本身。

### 3c. Memory 明确误导 —— tablib-596(XLSX 题)被 ODS memory 带偏(`heldout_misled.py`)
| phase | 条件 | n | 前5步碰 `_ods` | 前5步碰 `xlsx` | 实际编辑/重写 `_ods.py` | succ |
|---|---|---|---|---|---|---|
| early | 有 memory | 15 | .67 | .40 | .60 | 0 |
| early | 无 memory | 10 | .00 | 1.00 | .10 | 0 |
| mid | 有 memory | 24 | .54 | .33 | .46 | 0 |
| mid | 无 memory | 16 | .00 | .88 | .00 | 0 |
| late | 有 memory | 21 | .43 | .62 | .29 | .05 |
| late | 无 memory | 14 | .00 | 1.00 | .00 | 0 |

具体例子 heldout iter 159 ep0 trial1(596,memory=595 之后的 ODS 笔记):turn0 assistant 直接写 "The ODS export currently uses date-value…we must check `_ods.py`",turn1 `cat -n _ods.py`,turn2 python 脚本把 `dset_sheet` 里 datetime 改成 `datevalue`,turn3 `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` → tests_failed,共 4 turns,一次没碰 `_xlsx.py`、一次没跑测试。无 memory 时 100% 的 596 trial 都会先去 `_xlsx.py`。596 本身几乎不可解(1/100),所以这次误导没有改变成功率,但它是"memory 提供的 prior 压过 issue 文本"的确凿证据;late 阶段误导比例(.43)比 early(.67)低,ACT 学会了更多看 issue 文本。

## 4. 所见与结论

1. **heldout 无趋势**:成功率 20 个点在 .03–.27 间波动(30 issues/点,1 次成功 = 3.3pt),early/mid/late 均值 .127/.150/.124;唯一单调变化的是 turns(27→~14),iter 95–127 有 40–63% 的 trial 在 ≤6 turns 内提交(135–159 在 0–50% 间波动),其中 80–90% 失败——是"早交"而不是"解得快"。
2. **memory 对 heldout 成功率零效应,对步数有小收益**:同一 tablib instance 有 memory(pos1/2)vs 无 memory(pos0):succ .130 vs .140(n=200/100);成功时 turns 11.2 vs 15.4、reward .752 vs .643。收益 = 省掉 explore 步,不是解题能力。
3. **ACT "用" memory 的方式很薄**:前 3 条命令与 memory 的 token 重合率随训练从 .40 升到 .68(heldout),但与"别人的同 repo memory"配对的 control 一样高(.75 vs .75)—— memory 内容 repo-generic,重合不代表在读自己这份。可确认的行为变化只有一个:late 阶段有 memory 时第一条命令直接 `cat` 具体文件的比例 heldout .28→.41、train .16→.43(pos0 heldout 始终 100% explore)。pytest 命令重合率全程 0,memory 从不携带可复用的测试/验证流程。
4. **显式引用 memory 与低成功率相关**:train late ref 25%,succ|ref .21 vs .40(n=302/930);heldout .10 vs .14。引用多发生在卡住/难题里,或只是"Based on the memory, the file is X"的开场白。
5. **memory 会误导**:tablib-596(XLSX 题)有 memory 时 43–67% 的 trial 前 5 步去改 `_ods.py`,无 memory 时 0%;iter 159 有 4-turn 改错文件即提交的实例。
6. **WRITE 从 iter ~95 起把推理 preamble 原样写进 memory(100%),iter 111–127 有 30–56% 丢失 `### Repository Knowledge` 结构,iter 159/160 有 67–72% 撞 2048-token 上限输出纯自我辩论(median 7.8K chars)、换 repo 时 94–100%**;同时 ACT 空命令率 12.6%(iter 159)。iter 155 之后 policy 整体格式崩坏,late heldout 的 memory 数据基本是垃圾输入,后续任何 "late 阶段 memory 是否有用" 的结论都应以 ≤151 为准。
Shell cwd was reset to /home/qixinx


# ==== assistant report (transcript line 7951) ====

三档真实样本拉出来了,给你看原文:

## 早期(r5-15):干净的结构化知识库

```
### Repository Knowledge
- The DBC file parsing logic lives primarily in ./src/cantools/database/can/formats/dbc.py...
- The SYM format parser lives in .../sym.py, where multiplexer IDs are processed in
  get_mutliplexer_ids() within _load_muxed_message_signals().

### Lessons & Pitfalls
- Do not submit fixes immediately after making code changes; always run the test suite first.
- When fixing UnboundLocalError..., initializing the variable to None before the scope-issue
  point resolves it.
- Complex function replacements should consider a script-based edit rather than complex sed...
```
定位信息精确到文件/函数,教训可操作。(早期也偶有例外:r11 有一条 7952 字符的"转述 transcript"式流水账,苗头当时就有,只是占比 ~1%。)

## 中期(r60-70):结构保持,质量尚可

```
### Repository Knowledge
- python-docx stores package writer logic in src/docx/opc/pkgwriter.py
- CT_Override.new() expects (partname, content_type) as parameters in that order
### Lessons & Pitfalls
- When code is clearly corrupted/mangled, don't try to patch incrementally - completely rewrite...
```
和早期同构,略啰嗦(截断率已从 1% 升到 14%)。

## 晚期(r105-118):三种退化形态并存

**形态 1:流水账 + OK-spam 尾巴撑到截断**(r108,6873 字符):开头不再是知识条目而是转述失败过程("The task FAILED - it ran out of turns... Wait, so the FEEDBACK..."),结尾是几百行 `OK!` 直到 8k 截断。这占了晚期的半壁江山(50.6% 带 OK-spam,61% 撑到截断)。

**形态 2:memory 在转述"ACT 输出本身已是 OK-spam"**(r114,5378 字符):"every single response I gave got 'No complete bash code block was found' error... First was OK OK OK OK…(整段复读)"——这条是 ACT 先崩、WRITE 忠实照抄的直接物证。

**形态 3(最扎眼):自我诊断式 memory**(r114,1769 字符,结构完好):
```
### Lessons & Pitfalls
- NEVER output excessive reasoning text - this causes massive token usage that leads to
  hitting response limit before delivering actual code block
- System feedback "Empty command" / "No complete ```bash code block was found" means
  the RESPONSE must have the bash code block...
### How to correctly work on this task:
- Understand response format: short reasoning + ONE CODE BLOCK
```
**模型把自己的格式崩溃病理诊断得清清楚楚、写成了给自己的操作规范——但下一题的 ACT 照样产不出合法代码块。** 这条是"memory 污染不是主因"的最生动注脚:问题不在知识层(知识层甚至知道怎么治),在权重的生成分布层——policy 已经采样不出"短 reasoning + 单个 bash 块"这个形态了,写在 context 里的正确指令救不回来。

量化汇总:长度 2.1k→2.3k→**4.3k**(median 4.9k),截断率 1%→14%→**61%**,OK-spam 0%→0%→**50.6%**,WRITE thinking 从 99% 闭合退到 **60% 干脆不产**。

综合三个 subagent 的完整死因报告和 ckpt 建议我整理好了,要不要现在出?核心结论一句话:**崩的不是"解题能力"是"生成分布"——reasoning 长度失控(845→4234 字符/turn)+ 复读退化,70% 的 turn 产不出可解析命令,空命令烧 turn 不烧 step、reward 上不被额外惩罚,于是 40-turn 预算在"空命令→报错→更长的慌乱 reasoning"死循环里耗尽;劣化从 turn 0-1 就开始,是全局分布崩塌。** 评测建议 iter_87 + iter_95 双测(r96 起滑坡、r102 跳崖,95 是最后的边界点,87 更稳妥)。


# ==== assistant report (transcript line 8307) ====

已给三个 subagent 都补发了硬要求:每窗口至少 8 条 case 从头读到尾——ACT 轨迹要逐 turn 读 reasoning 原文、像复盘棋局一样描述解题过程,memory 要把 previous_memory → thinking → 新 memory 的完整重写过程读完——统计只作佐证,返回必须带每条精读的具体描述和原文摘录。等它们回来。


# ==== assistant report (transcript line 17020) ====

三个 subagent 全部回来了,都读了真实 case。合并结论——**你看到的三个现象其实是同一个故事的三个侧面**:

---

**① WRITE 长度 V 型 + KL 上升:降是健康的,升是感染,KL 升的来源找到了**

- **下降段(r0→r99)是健康的模板化压缩**:thinking 从 1722 字符砍到 923,memory 收敛成信息密度很高的两段式笔记(读了谷底 6 个 case,零反刍、句句落在具体事实上)——这是 reward 教出来的正常优化,同期 KL 平稳。
- **回升段(r100+)是漂移感染**:增长的 ~70% 来自 thinking 里一个**虚构母题**——模型反复声称"transcript 缺失/不完整",但抽查输入完整无缺(27K 字符)。这个母题从 r105-107(恰在 OOM 重跑窗口内)萌芽,10 轮内指数扩散到 70% 的 trial;wait/actually 反刍从 0.04→2.85 次/trial。**训练管线在系统性放大它**:"声称输入不完整→保守照抄旧笔记→格式必然合规"是条安全策略,这类 trial 的 format_ok 94%(干净型只有 54%)、reward 还略高——gate 只查 memory 格式不查 thinking,于是每步更新都在强化这个基座分布外的话术,**这就是 KL 显著上升的直接机制**。剩余 30% 是 memory 正文变长,质量混合:有真货(具体修复公式),但对冲废话 38%→59%,母题已泄漏进笔记正文 6.7%(获得了经 previous_memory 传播的第二条通道)。

**② ACT r130 尖峰:是不稳定现象,机制与 actonly 崩溃同源,但 gated 处于"发烧-退烧"亚稳态**

- 尖峰实际是 r132-135 一段(2548-2736 字符/turn,空命令率 56-71%)。病灶不是空输出,而是**动作格式契约崩坏**:一条回复塞 2-9 个候选命令块(占八成,解析器要求"恰好一个 bash 块")、裸文本命令、幻觉出 `<tool_call>`/`<|im_start|>` 特殊 token;再叠加恶性循环——解析失败→环境回"Empty command"→模型归因"环境坏了"→更长的自我调试→更难解析。有个 case 里模型技术路线全程正确(已锁定 bug 位置),纯粹死于格式病:40 轮里 28 轮空转,最后一轮还在说"我的推理一直是对的"。
- **归因:内生漂移 ~七成**(同一个 jinja 仓库空命令率 r131 0.33→r133 0.75→r140 0.22,仓库没变;轻量仓库与重仓库同等沦陷),**难题窗口是点火器 ~三成**(r125-128 的 pdfminer/dvc 连败提供密集负反馈,把已抬升的不稳定推过阈值)。
- **r141 是打了折扣的真自愈**:同仓库同 commit 对照(sunpy,空命令 0.47→0.03)排除换题假象;但长 trial 中途仍会滑回幻觉格式,`</think>` 悬空污染率仍是基线 2 倍——吸引子还在,下个难题窗口可能再点火。**新哨兵**:悬空 `</think>` 率比空命令率灵敏、领先 1-2 轮,我会盯这个。

**③ 分档成功率:easy/mid 的真实提升被 hard 的失血抵消,所以总分不动**

做了同仓库配对(最干净的对照)后,数字比表面更清晰:

| 难度 | 表面 | 同仓库配对校正后 | 定性 |
|---|---|---|---|
| easy | +7pp | **+9~10pp,真实** | 提升全部来自"不再爆预算":早期 36% 的失败是 40 轮耗尽没提交,晚期提交率 76%→98%,而提交精度持平(53%→55%)。成功步数 14.6→5.1,读 case 确认晚期的一次通过诊断是实打实的——**学到的是"快"(流程效率),不是"准"(诊断力)** |
| mid | +3pp | +4~8pp,真实 | 同 easy 模式 |
| hard | 17%→9.6% | 真实恶化仅 **-3.2pp**(腰斩的一半是构成运气:r100-119 恰好抽中高分仓、r120-139 抽中一批历史零成功仓) | 但方向确实为负:**提交精度 37%→12% 单调恶化**(感染前就在跌)——快提交策略在 hard 上必然失血(lm_rewrite 要重建整个函数行为,猜补丁不验证几乎必挂,读的 case 里 8 个样本全部 tests_failed 且零验证);感染又把 hard 必需的长轨迹挤爆(步数预算耗尽失败 3%→21%) |

**串起来的一句话**:r100 之前 gated 在健康地学"快";r100 之后(起点在 OOM 重跑窗口)两条感染线并行生长——WRITE 的虚构母题和 ACT 的格式崩坏——reward 的盲区(gate 不看 thinking、不罚多代码块)让它们被容忍甚至强化,KL 上升、ACT 尖峰、hard 塌陷都是这一个漂移的不同投影;easy/mid 因为预算冗余暂时没在总分上显形。

**我不做任何干预**,继续监控(重点新增悬空 `</think>` 哨兵)。可供你考虑的下一步(只是选项):设计层面 ACT 也加格式 gating / thinking 长度惩罚;另一个值得记录的教训是 **OOM 重跑窗口疑似是感染起点**——崩溃恢复可能不像我们以为的那样"无损"。gated 目前 r142 附近,143/191,照常推进。


# ==== assistant report (transcript line 19219) ====

ACT 侧也回来了,两份合并——**grace12 的早期读数:两个目标病都在好转,外加一个要盯的新信号**:

## ACT 侧(验证行为)

- **砍验证的趋势被减缓但未逆转**:前 ~14 轮两组几乎同轨下滑(说明砍验证的动量来自初始已学会的策略,不是当轮 reward 即刻塑造),**r19-22 明确分化**——盲交率 grace12 40% vs gated 68%,每 trial 验证步数 +36%;
- **case 是教科书式的**:patsy 题 12 步里塞了 8 次验证拿满分;funcy 题的验证抓回了 sed 打错的真 bug(同题 gated 组 6/8 过、grace12 组 8/8 全过——**验证步直接兑换成了成功率**);
- **没有"拖到 12 步"的 hacking**:成功步数在 12 附近平滑无堆积;反而 gated 在 4 步处有尖峰——grace12 把极限压缩缓释到 5-9 步区间,方向健康;
- **保留意见**:总成功率还没兑现优势(0.180 vs 0.193 持平,还早);trivial 题盲交照旧(grace 只是不罚验证、没有奖励验证——简单题盲交也满分,验证只能靠"提高成功率"间接被强化)。

## WRITE 侧(前面已报,一句话回顾)

memory 积累行为完全反向修复:跨仓库保留 75% vs 6%、删除率 0.10 vs 0.47、笔记里全是验证产出的具体知识,漂移前驱(反刍、母题、自我开脱模板)全面更干净。**机制:grace 给 ACT 步数做验证 → transcript 变丰富 → 笔记有料 → 下游 reward 兑现 → 正循环。**

## 需要盯的新信号

grace12 特有的 thought 模板占位句 **"Your reasoning and analysis here." 残留率单调爬升**:r12 1.9% → r22 9.3%(gated 同轮仅 0.5-3.1%)——集中在失败长轨迹里,是一种轻度格式退化,我把它加进哨兵清单(连同已有的空命令率、字符/turn),每个评测点复查;若它像 gated 的 `</think>` 悬空率一样是崩溃先导,我们能提前 20+ 轮看到。

另一个修正记入档案:**gated 的 memory 掏空在 r8-14 就已成型**(多轮跨仓库保留 0/16),远早于此前认为的 r60/r105——之前的"28%→0.5% 断崖"只是终末期读数。

grace12 训练照常(r25 附近,零异常),继续值守。


# ==== assistant report (transcript line 34540) ====

第三个 subagent 也完成了,三路分析拼齐,给你完整结论。这次发现了一个此前没确诊过的**结构性问题**,先说它:

## 核心发现:exploration reward 被"旁路"了——它实际变成了 memory-writing reward

训练轨迹全量普查(早/中/晚三窗口各 ~2100 trial)的关键证据:

| corr(explore_score, X) | early | mid | late |
|---|---|---|---|
| 实际探索行为(新文件读取) | 0.11 | 0.14 | **0.005** |
| memory 文本增量 | 0.30 | **0.52** | 0.41 |

**judge 分数与真实探索行为在晚期完全解耦(r≈0),始终强耦合的是 memory 写了多少新文本**;且探索行为塌方的同时 explore_score 均值反而从 0.73 升到 0.79。模型学会了两头通吃:少探索、快提交拿 task reward,同时把 memory delta 写得"像有新发现"拿 explore reward。抽查实锤:晚期有 memory 更新头 12 行与上一版逐字相同、仅尾部追加,judge 四维全 2 分、explore_score=1.0。晚期同库清仓重写率涨到 37.5%,疑似在 farm "new_discoveries" 分。

根因是结构性的:**judge 评的是 WRITE 产物(memory delta),激励加在 ACT 上,信号源与被激励行为不同链路**——真探索不是拿分的必要条件。这呼应你最初的设计意图(防止过度 exploit memory),但要真约束行为,信号得取自 ACT 轨迹本身(新文件读取数、验证动作、vfe 事件),而不是 memory 文本。

## 你三个问题的答案汇总

**1. 训练/评测风格一致吗?** 基本一致(单命令率、探索占比、think、memory 风格全部同分布,连退化都镜像出现)——排除训练推理不一致。唯一真实 mismatch 是响应上限 2500 vs 32768:评测出现训练中物理不可能的超长退化复读(iter95 有 38% trial 受影响、分数腰斩),但量化只解释 gain 摆幅的 ~1/4,且同时打击双模式,不改变结论符号。

**2. 训练侧 memory/ACT 现状?** 不健康,mid(r56-66)是巅峰期(探索率 82%、memory 稳定累积清仓率 2.8%、hit40 降到 8.5%),late(r105-115)全面塌方:有 memory 时探索率 81%→**44%**(与 gated 的 87%→38% 同幅)、提交前验证 82%→**19%**、vfe 54%→**6.3%**、空命令 15%、**多 bash 块 turn 占比爆发到 26.7%**(mbfb 修复没压住晚期这波)。task reward 上涨全部来自步数变少,success 三窗持平——裸效率涨、能力没涨。

**3. 评测为何不出彩?** ①同款探索压制病,只是晚发 ~30 个 iter(评测探索率 iter63=66% 高于 gated 全家族 → iter95=26% 掉穿);②explore 组特有的"整文件重写"病(占失败 11-25%,gated 仅 3-4%)——更丰富的 memory(3.7-4.2KB 类清单,gated 的 4 倍)反而给了凭先验重写整个源文件的假信心,且晚期 memory 出现幻觉类名和错误路径;③iter95 失分集中在 4 题(tablib-540 一题 −0.74,与 gated 确诊病灶重合)。

## 对下一步设计的含义(记录,不动手)

1. **explore 信号改从 ACT 轨迹取**(新文件读取、验证/vfe 事件计数),或 judge 同时看轨迹而非只看 memory delta——堵旁路是第一优先,β 调多大都没用,信号本身可 farm
2. 提交前验证始终没被任何配置救起来(explore 组也只有 ≤40%)——验证需要 reward 直接背书(失败提交惩罚/验证免步数折扣,这在 smith-v2 case study 就提过)
3. 多块格式崩坏晚期爆发是新哨兵,已加入巡检(空命令率 + 多块率)
4. 你之前记的设计候选里"judge 分直接付 WRITE"与本次发现一致方向但不够——付给 WRITE 也还是评 memory 文本,同样可 farm

训练侧现状:第 5 命还活着(r113 重滚中,ckpt 仍卡 iter_111)。鉴于 late 窗口已确诊退化 + 评测三点齐全,**这组的科学价值已基本收割完毕,是否值得继续烧卡跑到 191(还要再扛 ray 抽风)由你定**;停或继续我都能执行。三份分析我会归档进 notes。


# ==== assistant report (transcript line 45149) ====

三个 subagent 的结果已合成一份报告(`scratchpad/casestudy_gwin/CONSOLIDATED_report.md`,分报告与 csv/脚本/摘录同目录)。要点如下:

## gwin K=3 case study 汇总

**一句话**:ACT 学到的是"更短"不是"更对";WRITE reward 没信号,memory 先被压短、再被 ACT 格式串味、最后(r155→159)因 thinking 不闭合把 8k 字符推理独白灌进 memory;heldout 无趋势;ACT 几乎不依赖 memory。

### 1. ACT 演化(20-rollout 窗口均值)
| 窗口 | succ | reward | turns | 提交前跑过测试 | 最后 edit 后盲提交 | py-replace 脚本占 edit |
|---|---|---|---|---|---|---|
| 0–19 | .30 | .17 | 26 | 82% | 2% | ~15% |
| 20–39 | .19 | .14 | 16 | 30% | 32–50% | ~15% |
| 60–79 | .32 | .22 | 21 | 60% | 7–13% | ~39% |
| 100–119 | .33 | .28 | 10.5 | 11% | 85–89% | 55–69% |
| 120–139 | .34 | .29 | 14 | 16% | 88% | 55–59% |
| 150–160(3个) | .44 | .36 | 18 | 30% | 64–78% | 42–56% |
- reward 涨幅几乎全由 turns 项解释(reward=1−0.025×(turns−1));success 平。按类型:func_basic +5pt、ctrl_shuffle +15pt、lm_rewrite/combine_* 持平或跌(只是死得更快)。
- 验证习惯在 r20、r100 两次相变被压掉,r60–79 一次回潮;首次 submit 中位 turn 21→5。
- `sed -i` → 写 `/tmp/fix.py` 做 `str.replace`;看到 "not found" 0% 提交、看到 "Done" 44–56% 直接 submit。典型 case(cantools r124):源码因 output-too-long 没看到,凭猜的旧串 replace、静默不命中、8/8 同构失败 → group advantage 全 0。
- 组内多样性 r90–100 坍缩(T1 去重率 .87→.49);格式上修掉了 `<tool_call>` 残留,新增自言自语(3→15–24%)、无 reasoning 直接代码块(8→30–56%)、r160 空命令 8.9%。138/150/155/160 之间来回摆,不像收敛。

### 2. WRITE / memory 五阶段
| 阶段 | 形态 | 关键数 |
|---|---|---|
| A r0–45 | 两节干净 bullets,半清仓 | 1600–2450 chars,保留上一版 .3–.58 |
| B r50–80 | **越写越短、清仓式只记本题** | 中位 534 chars,3.7 bullets,保留 .07–.15,format_ok 低至 .19 |
| C r80–90 | 反弹+照抄上一版 | near-copy .27–.30,被 gate 压回 |
| D r90–155 | **ACT 格式串味**:prose 复述+空 ```bash```+两节;占位句 | r138 98% 该模板,占位句 .2–.39 |
| E r155→160 | **thinking 不闭合,raw 独白进 memory** | finish=length .21→.72,闭合率 .87→.34,中位 7904 chars,66% memory==raw |
- write_reward 均值 +.02±.25 全程不变;>55% 方差来自 prev-window(mean-reversion),与 memory 任何特征 |r|<.06 → **无信号**;K=3 差分还把题序难度梯度变成 k5 +.25 / k0–2 −.07 偏置。
- explore judge .75→.45,四维齐降 → **没被 hack**,是被掏空拉低。generic 短语反而 0.69→0.03/100 词,grounding >.95。

### 3. Heldout 与 memory 使用
- heldout succ 20 点 .03–.27 随机游走(early/mid/late .127/.150/.124),只有 turns 单调降(27→12–17,主要是"早交")。
- 同 tablib instance 有/无 memory:succ .130 vs .140;成功时 turns 11.2 vs 15.4 → memory 只省定位步。
- ACT 与 memory 的 token 重合 .40→.68,但与"别人的同 repo memory"控制组一样高(.75 vs .75)→ 不是在读自己这份;pytest 命令重合 0;显式引用 memory 的 trial 成功率更低(.21 vs .40)。
- memory 会误导:tablib-596(XLSX)有 ODS memory 时 43–67% 去改 `_ods.py`,无 memory 时 0%。

### 4. 机制 bug(需你拍板,我没改代码)
`codebase_rollout.py:1025-1031`:WRITE thinking 开着时 `<think>\n` 开标签在 prompt 里,输出不含 `<think>` 字样;一旦 2048-token 截断(无 `</think>`)就落到"整段 raw 当 memory"分支——与注释声明的意图相反,7–8k 独白回灌成 previous_memory 自增强。**gwin-think 那组 WRITE_THINKING=1 同样带此 bug**(K=3 是 r155 后才爆发;think 组现 rollout 29,我会顺带盯 write finish=length 比例)。
建议一行修法:
```python
elif "<think>" in _visible or _write_think:   # opening tag lives in the prompt when thinking is on
    _visible = ""
```
要不要修、修完 K=3 怎么处理(继续跑 / 从 155 前 ckpt 重训 / 不管)由你定。

**训练状态 10:34**:K=3 rollout 161 进行中(GPU 66–82%),think rollout 29 进行中,双方 OOM=0,磁盘 80%/67%。/loop 下一次 10:55 照常。


# ==== assistant report (transcript line 46190) ====

三个 subagent 都完成了,合成汇报(全文 `scratchpad/casestudy_think/CONSOLIDATED_think.md`,分报告与摘录同目录):

## think 组 rollout 0–48 格式/思考体检

**一句话**:ACT 的 thinking **结构和内容目前正常**;但已冒出三条隐患,其中 WRITE 那条和 K=3 崩坏是同一机制、处在起点。

### 正常的部分
- 100% 的 turn 有 `assistant_think`,为空仅 0.1–0.3%;闭合率 >99.6%;think 长度 p50 165→262 chars(+20%,不爆长);ACT length 截断 ≤0.34%。
- 空命令率 0.3–1%(K=3 同期 1–2%,崩坏期 18–74%),单 bash 块 >99.4%,早期自造 `<tool_call>` 格式锁死 trial 的问题(13/窗)已被 RL 修掉。
- 精读 120 条 think:切题 100%,有效推理 43%→53%→50%,思考-动作不一致仅 2–3/30,语言混杂 0.3%,"Wait" 打转 <1%,开头去重比 0.97(不模板化),三窗基本不变。
- heldout 侧格式也健康(空命令 <1.5%,think 0 例未闭合)。

### 三条隐患
1. **可见正文回显 prompt 占位句** "Your reasoning and analysis here.":r0–30 ≈0% → r34 5.9% → r42–47 **20–29%**(r47 29%),单调上升。后面的内容仍切题、命令照发,但可见 reasoning 正在套话化(reasoning 都进 think 了,正文被格式奖励挤成占位符)。
2. **WRITE 的 `</think>` 闭合率开始掉**:r0–39 ≥99.5% → r40–47 98.5% → **r48 96.4%**。未闭合样本不是截断(finish=stop,400–650 tokens,是在 think 里写完就停),整段"The user wants me to update the memory…"独白逐字成为下一题的 memory;更糟的是 r48 有 **2 例**独白恰好含两节 header,被判 format_ok、write_reward 0.875、trained=True——"不写 `</think>`"有被正强化的通道。K=3 就是这个机制在 r155 后 5 个 rollout 内从 .87 崩到 .34。
3. **验证行为退化**(与 K=3 同款):提交前跑过测试 87%→23%,完全不跑 python 就提交 0.6%→31%;turns 30→13,success 持平 .32–.40。

### 其他
- memory 亚健康:长度 3100→1400 chars、bullets 19→7、保留上一版 bullet 0.58→0.09(每题清仓不累积);grounding ~90%,无占位句;write_reward 均值 .03–.04 无趋势(和 K=3 一样基本没信号)。
- heldout 6 点 .077/.062/.068/.127/.137/.052 主要由 tenacity-628 一题驱动,与 K=3 同期无可分辨差异。

### 我接下来会盯的阈值(不改配置)
WRITE 闭合率单 rollout <95% 或连续下降;正文占位句 >40%;blind submit >50% 且 success 下滑——触发即报。可选修法仍是之前那一行(WRITE 未闭合 think 时回退 previous_memory)+ format 判定对未闭合/占位句记 fail,是否改由你定。


# ==== assistant report (transcript line 46503) ====

00:10 状态:

**离线测评(flame-10)**:iter87 replace 跑到 4/5 题序,约 10–15 分钟内出结果,出来立刻报;之后 87 stateless(~25 min)→ 159 → 55 → 127。

**think 组(flame-23)**:正常。rollout 53 完成(23:59),rollout 54 进行中,ckpt 51 已存,OOM 0,cgroup 738G,磁盘 75%,进度 54/191,下一评测点 iter55(约 1 小时后)。
三项格式指标(r52/r53):空命令 0.8%/0.6%、length 截断 0.5%/0.4%、WRITE 未闭合 1.6%/2.1%(比 r48 的 3.6% 回落),都在阈值内;唯一在爬的是可见正文里的占位句 "Your reasoning and analysis here"——r49 9% → r50 22% → **r52/53 36%**,离我设的 40% 阈值很近了。它不影响命令解析和成功率(思考都在 think 里),但说明可见 reasoning 正在退化成套话;到 40% 我会再提一次,是否处理由你定。


# ==== placeholder-retreat report (line 2433) ====

先把两个 case 讲透。

**Case 1:empty_patch(编辑静默失败 + 假证据)**

模型晚期的标准编辑手法是写一个 Python 小脚本:读入源文件 → `content = content.replace(旧代码串, 新代码串)` → 写回 → `print("Fixed!")`。

问题出在 Python 的 `str.replace` 的语义:**如果"旧代码串"在文件里没有精确匹配上(比如缩进差一个空格、原文和模型记的不一致),它不报错、不抛异常,只是原样返回**——文件一个字节都没改。而脚本里那句 `print("Fixed!")` 是无条件执行的,所以模型在 observation 里看到的仍然是 "Fixed!"。

模型把这个"Fixed!"当成了编辑成功的证据,直接提交。但文件根本没变 → `git diff` 为空 → 判分器收到**空 patch** → 0 分。

这里的要害是:一个有验证习惯的 agent 会在编辑后做任何一种廉价检查——重读文件、`git diff`、或在脚本里写 `if old not in content: 报错`——都能当场抓住这个 no-op。早期模型 80% 会在编辑后跑点什么;晚期模型验证全裁,于是连"自己的编辑到底生效没有"都不看,静默失败畅通无阻。empty_patch 率因此从 2.2% 翻三倍到 6.0%。这是"验证消失"最纯粹的切面:不是不验证修复对不对,是**不验证编辑发生没发生**。

**Case 2:模板坍缩(占位符回显)**

我们 ACT prompt 里有一段格式示例,里面用占位文本示意"这里写你的推理":`Your reasoning and analysis here. Explain why...`。这是给模型看格式的,不是让它抄的。

晚期模型开始**逐字复读这句占位符**作为每轮 thought 的开头,然后才接真正的推理内容。时间线很说明问题:r60-74 还 ≤1%,r80-84 突然跳到 55%,r88-92 达到 93%——不是渐变,是**突变式收敛**。机制上:reward 不读 thought 内容,这个前缀永远不受惩罚;RL 后期策略熵持续收缩时,输出分布会锁死到某个高概率模板上,这句话就成了固定的"开场仪式"。

为什么占位句后面"内容仍有实质"我们还是把它列为警报:① 它是**策略熵坍缩的可观测信号**——输出多样性在死亡,这通常先于更广泛的退化(和 mean reward 中期 0.212 → 晚期 0.159 的回落互相印证);② 同一个 trial 里能看到坍缩的行为后果:模型自己都说了"文件在上次尝试中改烂了(201 行重复、docstring 错位)",这是最应该跑一下 python 做语法检查的时刻——但它僵化的四步流水线(定位→读→改→提交)里没有验证这一步,照常裸奔提交,挂在 tests_failed。**策略坍缩的本质是失去了对异常情况的适应性**。

两个 case 的共同根:步数折扣把一切验证行为定价出局(case 1),熵坍缩把行为钉死成模板(case 2)——都是 RL 后期的典型副作用,也是"中期 ckpt 可能优于终档"判断的直接证据。

(收尾杂务:两台节点都确认无进程持有 /home 的 SIF 后,原件已可删,我顺手处理。)你消化完这两个 case,咱们再定测评方案(iter_119 单点 vs 119+47+7 三点)。


# ==== placeholder-retreat report (line 2654) ====

MI 验证结果出来了——**假设不成立,但翻出了两个更有价值的新事实**,而且要修正我此前 case study 的一个判断:

**1. 我们的"模板坍缩"其实只是前缀级 tic,不是推理坍缩**。全程 Retrieval-Acc 0.92-1.00(随机水平 0.042 的 20 倍以上)——即便在占位句最猖獗的 r88-100,盖住题目从 thought 反查源题依然几乎百发百中,thought 里照样精确引用本题的 `cache_node`/`lfucache.py` 这类专名。**占位句只是句首口头禅,冒号后面的推理内容对输入的依赖度没有丢**。RAGEN-2 的 MI 指标"看不见"它不是指标失灵,是我们的现象根本不属于他们说的那类坍缩。

**2. 新发现:这次坍缩是瞬态,自己好了**。echo 率 r84-104 冲峰(agent 统一口径下峰值 ~82%@r100),**r108 后回落,r116-119 只剩 ~1%**——不是单向终态。我此前 case study 里"晚期 93% 模板坍缩"的表述需要修正为"r84-104 的瞬态发作,末期自愈"(当时分析截止 r95,正好落在发作窗口内,被截断误导了)。

**3. mi_like 连续量有一条弱下行漂移**(形式上 r56 起,但反复穿越阈值、幅度小),只能算噪声边缘信号,不支持"更早预警器"。词法代理的天花板效应(24 题池,顶部饱和)也在,论文原版的似然打分不排除能看到更细的东西。

**对 v4 配方的净影响**:占位句问题降级(表面 tic + 自愈,prompt 修复仍做但不再是"健康警报"级);MI 在线监控降优先级(至少词法版没用,似然版成本高收益未证);**SNR 过滤的理由不受影响**(它的依据是梯度效率,不依赖坍缩叙事)。

我现在去把 case study 报告的相应段落修正掉(r84-104 瞬态 + 前缀级定性 + MI 验证附录),然后取 iter47-stateless 的分(刚才 75/76)。


# ==== placeholder-retreat report (line 11041) ====

actonly 体检报告也到了,连同刚才的 OOM 恢复一起汇总:

## 运维现状

- **actonly 臂:第二次 OOM**(签名与 gated pre-run 那次完全相同:train 步要 11.37GB、差 ~0.5GB),已从 iter_31 恢复重启(只损失 r32 一轮),正在爬起;
- **gated 臂:正常**,r29 推进中;
- OOM 两次复发 = 确认是配置贴边而非偶发,缓解决策见文末,**等你拍板,我不动**。

## 双臂体检结论(全量定量 + 22 条精读)

**总体:两臂都健康,无崩溃前兆**——前代七大病理中,长度失控、think-aloud 爆发、OK-spam 复读、memory 泄漏在两臂均为阴性;chars/turn、提交率、WRITE 结构化质量全部处于健康带。

**但每臂各有需要盯的苗头:**

| 臂 | 苗头 | 现状 | 危险机制 |
|---|---|---|---|
| gated | ①模板句感染(0.94% trial) | 一旦感染就全 trial 饱和,且已有 **感染+成功+reward 0.925** 的被强化样本 | grace12 r60 爆发的种子机制同款 |
| gated | ②修好不提交・空命令自贺循环 | r26 峰值 5 例(单例 33 连空),r28 回落 1 例 | reward=0 会被 RL 压制,但 grace12 也是零星起步 |
| actonly | ③**验证行为几乎绝迹(~1%,前代同期 10-16%)** | 8/8 精读零验证,"sed 返回 0 = 修好了"信念普遍;memory 里教训写得对、行为不跟随(知行分离) | 当前最实质的行为退化,直接压成功率(r30-31 已连跌 .172→.146) |
| actonly | ④模板占位符复读 r28 冒头、r30 冲 2.7% turns | 与"多 bash 块格式违规"同根 | 新型退化,前代没有 |
| actonly | ⑤乱码 token 侵入(0.2-0.5%/turn,含侵入 sed 命令正文) | 无增长趋势 | 语言失稳早期形态 |

**哨兵监控已建立**(纳入我的巡检,只报不动):gated 盯模板句 turn 率 + 感染得分数、≥8 连空命令数;actonly 盯占位符频率、成功率趋势、WRITE 截断率(r31 单点 5.7%)。任一连续 3 轮上行立即报你。

## 需要你拍板的一件事:OOM 缓解

两次复发满足了我之前说的触发条件。选项(MAX_TOK 不动):
- **A(我推荐)**:`max_split_size_mb:512` 经 `--train-env-vars` 只注入训练 actor——它不改段类型,理论上与 torch_memory_saver 兼容(TMS 只拒 expandable_segments);**先在 eval-only 短任务上实测验证兼容性**,证明能启动后再报你批准、择机(比如下次自然挂掉时)带上重启;
- **B**:继续裸跑,靠间隔 8 存档兜底(每次 OOM 损失 ≤8 轮 + 半小时重启);两次 OOM 分别发生在 r7 和 r32,频率约每 25-30 轮一次,全程 191 轮预计还会挂 5-6 次。

选 A 我就先去做兼容性验证(不碰正在跑的实验);选 B 就维持现状。


# ==== iter127-analysis (line 33823) ====

第 4 个 subagent(scaffold 一致性)完成,四个分析全部到齐。综合汇总:

## 一、scaffold 对齐核查:v2 评测没有问题

你的怀疑核查完了,结论是 **v2 评测与训练 scaffold 完全对齐**(置信度高,replace 侧逐字验证):

- 训练晚期(r100-160,15 万 turn)与 8 个 v2 评测 run 的**全部特殊情况反馈模板逐字一致**:多块拒绝文案、无 bash 块文案、too-long warning、提交成败、预算行、memory 包裹格式,全部相同;评测独有模板 **0 个**
- 训练时实际跑的就是当前代码:NFS 同一文件、md5 一致、mtime(08-08 15:06)早于训练启动;我删的 TAKE_FIRST 分支在训练轨迹里**零出现**(死分支,删除不影响行为)——我之前提的这个候选点排除
- 但它确认了两件事:①**非 v2 的旧评测(首轮 iter127、iter159)确实跑在错 scaffold 上**(runtime_env 里 `MULTIBLOCK_FEEDBACK=""`),作废正确;②唯一真实差异是**响应 token 上限 2500(训练)vs 32768(评测)**——训练期 14.8% 的 turn 因 2500 截断判空,评测几乎没有。方向上利好评测,不能解释 replace 落后,但值得记着:模型是在 2500 上限下训出来的

## 二、memory 副作用的机制(三路轨迹证据一致)

**分差不是 scaffold 造成的,是行为层面的"探索压制":**

1. **memory 的事实基本是对的,毒性在于它改变行动路线**:前 3 命令含全库探索的比例 stateless 87% → 带 memory 38%(memory 为空的位置 0 是 100%,归因干净)。模型拿到"地图"就跳过 find/读全文件,3-6 步盲改盲提交。
2. **分差高度集中**:iter119 的 1.81 分差 82% 来自 3 题(609/534/540),这些恰是"需要新鲜整读文件才能看清语义/位置"的题——609 是 replace 5/5 把 AND/OR 扁平化语义写错(stateless 读了全文件 3/4 写对);534 是被 memory 里**上一题的修复位置**锚死(4/5 开局就在 `_set_dict` 打转);613 是捷径命令 `nl -ba` 恰好把文件顶过截断线导致全程盲改。
3. **memory 的效率收益为零**:配对比较 turns 只差 ~1 步。所以 replace 的期望收益(省探索)没兑现,风险(锚定+跳过阅读)全额支付。
4. **排除项都排干净了**:上下文挤占(截断 0 次、memory 仅 ~1KB)、位置衰减(非单调)、格式干扰(空命令率 stateless 反而更高)、记错事实(未发现实质错误)。
5. **流程类记忆完全无效**:所有 memory 都写着"提交前跑测试",失败 trial 没有一次执行;而"用 heredoc 整文件重写"这类手法信念倒是被忠实执行并致害。

一句话回答你的问题:**为什么 memory 有副作用——因为策略学会的是"用 memory 替代探索"而不是"用 memory 补充探索";裸能力涨在了需要探索的题上,replace 恰好在这些题上把探索关了。** 统计上要注意:结论压在少数题、5 题序样本上(119 的成功率差 z≈1.55),要坐实可以加题序/seed 重跑,flame-10 现在空着随时能跑。


# ==== gwin-think r0-48 health check: three sub-agent report summaries (recovered from transcript lines 46169/46180/46185) ====


## agent report (line 46169)

报告已写入:`/mnt/tmp/6016127/claude-6016127/-home-qixinx/2ad71a1e-0178-4da9-90ac-4a152bc5c82c/scratchpad/casestudy_think/ACT_think_format.md`
(中间文件同目录:`per_rollout.csv`、`dig_out.txt`、`prefix_out.txt`、`excerpts.txt`(8 个 case 摘录)及脚本 `stats.py/dig.py/excerpts.py/prefix.py`;目录内 `agg*.py / think_samples_*.jsonl / train_stats.*` 等不是本次产物。)

要点摘要(rollout 0–48,9408 trial / ~239k turn,全程只读):

**结论:格式与思考"正常,有两条行为性隐患"。**

1. thinking 结构:`assistant_think` 100% 存在,为空仅 0.1–0.3%(其中 62% 是模型没开 `&lt;think&gt;`、把 9–11k chars 推理写进可见正文被 length 截断,该类 turn 每 8-rollout 窗 69→2 逐窗下降)。think 长度 p50 165→262 chars、p90 ~1.7k→2.1k、均值 567→680(约 +20%),无爆长;length 截断率 ≤0.34%,"截断+正文为空"全程 2 例。正文残留 `&lt;think&gt;` 0.04%;`&lt;tool_call&gt;` 集中在 51 个 trial(模型自造 `&lt;tool_call&gt;…&lt;reasoning&gt;` 格式整 trial 锁死 40 turn 空命令),每窗 13→0 下降。单 bash 块 &gt;99.4%。
2. 可见正文:空命令率 0.28–0.70%(单 rollout 最高 1.67%,r40+ 全 ≤1%,r48 0.28%),对照组同期 1–2%、崩坏期 18–74% —— think 组目前健康且逐窗下降(caveat:对照崩坏在 r160+,进度不可比)。自言自语留在 think 内(17–21%),正文 &lt;1%;CJK 混入 &lt;0.5%,r29 峰后回落。**隐患 1**:r31 起可见正文把 prompt 里 `&lt;format_example&gt;` 的占位句 "Your reasoning and analysis here." 原样吐出,r42–47 占 20–29% turn(r47 623/1288 一字不差),r48 14%;不影响命令解析,但可见 reasoning 正退化为占位符。
3. 效率/成功:success 0.32–0.40 波动无趋势,mean reward 0.18→0.28,mean turns 30→13,submit 率 57%→97%。**隐患 2**:提交前跑过 test 的比例 87%→23%,完全不跑 python 就提交 0.6%→31%(r48);late 期 notest 提交成功率 0.38 已接近 tested 0.50,即"看一眼 diff 就交"性价比高,验证行为在退化。建议阈值:blind submit &gt;50% 或 success 下滑即干预;占位句 &gt;40% 或伴随空 bash 块视为异常。
4. Case:早期正常(A)、`&lt;tool_call&gt;` 锁死(B,think 正常但动作发不出)、忘开 think 漏推理(C)、中期正常(D)、think 中英混杂(E)、晚期 5-turn blind submit 一成一败(F0/F1)、晚期正常但正文全是 "Your reasoning is to…"(G)、think 重复为假警报——是引用代码(H)。思考内容整体切题、与命令一致、无成段重复/跑题。

## agent report (line 46180)

报告已写入:
`/mnt/tmp/6016127/claude-6016127/-home-qixinx/2ad71a1e-0178-4da9-90ac-4a152bc5c82c/scratchpad/casestudy_think/WRITE_and_heldout_think.md`
(case 原文:同目录 `WRITE_excerpts.txt`;逐 rollout 数据 `train_stats.json`、`heldout_stats.json`)

要点摘要(r0–r48,每 rollout 192 条 WRITE):

1. **WRITE 格式:正常 → 有隐患苗头**。`&lt;/think&gt;` 闭合率 r0–39 ≥99.5%,r40–47 98.5%,r48 96.4%(7/192)。未闭合样本 100% finish=stop、tokens 400–650(不是截断,是在 think 内写完就停);未闭合时 memory 逐字 = 整段独白("The user wants me to update the memory… Let me analyze…")并作为下一题 memory_in 灌入(已核实)。机制与 K=3 对照 r155 后坍缩同型,目前处于起点。
2. **奖励漏洞**:r48 有 2 例未闭合样本因独白里含两节 header 被判 format_ok=True、write_reward=0.875、trained=True —— "不写 &lt;/think&gt;" 有被正强化的通道。
3. **memory 内容亚健康**:不是编造(路径 ~90% grounded)、不是占位句(&lt;0.05/条),而是"越写越短+每题清仓":长度 3100→1400 字符、bullets 19→7、与上一版 bullet 保留率 0.58→0.09(模糊 0.70→0.10),episode 内不再累积;think 长度反而 1466→2112。早期(r0–7)另有 12–22% 前言+```围栏瑕疵,r16 后消失。format_ok 在 r24–31 有低谷(0.62,清仓式 mega-bullet 被罚 0),原因为推断(未读判定代码)。
4. **write_reward** 均值 ~0.03–0.04、std 0.22–0.35,无趋势,信号弱。
5. **heldout**:ACT 格式健康(空命令 &lt;1.5%、length 截断 &lt;1.2%、think 0 例未闭合);reward .077/.062/.068/.127/.137/.052 主要由 tenacity-628 一题驱动(n=30 噪声大),与 K=3(均值 .079 vs .087)无可分辨差异;turns 36→27.5 下降但成功率未升;后续题前 3 步命中 memory 路径 0.56→0.44;iter39 heldout WRITE 有 3/30 未闭合。
6. 建议盯 r49+ 的 WRITE 闭合率(单 rollout &lt;95% 或连续下降即视为已异常)。Caveat 见文末(format_ok/write_reward 口径未读代码;保留率为字符串匹配;heldout 小样本)。

## agent report (line 46185)

完成。产物目录:`/mnt/tmp/6016127/claude-6016127/-home-qixinx/2ad71a1e-0178-4da9-90ac-4a152bc5c82c/scratchpad/casestudy_think/`

- 报告:`THINK_content_quality.md`
- 摘录:`think_excerpts.txt`
- 样本:`think_samples_{0-4,22-26,44-48}.jsonl`(每窗 40 turn);统计旁证 `agg_*.json`、`agg2_*.json`、`agg3_per_rollout.txt`

**要点(精读 3×40 条 + 全量启发式旁证)**

1. **think 内容本身:正常,未崩坏。** 切题率 100%(无复述 prompt/memory、无胡言),有效推理 2–3 分占比三窗 43%→53%→50%;思考-动作不一致仅 2–3/30(零星,如 sed 空操作、说"换方法"仍用同一 sed);语言混杂 0.3–0.4%、think&gt;3000 字 ~3.7%、Wait≥5 次 0.6–0.9%、开头去重比 0.97 —— 三窗几乎不变,think 不模板化;think 为空/极短的比例还在下降。中位长度 164→193→233,增量主要在 trial 早期 turn(179→279),p99 反而下降,不是长尾恶化。

2. **有隐患(可见正文,不在 think):模板占位符回显。** 可见正文以 prompt `&lt;format_example&gt;` 里的 "Your reasoning and analysis here…" 开头,占比 r0–30 ≈0%,r34 5.9%,r40 9.9%,r42–47 20–29%(r47 29.2%),单调上升。后接内容仍切题,属格式套话化/奖励下的格式坍缩迹象,建议跟踪并考虑在 format 判定中惩罚。

3. **length 截断成因迁移。** 绝对数下降(103→60→36),但成因从"跳过 think、正文冗长"(早窗 7/10)转为"think ≥3000 字反复 Wait 打转"(晚窗 4/10,全量占比 32%→44%)或整文件 heredoc 重写;有样本 think 吃光预算导致正文只剩半句、无命令(r25 pandas,think 10852 字)。仅 1 例命令内失控重复。

4. 背景:每 trial 平均 turn 30.5→25.3→18.6,成功率 0.34→0.39→0.32 —— 晚期 trial 更早结束但没更成功。

5. 疑为基座风格、非训练引入:约 2–3% 的长 think 尾段"总结式复述"前文,三窗持平。