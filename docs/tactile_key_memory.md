# 触觉键记忆库（`feat/tactile-key-memory`）

## 一句话

原来的 `capx/tactile_skills` 按**任务名**路由：61 条散文里凡是挂到该任务的，全部整段投进 prompt，哪几条对得上眼前这个读数，交给模型自己挑。这一版改成按**触觉读数本身**检索——读数归约成一组分档（`depth=face`、`coh_gap=split`…），只有当某张卡片声明的档位与当前读数吻合，它才出现在模型眼前。常驻的只剩六条原则和极少数任务事实。

## 三层，以及为什么这么分

| 层 | 在哪 | 什么时候出现 | 为什么在这一层 |
|---|---|---|---|
| 常驻原则（6 条） | `capx/tactile_memory/core.yaml::principles` | 每个能读凝胶的任务，每次 | 对任何任务都成立，且 API docstring 说不了——docstring 讲字段**是什么**，原则讲它**不是什么**（单帧不是结果、坏垫不读成坏） |
| 任务事实 | 同上 `task_lines` | 只对命中的任务 | 只在一个任务上成立，但漏掉的代价是整轮跑废（例：合爪报错＝球坐上了） |
| 卡片 | `capx/tactile_memory/bank.yaml` | 只在**当前读数**命中时 | 窄、带数字，只有在适用时才值得花字 |

判据一句话：能在动手前说清楚的进前两层，必须看着凝胶才能说的进卡片层。

## 键：一个读数，一组档位

`capx/tactile_memory/signature.py` 把 `get_tactile_data()` 的返回压成 14 个维度，每个维度取一个档，绝不取数：

| 维度 | 档位 | 切点依据 |
|---|---|---|
| `contact` | none / left / right / both | 接触标志是**量程**标志，不是压上 |
| `depth` | missing / far / near / face / pressed / floor | 34 远平面、28.3 凝胶面、24 相机地板 |
| `asym` | unknown / even / left / right | 两垫深度差 0.5 mm＝单边加载 |
| `disp` | missing / still / trace / light / moving / heavy | 空合爪 <0.01 px，真抓 1.5 px+ |
| `disp_gap` | unknown / even / split | 一垫被爪拖着走时两垫差 20 px 级 |
| `coh` | missing / lost / low / mid / high | 健康 0.42–0.68，坏垫 0.09–0.10 |
| `coh_gap` | unknown / even / split | 同上，取两垫之差 |
| `shear` | none / small / medium / large | 剪切向量模长 |
| `shear_axis` | none / x / y | 某一分量是另一分量 2 倍以上才命名轴 |
| `aniso` | unknown / iso / banded | 1.0＝均匀变形 |
| `relief` | unknown / smooth / rough | 光滑 0.014–0.038 mm，凸点 0.09–0.25 mm |
| `depth_trend` | unknown / falling / static / rising | 与上一次读数比，0.2 mm 以内算静止 |
| `disp_trend` | unknown / static / growing | 位移是自 reset 累计量，所以趋势＝速率 |
| `shear_trend` | unknown / static / growing | 同上，0.1 px 以内算没动 |

后三个维度要两次读数，第一次读数一律 `unknown`——这是诚实的答案，不是缺省。存在它们是因为常驻原则第一条（"每个字段是状态不是事件"）必须能用卡片表达：最值钱的判断几乎都是**变化**（"距离还在变小"、"已经不再下沉"、"标记还在被拖"），静态档位说不出这些。两垫各自"上一次"由 API 自己记（`_remember`）：任何一次带 `tactile` 的动作回包和每一次 `get_tactile_data` 都会更新它，所以比较的两端永远是相邻的两次观察，顺序与 trace 行一致。

唯一的错配风险：`/reset` 不保证回触觉，所以新 episode 的第一次读数是跟上一 episode 的最后一次比。一次而已，且只影响 `*_trend` 三个维度。

两条约定写死在代码里：

- **0.0 是"没报"，不是"零"**。深度/位移/相干/起伏的 0.0 一律进 `missing` 档。剪切向量是唯一例外——0.0 在那里也是诚实读数（两个方向的位移正好抵消），所以 `shear=none` 分不清"没剪切"和"没上报"，这也是 `contact` 必须单列一个维度的原因。
- **键零输入**。模型永远不提供数字；键是从服务端已经发回来的响应里算的。所以卡片不可能匹配到一个从没取过的读数，模型记错一个数也不会把检索带偏。

## 检索什么时候发生

只有 `get_tactile_data()` 一次调用：读完之后按当前读数算键、取出命中的卡片，打在标量下面（`TACTILE MATCHES (this reading):`）。返回字典的形状不变，模型看到的还是那些字段。

选它而不是新加一个 `recall_...()` 工具，是因为加了工具就等于把实验变成"模型会不会想起来问"——检索是自动的，A/B 才只量知识本身。（代价：每次读数都算一遍键，纯本地几微秒。）

任务名从 API 持有的 env 上取；取不到就只有跨任务卡片能命中。

## 卡片长什么样

```yaml
- id: dead-pad-after-close
  tasks: [blind_pick_balls]        # 省略 = 任何任务
  when: {disp_gap: split, coh_gap: split}
  do: >-
    Read the pad that is still coherent; the quiet one describes the drag.
  why: 可选，给人看的出处。不投给模型。
  evidence: [blind_pick_balls/seed_3/trial_12]
  status: observed                 # 或 hypothesis
```

加载时会被拒的：不存在的维度、越界的档位、重复 id、`when` 为空、`do` 超过 240 字——以及**任何提到 `check_success` / reward / ground truth 的字样**。卡片只能写凝胶报告的事实，这样同一套库才能投给没有 ground truth 的任务。

`status` 必须诚实：`observed` 留给真的对着结局量过的卡片，`hypothesis` 是"见过但没量过"。两者都会投递，但状态跟着文本一起走。

## 挖卡片的流程

前提：main 上的 `CAPX_TACTILE_TRACE`（每次带回触觉的调用一行 JSONL，每行带 `trial` 号）。没有 trace 就没有可挖的东西——历史日志的 `Tactile:` 行缺剪切和 anisotropy，正好是最关键的两组。

1. 跑带 trace 的 trial：
   `CAPX_TACTILE_TRACE=outputs/<model>/univtac_<task> python ...`
   一个进程跑完所有 trial 也只写这一个文件；`set_trial` 写在行里，所以照样能按 trial 拆开。
2. 分档统计：
   `python scripts/build_tactile_bank.py --root outputs/<model>`
3. **闸门（M0）**：如果档位分不开好跑和坏跑，卡片库帮不上忙，停在这里，不要先去调 prompt。
4. 只把分开的档位手写进 `bank.yaml`——脚本只出报表，不出卡片。档位是传感器的事实，卡片上那句话是判断，判断不能挖，只能写。
5. A/B：`CAPX_TACTILE_BANK=<路径>` 指向空库或满库，代码不动。

**单位是 trial，而且是 trial 的一次读数**：取**最深的那次**（两垫深度最小；并列取最早）。理由不是省事：一个 trial 会走过很多状态，如果按"经过过哪些档"统计，几乎每个档都覆盖几乎每个 trial——那是"手臂动过"的事实，不是结局的事实，报表永远分不开。每个 trial 只取一次最深读数，档位重新变成对 trial 的划分，坏率才有资格和基线比。

读报表要看的是**窄而坏率高**的档：宽档的坏率必然贴近基线（那是地板，不是信号）。另外两种"分离"不算数——最深读数还是远平面（根本没碰到，模型自己看得见）、以及 `unknown` 档（模型只读了一次，是观测稀疏不是知识）。

## 这个分支删了什么

- `capx/tactile_skills/`（61 条技能 + `library.py` + `taxonomy.yaml` + `MAINTENANCE.md`，648K）
- `tests/integrations/test_tactile_skills.py`（44 条）
- `docs/univtac_tactile_skill_inventory.md`、`docs/univtac_tactile_skill_library.md`
- `base.py::_tactile_skill_block()`（按任务名整段投递）、灯泡感知线上的 `tactile_skills = (...)` 元组

main 上这些原样保留；本分支不借用、也不兼容旧库。

## 现状（2026-10-28）

- `bank.yaml` 是**空的**——这是支持的状态，不是坏的：常驻原则撑着，卡片等挖。
- 测试：`tests/tactile_memory/` 84 条全绿。
- 未跑 eval。
- **M0 在历史数据上没过，也没法过**（见下）：122 个好 run / 40 个坏 run 全在 `outputs/`，但每轮只有 3–5 个 `Tactile:` 读数，且没有剪切/anisotropy。能分开的档只有两类——"最深读数还是远平面"（根本没碰到）和 `unknown`（只读了一次，观测稀疏）。两者都是策略/观测密度的事实，不是触觉知识。要过闸门必须跑带 trace 的 trial。
- **第一次带 trace 的真相试跑（2026-10-28，blind_pick_balls seed 8）**：trace 机制成立，7 行里有剪切和 anisotropy；但这轮不可挖——程序在第一次下探时就被运动规划拒绝，25 秒就结束了，7 个读数全是远平面。原因是新库丢了旧库的一条任务事实：旧 prompt 写着"球心高度只在 x≈0.39–0.62 够得着"，之前 6 个成功程序全用 `linspace(0.39, 0.62)`；新 prompt 没有了，这一轮用 `linspace(0.360, 0.660)`，两头都出界（该任务拒绝一次动作就整轮闩死）。已作为任务事实补回 `core.yaml`。
