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
   一个进程跑完所有 trial 也只写这一个文件；`set_trial` 写在行里，所以照样能按 trial 拆开。每行还带一个 `run`（进程启动时间 + pid）：文件是追加写的，同一个 seed 重跑一次就会把同样的 trial 号再写一遍，只有 trial 号的话第二次会被当成第一次的更多证据，两次结局不同的话更会被读成一个既成功又失败的 trial。
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

- `bank.yaml` 有**一张卡片**（`relief-under-a-squeeze-is-the-crush`，见第三轮），其余等挖。空库仍然是支持的状态。
- 测试：`tests/tactile_memory/` 88 条全绿。
- 未跑 eval。
- **M0 在历史数据上没过，也没法过**（见下）：122 个好 run / 40 个坏 run 全在 `outputs/`，但每轮只有 3–5 个 `Tactile:` 读数，且没有剪切/anisotropy。能分开的档只有两类——"最深读数还是远平面"（根本没碰到）和 `unknown`（只读了一次，观测稀疏）。两者都是策略/观测密度的事实，不是触觉知识。要过闸门必须跑带 trace 的 trial。
- **两次带 trace 的真相试跑（2026-10-28，blind_pick_balls seed 8，gpt-6-astra）**：trace 机制成立（行里有剪切和 anisotropy），但两轮都不可挖——都死在第一次下探，读到的全是远平面，程序各只有 1 个 code block。
  - 第一轮（`8714b4e`，10:27–10:34，431 s）：新库丢了旧库的任务事实，程序用 `linspace(0.360, 0.660)`，两头出界，第一次下探就被规划拒绝。已把带补回 `core.yaml`。
  - 第二轮（`42807e9`，241.86 s）：补回来的带写成了 **0.38–0.63**，程序照抄成 `linspace(0.380, 0.630, 11)`；第一次下探（x=0.380，法兰 0.168 ＝垫心 0.050）走了 206 mm，停在离目标 **11.2 mm** 处，超过适配器 5 mm 的 settle 容差 ⇒ 客户端抛 `RuntimeError: Motion failed`，程序没接异常，整轮结束。同批 6 个成功程序全部从 **x ≥ 0.39** 起步（`linspace(0.39, 0.62)`）。旧库 A25 的实测表里 0.380 本来就只是"垫心到 0.0476、呼叫被拒"，**真到位的点是 0.390**；0.38 与 0.63 是我把记忆里的探测端点当成了安全带，两头各放宽一格。已改成 **0.39–0.62**，并把后果写进去（停住 ⇒ motion failure ⇒ 整轮结束）。
  - 顺带证伪了一个猜测：两轮失败都不是朝向造成的——第二轮硬编码的 `[0,1,0,0]` 被接受（成功程序里 trial_02/05 也这么写），同一个四元数的水平移动也成功了，只有下探那一步停住。
- **第三轮（`5171fbc`，889.21 s）——第一次真的拿起一颗球，但那是一颗光滑球**：第一次合爪（`force_threshold=26.5, adaptive=True`）读到 relief 0.152/0.155、两垫 depth 26.49/26.11、disp 7.03/7.00，程序据此判"带凸点"，抬到 0.403、搬到碗上方，全程两垫都夹着。死在最后一次下探：命令垫心 0.090（腕 0.208），手走了 192 mm、停在离目标 **5.0 mm** 处，越过适配器 5 mm 的 settle 容差 ⇒ 抛 motion failure ⇒ 程序没接异常，整轮结束。
  - **抓住的是 4 号球，`rubber`/`soft`——光滑球**。服务端 `physical_trace.jsonl` 里 held 出现在 1142–1616 步，`verified` 也只有它。所以这一轮不是"差一次 `open_gripper`"：碗里要放下的是错误材质，规则里那就是失败。真正的死点在**第一次合爪**——判材质的那一刻就判错了，后面搬得再好也是白搬。
  - 为什么读错：26.5 的深度伺服把球压住了（那一刻两垫各约 **120 N**），起伏量到的是压痕不是表面。同一条通道只在**固定开口**下才是表面测量：按 `position=0.6`（48 mm）坐上去，光滑 0.019–0.038（polymer 0.019–0.029、rubber 0.038）、凸点 0.088–0.246，阈值 0.06 分得开；压碎态的光滑橡胶球读 0.15，正落在凸点带里。depth 也救不了：真坐姿的压深 0.6–2.3 mm ⇒ depth 26–27.7，跟压碎态的 26.5 重叠。
  - 知识分成两层补进去了：
    - **任务事实**（`core.yaml` 第四条）只写"什么时候这个量才是表面量"——起伏只在按开口设定握宽时才测表面，按 gel 深度合爪会一直挤下去、读到的数是压痕。这段话在动手前就该知道，漏了就是整轮报废。
    - **卡片**（`bank.yaml` 第一张，`when: {relief: rough}`）放数字与判断：固定 0.6 开口（48 mm）下光滑 0.02–0.04、凸点 0.09–0.25；深度伺服压碎时光滑橡胶球读 0.15。数字只有在起伏真是 rough 时才值得花字，所以它进卡片。
    - 这张卡片**不能只靠键自证**：两种握法在键上长得一模一样（depth 同为 pressed、relief 同为 rough），卡片无法区分"凸点球"和"被压碎的光滑球"，所以它写的是"去比一次按开口坐的读数"，而不是直接给结论。卡片内容在两种情形下都成立。
  - 交付那一下的教训仍然成立（只是对这一轮没有意义）：6 个成功程序全部从 **腕 0.23（垫心 0.112）** 松手，或把交付下探包在 try/except 里（trial_05 的注释就写着 "A refused descent leaves the held ball above the bowl"）；这一轮把目标压到垫心 0.090，比任何成功程序都低，正撞上"带球下探会停短"——新库把旧库的 P8 丢了，已作为第三条任务事实补回。
  - 这是**第一条可挖的 trace**，最深读数是一次真实抓取，不再是远平面；但它同时是起伏通道的一条反例（压碎态光滑球读成凸点），M0 上它只算一颗球、一种情况。
- **归因规则（2026-10-28）**：一个目录就是一个 trial——同号的两个目录（重跑结局不同）算两个，以前后一个会把前一个盖掉；run 级 trace 按 (`run`, `trial`) 切分，以前只按 `trial`，重跑会把两次尝试并成一个；某个 trial 号对应两个目录时，这些行无法归属到具体某次跑，改用各自目录里的 `summary.txt`（丢掉剪切和 anisotropy，但不猜）。
