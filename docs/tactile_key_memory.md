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

`capx/tactile_memory/signature.py` 把 `get_tactile_data()` 的返回压成 11 个维度，每个维度取一个档，绝不取数：

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

前提：main 上的 `CAPX_TACTILE_TRACE`（每次读数一行 JSONL）。没有 trace 就没有可挖的东西——历史日志的 `Tactile:` 行缺剪切和 anisotropy，正好是最关键的两组。

1. 跑带 trace 的 trial：
   `CAPX_TACTILE_TRACE=outputs/<model>/univtac_<task>/trial_XX python ...`
2. 分档统计：
   `python scripts/build_tactile_bank.py --success <好跑的根> --failure <坏跑的根>`
3. **闸门（M0）**：如果档位分不开好跑和坏跑，卡片库帮不上忙，停在这里，不要先去调 prompt。
4. 只把分开的档位手写进 `bank.yaml`——脚本只出报表，不出卡片。档位是传感器的事实，卡片上那句话是判断，判断不能挖，只能写。
5. A/B：`CAPX_TACTILE_BANK=<路径>` 指向空库或满库，代码不动。

读数不是独立样本（一轮贡献几百行，彼此高度相关），所以报表回答的是"坏跑会经过哪些档位"，不是显著性。只在坏跑里出现过的档位值得看，到处都出现的档位不值得。

## 这个分支删了什么

- `capx/tactile_skills/`（61 条技能 + `library.py` + `taxonomy.yaml` + `MAINTENANCE.md`，648K）
- `tests/integrations/test_tactile_skills.py`（44 条）
- `docs/univtac_tactile_skill_inventory.md`、`docs/univtac_tactile_skill_library.md`
- `base.py::_tactile_skill_block()`（按任务名整段投递）、灯泡感知线上的 `tactile_skills = (...)` 元组

main 上这些原样保留；本分支不借用、也不兼容旧库。

## 现状（2026-10-27）

- `bank.yaml` 是**空的**——这是支持的状态，不是坏的：常驻原则撑着，卡片等挖。
- 测试：`tests/tactile_memory/` 62 条全绿。
- 未跑 eval。
