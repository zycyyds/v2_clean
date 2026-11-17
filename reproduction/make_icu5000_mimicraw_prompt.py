#!/usr/bin/env python3
"""make_icu5000_mimicraw_prompt.py —— 从 icu5000 mimicrawerror 数据集生成 agent prompt。

与 K29 版的区别：K29 的 prompt 是**整篇模板**生成的；icu5000 的基础 prompt
（eval_handoff/prompts/icu5000_{1.6pct,4.7pct}_val_prompt_v1.md）已经存在且
定稿，本脚本是**插入式**的：

  ① 把「错误注入量级」那三行的数字换成 mimicraw 档的实测值
     （mimicraw 档在注入错误之外还多了一类 MIMIC 原生错误）
  ② 在「### 推导与修正流程」之前插入一节
     「## 新增错误类：MIMIC 自身的原生错误（不是注入器造成的）」
  ③ 再插入一小节「itemid 列上的 CareVue 别名注入」
     （icu5000 特有：test 里 163 万条 key 列注入，不修就大面积键对不上）

prompt 里**只用聚合统计**（每 itemid 的格数、值的范围、比例），
不出现任何 subject_id / hadm_id / stay_id / 病人级取值。

用法：
  python3 make_icu5000_mimicraw_prompt.py \
      --dataset <mimicraw 数据集目录> \
      --base-prompt <基础 prompt.md> \
      --stats <icu5000_native_stats.py 的输出 JSON> \
      --out <输出 prompt.md>
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(2 ** 31)
CE = "icu/chartevents.csv"

HARD_RANGE = [
    (220045, "heart rate（心率）", 20, 300),
    (220050, "arterial BP systolic（动脉收缩压）", 40, 400),
    (220051, "arterial BP diastolic（动脉舒张压）", 20, 250),
    (220052, "arterial BP mean（动脉平均压）", 20, 250),
    (220059, "pulmonary artery pressure systolic（肺动脉收缩压）", 0, 150),
    (220061, "pulmonary artery pressure mean（肺动脉平均压）", 0, 100),
    (220074, "central venous pressure（中心静脉压）", -20, 80),
    (220179, "non invasive BP systolic（无创收缩压）", 40, 400),
    (220180, "non invasive BP diastolic（无创舒张压）", 20, 250),
    (220181, "non invasive BP mean（无创平均压）", 20, 250),
    (220210, "respiratory rate（呼吸频率）", 0, 100),
    (223762, "temperature celsius（体温 ℃）", 25, 45),
    (224688, "respiratory rate (set)（呼吸频率-设定）", 0, 100),
    (224689, "respiratory rate (spontaneous)（呼吸频率-自主）", 0, 100),
    (224690, "respiratory rate (total)（呼吸频率-总计）", 0, 100),
    (227466, "ptt（凝血酶原时间）", 8, 200),
    (224167, "manual BP systolic left（人工左侧收缩压）", 40, 400),
    (224643, "manual BP diastolic left（人工左侧舒张压）", 20, 250),
    (227242, "manual BP diastolic right（人工右侧舒张压）", 20, 250),
    (227243, "manual BP systolic right（人工右侧收缩压）", 40, 400),
]
HARD = {i: (lo, hi) for i, _, lo, hi in HARD_RANGE}
LABEL = {i: lab for i, lab, _, _ in HARD_RANGE}
SENTINELS = [-999, -998, -997, -990, 9999, 99999, 999999, 9999999, 220000, 220045]

# 枚举型 itemid：同一时刻多条不同取值是合法的，不是重复行。
# 这两个在 icu5000 的 reference 里同样存在（实测行数 / 多值组数见 stats JSON）。
ENUM_ITEMS = [(230033, "Stool Management（排泄管理）"),
              (229864, "Pressure Injury Location（压疮部位）")]

ANCHOR = "### 推导与修正流程"


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_log_stats(ds, split):
    p = ds / split / "modification_log.csv"
    if not p.is_file():
        return None
    op = Counter()
    sub = Counter()
    n = 0
    with p.open(newline="", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            n += 1
            op[e.get("operation", "")] += 1
            sub[e.get("error_subtype", "")] += 1
    return {"n": n, "op": op, "sub": sub}


def native_stats(ds, split):
    p = ds / split / "mimic_native_error_log.csv"
    if not p.is_file():
        return None
    per_item = Counter()
    per_mode = Counter()
    inj = Counter()
    n = 0
    with p.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n += 1
            per_item[r["itemid"]] += 1
            per_mode[r["detection_mode"]] += 1
            inj[r["also_had_injection"]] += 1
    return {"n": n, "per_item": per_item, "per_mode": per_mode, "inj": inj}


def train_examples(ds):
    """train 里注入的原生错误示例 —— 按 itemid 聚合成可安全公开的描述。"""
    p = ds / "train" / "mimic_native_error_log.csv"
    if not p.is_file():
        return [], set(), {}
    rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
    per_item = defaultdict(list)
    for r in rows:
        per_item[int(r["itemid"])].append(r)
    desc = []
    for it, rs in sorted(per_item.items(), key=lambda x: -len(x[1])):
        vals = [fnum(r["mimic_native_value"]) for r in rs]
        reps = [fnum(r["repaired_value"]) for r in rs]
        vals = [v for v in vals if v is not None]
        reps = [r for r in reps if r is not None]
        desc.append({
            "itemid": it, "label": LABEL.get(it, ""), "n": len(rs),
            "lo": min(vals) if vals else None,
            "hi": max(vals) if vals else None,
            "reps": sorted(set(reps)),
            "n_peer": sorted(set(int(r["n_peer_readings"]) for r in rs)),
        })
    return desc, set(per_item), per_item


def val_example_stats(ds, train_items):
    p = ds / "validation" / "mimic_native_error_log.csv"
    if not p.is_file():
        return {"n": 0, "items": [], "new_items": []}
    rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
    items = sorted({int(r["itemid"]) for r in rows})
    return {"n": len(rows), "items": items,
            "new_items": [i for i in items if i not in train_items]}


def train_rows(ds):
    """train raw 的总行数（给「数据量差 X 倍」那行用）"""
    p = ds / "train" / "raw" / CE
    if not p.is_file():
        return None
    n = 0
    with p.open(newline="", encoding="utf-8") as f:
        for _ in csv.reader(f):
            n += 1
    return max(0, n - 1)


def fmt_num(x):
    return "{:,}".format(int(x))


def build_section(ctx):
    """新增的两节（原生错误类 + itemid 别名类）"""
    dev = ctx["dev"]
    cvp = ctx["cvp"]
    alias = ctx["alias"]
    vec = ctx["vec"]

    tbl = "\n".join("| %d | %s | [%d, %d] |" % (i, lab, lo, hi)
                    for i, lab, lo, hi in HARD_RANGE)

    ex_lines = []
    for d in ctx["examples"]:
        reps = "、".join("%g" % x for x in d["reps"][:3])
        ex_lines.append(
            "- **itemid %d（%s）**：train raw 里该测量项有 **%d 格**是孤立尖峰，"
            "取值 %g ~ %g；train reference 把它们改成了 **%s**（= 该患者该测量项其余读数的中位数）。"
            % (d["itemid"], d["label"], d["n"], d["lo"], d["hi"], reps))

    enum_lines = []
    for i, lab in ENUM_ITEMS:
        s = str(i)
        enum_lines.append(
            "- **itemid %d（%s）**：test reference 里 %s 行，其中 **%s 组**是"
            "「同一 (subject_id, hadm_id, stay_id, charttime, itemid) 出现多条不同取值」"
            "—— 这是**合法的多值记录**（多个部位 / 多个选项），**不是重复行**，不要去重删除。"
            % (i, lab, fmt_num(ctx["enum"]["rows"].get(s, 0)),
               fmt_num(ctx["enum"]["multi_groups"].get(s, 0))))

    if dev:
        dev_txt = ("本数据集 test 上 **%s 个原生错误格**里，与自身常态"
                   "（该患者该项其余读数的中位数）相差**不到 ±20%%** 的格数是 **%d**；"
                   "这些错误格的偏离中位数是 **%g 倍**"
                   % (fmt_num(dev["n_cells"]), dev["n_lt_20pct"], dev["median_dev"]))
    else:
        dev_txt = "（本档未测得偏离度统计）"

    if cvp:
        cvp_txt = ("本数据集实测：中心静脉压（itemid %d）的原生错值在 **%g ~ %g**，"
                   "把 %d 格里的 **%d 格**除以 100 之后又落回 [%d, %d] 区间内 "
                   "（看着完全合理）⇒ **这条判据在 CVP 上会 %d/%d 全错**"
                   % (220074, cvp["min"], cvp["max"], cvp["n"],
                      cvp["div100_back_in_range"], -20, 80,
                      cvp["div100_back_in_range"], cvp["n"]))
    else:
        cvp_txt = "（本档未测得 CVP 统计）"

    if vec and vec.get("unit_scale_error"):
        u = vec["unit_scale_error"]
        s = vec.get("sentinel_chart_value")
        vec_txt = ("实测 `unit_scale_error` 的 %s 格里 %s 格（%.1f%%）、"
                   "`sentinel_chart_value` 的 %s 格里 %s 格（%.1f%%）的 `value` "
                   "直接等于 log 里登记的 clean_value"
                   % (fmt_num(u["sampled"]), fmt_num(u["value_eq_clean"]),
                      100.0 * u["value_eq_clean"] / max(1, u["value_nonempty"]),
                      fmt_num(s["sampled"]), fmt_num(s["value_eq_clean"]),
                      100.0 * s["value_eq_clean"] / max(1, s["value_nonempty"])))
    else:
        vec_txt = "（本档未测得 value 对照统计）"

    vr = ctx.get("value_reliability")
    if vr and vr.get("summary"):
        s2 = vr["summary"]
        parts = []
        for st in ("unit_scale_error", "sentinel_chart_value"):
            d = s2.get(st) or {}
            c = d.get("classes", {})
            eq = c.get("eq_str（value == clean_value）", 0)
            non = c.get("not_num（value 不是数值）", 0)
            n = d.get("sampled", 0)
            parts.append("`%s` 的 %s 格里 **%s 格（%.1f%%）**的 `value` 文本列正好等于"
                         " log 登记的 clean_value，另外 %s 格的 `value` **根本不是数值**"
                         % (st, fmt_num(n), fmt_num(eq), 100.0 * eq / max(1, n),
                            fmt_num(non)))
        vec_txt = ("实测：" + "；".join(parts) + "。\n\n"
                   "⚠️ **剩下的那七成不是「value 记错了」，而是这些行的 `value` 压根不是数值** —— "
                   "icu5000 的 `icu/chartevents.csv` 里有大量**文本型 chart 行**"
                   "（护理评估、约束等级、氧疗方式、活动能力…），`value` 列记的是描述性文本"
                   "（`Normal` / `Medium` / `3 rails up` 这类），而注入器给这些行的 `valuenum` "
                   "注入了数值类错误。\n\n"
                   "⇒ **只有当 `value` 本身能解析成数值、且正好等于 clean_value 时，才能拿它当还原线索**；"
                   "`value` 不是数值的那批格，得另想办法（按 itemid 的量纲规律 ÷100 / 用中位数 / "
                   "按 train 对照反推）。**别无脑套「÷100」或「抄 value」**。")

    if alias:
        alias_txt = ("test 的 modification_log 里 `column=itemid` 的条目共 **%s 条**，"
                     "raw 的 itemid 被改成了 `992{2|3}XXXX` 形式 —— 例如 `220045` → "
                     "`99220045` / `99230045`。实测 **%s 条（%.1f%%）**符合"
                     "「`992` + (2 或 3) + 原 itemid 末 4 位」这条规则"
                     % (fmt_num(alias["n"]), fmt_num(alias["rule_ok"]),
                        100.0 * alias["rule_ok"] / max(1, alias["n"])))
    else:
        alias_txt = "（本档未测得 itemid 别名统计）"

    return ALIAS_SECTION.format(alias=alias_txt) + NATIVE_SECTION.format(
        TBL=tbl,
        EXAMPLES="\n".join(ex_lines) if ex_lines else "（本数据集 train 里没有这类示例）",
        TRAIN_NATIVE=fmt_num(ctx["nat"]["train"]["n"]),
        TRAIN_ITEMS=("、".join(str(i) for i in sorted(ctx["train_items"]))
                     or "（无）"),
        N_TRAIN_EX=str(len(ctx["ex_items"])),
        TRAIN_EX_ITEMS=("、".join(str(i) for i in sorted(ctx["ex_items"]))
                        or "（无）"),
        TRAIN_OTHER_ITEMS=("、".join(str(i) for i in sorted(ctx["train_other_items"]))
                           or "（无）"),
        N_MISSING=str(len(ctx["missing_items"])),
        MISSING_ITEMS=("、".join(str(i) for i in ctx["missing_items"]) or "（无）"),
        VAL_N=fmt_num(ctx["val"]["n"]),
        VAL_ITEMS=("、".join(str(i) for i in ctx["val"]["items"]) or "（无）"),
        VAL_NEW_N=str(len(ctx["val"]["new_items"])),
        VAL_NEW_ITEMS=("、".join(str(i) for i in ctx["val"]["new_items"]) or "（无）"),
        ENUM="\n".join(enum_lines),
        SENTINELS="、".join(str(s) for s in SENTINELS),
        DEV=dev_txt,
        CVP=cvp_txt,
        VEC=vec_txt,
        TEST_NATIVE=fmt_num(ctx["nat"]["test"]["n"]),
    )


ALIAS_SECTION = """### ⚠️ 另外：itemid 列上还有一批 CareVue 别名注入（**key 列**，不改就大面积键对不上）

{alias}

⚠️ 这是 **key 列**上的错误 —— 你不改回去，这些行在按 key 对齐时**全部对不上**，
无论 valuenum 修得多好都拿不到分（改错值比键对不上更难挽回，但键对不上是**直接不参与计分**）。

⇒ **务必先修 key 列（itemid / charttime 这类），再修 valuenum，最后去重。**

"""


NATIVE_SECTION = """## 新增错误类：MIMIC 自身的原生错误（不是注入器造成的）

除了上面那些**注入**错误，train/validation raw 里还有另一类错误：**MIMIC 自己就录错的值** —— 任何人都测不出来的数。它们在数据上长得跟注入错误很像（都是"raw 的值不对、reference 的值对"），但成因不同、**修复方式也不同，别用注入错误那套规则去套**。

本数据集 `icu/chartevents.csv` 里这一类一共两类（**只在 `icu/chartevents.csv` 的 `valuenum` 上**，别的表没有）：

### 类 1：生理不可能区间（孤立尖峰）

下表是本数据集 `icu/chartevents.csv` 里所有带**生理不可能区间**的测量项。
**值落在区间外、且值不等于 0** 的，就是 MIMIC 自己录错的原生错误。

| itemid | 测量项 | 不可能区间 |
|---|---|---|
{TBL}

区间背后的通用规律（写代码时如果想按 label 归类也可以）：收缩压 [40,400]、舒张压 [20,250]、平均压 [20,250]、心率 [20,300]、呼吸频率 [0,100]、体温 [25,45]、中心静脉压 [-20,80]、肺动脉收缩压 [0,150]、肺动脉平均压 [0,100]、PTT [8,200]。

⚠️ **但"越界"本身还不够**。脓毒症 / 低血容量 / 疼痛也能让心率上 150~180、体温降、血压掉 ⇒ **越出"正常范围"不等于录错了**。所以判错要**同时**满足三条：

1. **值在区间外**（或本身是哨兵码）；
2. **与该患者该测量项的常态偏离 ≥ 20%** —— 常态 = 该患者该项其余读数的中位数。真实的病情造成的偏离都在 ±20% 以内。{DEV}。**如果某个越界值与该患者常态只差几个百分点，那多半是病情，不是错误 ⇒ 不要动它。**
3. **该患者该测量项至少还有一个区间内的读数** —— 如果**所有**读数都越界，那是整段平移（换能器校准、量程设置错一类的系统性问题），**真实值不可判定 ⇒ 保持原样不动**。

补充：**同一个坏值可能连续出现多个读数**（比如换传感器后一整段 CVP 都是 300 多）。这不影响修复 —— 因为中位数的池子**只取区间内的读数**，成串的坏值不会污染池子。

### 类 2：MIMIC 哨兵错误码（对**所有** itemid 生效）

MIMIC/CareVue 在"这个读数没取到 / 取错了"时会写入一个占位码。这些码**根本不是测量值**：

`{SENTINELS}`

**这一类对所有 itemid 生效** —— 不限上面那张表的 20 个。检验类项目（Albumin / Platelet / PT / INR / Hemoglobin / Hematocrit / WBC / BUN 等）没有"生理不可能区间"可用，但它们的哨兵码一样是垃圾值。

⚠️ `220000` 和 `220045` 这两个码长得像 itemid，但它们出现在 **`valuenum` 列**里时是错误码，不是 itemid。

### 怎么修：用**该患者自己**该测量项的中位数

修复值 = **该 (subject_id, itemid) 在「最终干净结果」里的可用读数的中位数**（偶数个时取中间两个的平均，和 numpy/pandas 的 median 一致）。

- 类 1 的可用读数 = 落在该 itemid 区间内 且 **不等于 0**
- 类 2 的可用读数 = **不是哨兵码** 且 **不等于 0**

这里"最终干净结果"是指你已经做完所有其它修复之后的那张干净表：cohort 选好了、重复行删掉了、注入错误（量纲放大、哨兵值、**itemid 别名**等）也修回来了。**换句话说：中位数要在 pipeline 的最后一步算**，对你自己已经产出的干净 `icu/chartevents.csv` 再做一遍后处理：

1. 先把前面所有错误修完（含 **itemid / charttime 这类 key 列**），产出干净的 chartevents 表；
2. 在这张干净表上，按上表逐行查：itemid 在表里 + valuenum 落在区间外 + valuenum != 0 → 这是原生错误格；
3. 另查一遍：valuenum 是上面那个哨兵码集合里的值 → 也是原生错误格；
4. 对每个这样的格，用**同一张干净表里**该 (subject_id, itemid) 的可用读数的中位数替换；
5. 同一 (subject_id, itemid) 的多个坏格，修复值是**同一个**中位数。

**为什么强调"最后一步"**：如果你在 cohort 筛选 / 去重 / 注入修复**之前**就去算中位数，池子里会混进 ①别的住院的读数 ②重复行 ③量纲放大/哨兵值弄坏的读数（它们落在区间外，会被当成坏值剔掉，于是本该参与中位数的正常读数少了几个）。这三种污染每一种都会让中位数偏一点，结果就和 reference 对不上。

### train 里能对照出来的例子（**请自己 diff train raw 与 train reference 验证下面的规律**）

train 里这一类错误一共 **{TRAIN_NATIVE} 格**，涉及这些 itemid：**{TRAIN_ITEMS}**。

{EXAMPLES}

⭐⭐⭐ **怎么区分「MIMIC 原生错误」和「注入的量纲放大」—— 看 `value` 文本列**

raw 的 `icu/chartevents.csv` 里有两列：**`value`（文本）** 和 **`valuenum`（数值）**。
本项目实测（同一个任务族 + 本数据集，两组各十几万格）：

| 情况 | raw 的 `value` 文本列 | raw 的 `valuenum` | 怎么修 |
|---|---|---|---|
| **注入的量纲放大**（×100） | 记的是**干净数**（比如 87） | 8700（脏） | 修成 `value` 文本列记的那个数 |
| **MIMIC 原生错误** | 记的是**那个坏数本身**（与 valuenum 一样） | 坏数 | 没有任何线索能还原它 ⇒ **用中位数** |

⇒ **判据：把 raw 同一行的 `value` 文本列和 `valuenum` 比一比**
  · 两者**相等** ⇒ 这一格是 **MIMIC 原生错误**（原始数据里 `value` 记的也是那个坏数，没有第二份记录可对照）⇒ 用中位数修；
  · 两者**不等** ⇒ 这是**注入的量纲放大** ⇒ 修成 `value` 文本列记的那个数。

⚠️ **reference 里没有 `value` 列**（只有 `valuenum` / `valueuom`）⇒ `value` 只能在 **raw** 里看，它是你的**推理线索**，不是你要输出的内容。

⇒ **反过来，对你修注入错误也有用 —— 但要挑着用**：注入类的错误（×100 量纲放大、哨兵码）在 raw 的 `value` 文本列里**有时**记的就是那个正确值。{VEC} ⇒ **当 `value` 能解析成数值、且与 `valuenum` 明显不在一个量级时，优先采用 `value`；`value` 不是数值的那些格（文本型 chart 行）不要套这条，得另想办法。**

⚠️ **不要用"÷100 之后落不落回区间"来判**：{CVP}。

⇒ **判定依据完整清单**（三条全满足才当原生错误）：
  1. **这个值越界**（在区间外，或本身是个哨兵码）；
  2. **`value` 文本列 == `valuenum`**（⇒ 没有任何线索能还原它）；
  3. **与该患者该测量项的常态（其余读数的中位数）偏离 ≥ 20%**，且**该患者该测量项还有区间内的读数**（否则是整段平移，真实值不可判定 ⇒ 不动）。
  满足这三条 ⇒ 用中位数修。

- ⚠️ **不要按"形态"识别原生错误**。上面这些例子只是**数值异常的其中几种形态**：真实的原生错误形态很多样 —— 有的是远高于上界的大数（CVP 常见 250~360），有的是只比区间边界略高/略低（舒张压 10~19、收缩压 30 上下），有的是负值，还有的是哨兵码（999999 这类）。**这些形态里，train 只出现了几种，测试集上还有别的，需要你自己从数据里发现并按同样的原则处理**。
- ⚠️ 还要注意**"先修注入再修原生"的顺序**：如果一个格同时被量纲放大和原生错误两种问题命中，先按注入规则修好，再走中位数填补 —— 否则被注入错误弄坏的读数会污染中位数。

### ⚠️⚠️ 去重必须放在**所有修复之后**（这条会直接影响原生错误格的计分）

本项目在另一个任务族上实测到过一个**很贵的坑**：清洗 pipeline 的顺序是「去重 → 修 charttime → 修 valuenum」，**去重跑在修 charttime 之前**。而修 charttime 会把**原本不同的两行改成同一时间 + 同一 itemid** ⇒ 它们变成新的重复行，但**没有再去重一次**。

后果：在那个任务族上，有 **78 个原生错误格因为"同一 5 键在输出里出现了 2 行"而键对不上**、直接不参与计分（agent 填的值其实是对的，但按 key 对齐取到的是另一行）—— 那 78 格全部是 charttime 类注入被修好之后新产生的重复。

⇒ **请把「去重」放在所有 cell 级修复（含 charttime / itemid 这类 key 列的修复）之后，或者修完 key 列之后再去重一次。** 否则你会把大量本该修对的格变成"键对不上"。

（⚠️ 本数据集的 itemid 别名注入量级是 **百万级** —— 比 charttime 类更大，所以这条在你的 pipeline 上更值钱。）

### 重要：train 里只有 {N_TRAIN_EX} 个 itemid 有可对照的例子（{TRAIN_EX_ITEMS}）

train 分片里只有 **{N_TRAIN_EX} 个测量项（{TRAIN_EX_ITEMS}）有上面那样的对照例子**。

（train raw 里还出现过、但没有可对照例子的测量项：{TRAIN_OTHER_ITEMS}）

**其余 {N_MISSING} 个 itemid（{MISSING_ITEMS}）在 train 里一个原生错误都没有**，但 **validation 和 test 上会出现** —— 所以**要按上表整张实现，不要只写 train 里见过的那几个**。

📌 **validation raw 里也有一批这类错误：共 {VAL_N} 格，覆盖 itemid {VAL_ITEMS}，其中 {VAL_NEW_N} 个（{VAL_NEW_ITEMS}）是 train 里完全没出现过的。** 修好它们是 validation 评分的一部分 —— 这正好是检验你有没有把规则**泛化**到 train 没见过的测量项上的地方。

⚠️ 换句话说：**train 只够让你确认「规则长什么样」，够不够你把它写对，要靠 validation 来验。** 区间表和 R2 修复规则是通用的，不要写成只认 train 那几个 itemid 的特判。

### 注意

- **不要对 0 动手**：MIMIC 里 0 常表示"未测到"或"呼吸机暂停"，不是错误。区间判定里 `!= 0` 是硬条件。
- **不要把 labs 超出 MIMIC 自带参考区间（ref_range_lower / ref_range_upper）的值当错误**：那是临床异常值（INR 1.2、贫血 Hgb 11、Cr 2.0），病人真的有这个值。base excess / 尿电解质 / 粪电解质的负值也是正常的。**这一类错误只在 `icu/chartevents.csv` 的 `valuenum` 上，别的表没有。**
- **如果该患者该测量项一个可用读数都没有**（整段都是坏值 —— 这通常是换能器校准一类的整体平移，真实值不可判定）：**保持原样不动**，不要用全局统计量硬凑。
- **不要扩大到所有离群点**：只处理"落在上表区间外"或"是哨兵码"的孤立尖峰。同一患者某项读数整体偏高或偏低是病情，不是错误。
- ⚠️ **枚举型项目同一时刻出现多条不同取值是合法的，不是重复行**：

{ENUM}

  这类项目如果按 `(subject_id, hadm_id, stay_id, charttime, itemid)` 分组会出现多个不同取值，**不要当成重复行删掉** —— 删了就是丢行，key 对不上，比改错值更难挽回。
- train 里这类错误只有几十格、validation 里几十格，**但测试集规模大 1,000 倍以上**（测试集的 (subject_id, itemid) 组合数是 train 的几十倍）。请把规则实现成**逐 (subject_id, itemid) 计算**的通用逻辑，不要写成只认上面这几个 itemid 的特判，也不要因为 train 里只出现过几个 itemid 就缩小实现范围。

（本次测试集上登记在册的 MIMIC 原生错误共 **{TEST_NATIVE} 格**，按上面三条规则逐格处理。）
"""


def replace_magnitude_lines(base: str, logs: dict, total_cells: dict) -> str:
    """把「错误注入量级」那三行的数字换成 mimicraw 档的实测值。

    原行形如：
      - train: 约 14,966 cell_update + 1,972 duplicate_row + 1,090 label_leakage 插入行 = 18,028 个错误分布在 10 个 stay 上（cell-level 1.30%，97,950 行）

    cell-level % = cell_update 条数 / total_cells（total_cells 取自权威 manifest），
    随 mimicraw 档新增的原生错误条目一起重算。
    """
    pat = re.compile(r"^- (train|validation|最终测试集): .*$", re.M)
    tail_re = re.compile(
        r"个错误分布在\s*(.*?)\s*个 stay 上（cell-level\s*([\d.]+)%，([\d,]+)\s*行）")

    def repl(m):
        key = m.group(1)
        split = {"train": "train", "validation": "validation",
                 "最终测试集": "test"}[key]
        st = logs.get(split)
        if not st:
            return m.group(0)
        op = st["op"]
        parts = []
        for o, label in (("cell_update", "cell_update"),
                         ("duplicate", "duplicate_row"),
                         ("insert", "label_leakage 插入行")):
            if op.get(o):
                parts.append("%s %s" % (fmt_num(op[o]), label))
        total = st["n"]
        line = "- %s: 约 %s = %s 个错误" % (key, " + ".join(parts), fmt_num(total))
        tail = tail_re.search(m.group(0))
        if tail:
            n_stay, _pct, rows = tail.group(1), tail.group(2), tail.group(3)
            tc = total_cells.get(split)
            pct = "%.2f" % (100.0 * op.get("cell_update", 0) / tc) if tc else _pct
            line += ("分布在 %s 个 stay 上（cell-level %s%%，%s 行）"
                     % (n_stay, pct, rows))
        return line

    return pat.sub(repl, base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base-prompt", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--value-stats", default=None,
                    help="icu5000_value_reliability.py 的输出 JSON（value 文本列的"
                         "可靠性分类；不给就退回 stats 里的粗比例）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ds = Path(args.dataset)
    stats = json.loads(Path(args.stats).read_text(encoding="utf-8"))

    logs = {s: load_log_stats(ds, s) for s in ("train", "validation", "test")}
    nat = {s: native_stats(ds, s) for s in ("train", "validation", "test")}
    examples, train_items, _ = train_examples(ds)
    val = val_example_stats(ds, train_items)

    ex_items = sorted({d["itemid"] for d in examples})
    # ⚠️ stats JSON 里 train_hard_items 的键是**字符串**（从 CSV 读的），
    #    而 itemid 集合一律是 int ⇒ 不转 int 的话：
    #      ① train_hard - ex_items 恒等于全部（字符串 ∩ 整数 = ∅）
    #      ② `i not in train_hard` 恒为真 ⇒ missing 变成全部 20 个
    #    ⇒ prompt 会同时说「X 有对照例子」和「X 在 train 里一个原生错误都没有」。
    train_hard = {int(k) for k in stats.get("train_hard_items", {})}
    train_other = sorted(train_hard - set(ex_items))
    missing = [i for i, _, _, _ in HARD_RANGE if i not in train_hard]

    ctx = {
        "logs": logs, "nat": nat, "examples": examples,
        "train_items": train_items, "ex_items": ex_items,
        "train_other_items": train_other, "missing_items": missing,
        "val": val,
        "dev": stats.get("deviation"),
        "cvp": (stats.get("deviation") or {}).get("cvp"),
        "alias": stats.get("alias"),
        "vec": stats.get("value_eq_clean"),
        "value_reliability": (
            json.loads(Path(args.value_stats).read_text(encoding="utf-8"))
            if args.value_stats else None),
        "enum": stats.get("enum") or {"rows": {}, "multi_groups": {}},
    }

    base = Path(args.base_prompt).read_text(encoding="utf-8")
    # total_cells 取自权威 manifest（prepare 步骤①先生成）—— 没有就用原行里的 %
    total_cells = {}
    mf = ds / "v2_rawerror_icu5000_final_report.json"
    if mf.is_file():
        md = json.loads(mf.read_text(encoding="utf-8"))
        for s, v in md.get("splits", {}).items():
            if v.get("total_cells"):
                total_cells[s] = v["total_cells"]
    base = replace_magnitude_lines(base, logs, total_cells)

    if ANCHOR not in base:
        raise SystemExit("FATAL: 基础 prompt 里找不到锚点 %r" % ANCHOR)
    section = build_section(ctx)
    base = base.replace(ANCHOR, section + "\n" + ANCHOR, 1)

    Path(args.out).write_text(base, encoding="utf-8")
    print("已写出:", args.out, "(%d bytes)" % len(base.encode("utf-8")))
    print("  native: train=%s validation=%s test=%s"
          % (nat["train"]["n"] if nat["train"] else 0,
             nat["validation"]["n"] if nat["validation"] else 0,
             nat["test"]["n"] if nat["test"] else 0))
    print("  log   : train=%s validation=%s test=%s"
          % (logs["train"]["n"] if logs["train"] else 0,
             logs["validation"]["n"] if logs["validation"] else 0,
             logs["test"]["n"] if logs["test"] else 0))


if __name__ == "__main__":
    main()
