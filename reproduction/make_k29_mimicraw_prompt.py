#!/usr/bin/env python3
"""make_k29_mimicraw_prompt.py —— 从构建好的 mimicrawerror 数据集生成 agent 的 prompt。

prompt 里**只用聚合统计**（每 itemid 的格数、值的范围），不出现任何
subject_id / hadm_id / 病人级取值 —— 这样 prompt 可以安全地进 git。

用法：
  python3 make_k29_mimicraw_prompt.py --dataset <数据集目录> --out <prompt.md>
"""
from __future__ import annotations

import argparse
import csv
import json
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
SENTINELS = [-999, -998, -997, -990, 9999, 99999, 999999, 9999999, 220000, 220045]

# 枚举型 itemid：同一时刻多条不同取值是合法的，不是重复行
ENUM_ITEMS = [(230033, "Stool Management（排泄管理）"),
              (229864, "Pressure Injury Location（压疮部位）")]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_log_stats(ds, split):
    p = ds / split / "modification_log.csv"
    if not p.is_file():
        return None
    sub = Counter(); op = Counter(); n = 0
    with p.open(newline="", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            n += 1; sub[e.get("error_subtype", "")] += 1; op[e.get("operation", "")] += 1
    return {"n": n, "sub": sub, "op": op}


def native_stats(ds, split):
    p = ds / split / "mimic_native_error_log.csv"
    if not p.is_file():
        return None
    rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
    per_item = Counter()
    per_mode = Counter()
    inj = Counter()
    for r in rows:
        per_item[r["itemid"]] += 1
        per_mode[r["detection_mode"]] += 1
        inj[r["also_had_injection"]] += 1
    return {"n": len(rows), "per_item": per_item, "per_mode": per_mode, "inj": inj}


def train_examples(ds):
    """train 里注入的原生错误示例 —— 按 itemid 聚合成可安全公开的描述。
    返回 (每 itemid 的描述 list, train 里出现过的 HARD itemid 集合, 枚举型统计)
    """
    nat = ds / "train" / "mimic_native_error_log.csv"
    raw = ds / "train" / "raw" / CE
    if not nat.is_file():
        return [], set(), {}
    rows = list(csv.DictReader(nat.open(newline="", encoding="utf-8")))

    # train raw 里出现过的 HARD itemid
    raw_items = Counter()
    for r in csv.DictReader(raw.open(newline="", encoding="utf-8")):
        try:
            it = int(r["itemid"])
        except ValueError:
            continue
        if it in HARD:
            raw_items[it] += 1

    desc = []
    per_item = defaultdict(list)
    for r in rows:
        per_item[r["itemid"]].append(r)
    for it, rs in sorted(per_item.items(), key=lambda x: -len(x[1])):
        it = int(it)
        vals = [fnum(r["mimic_native_value"]) for r in rs]
        reps = [fnum(r["repaired_value"]) for r in rs]
        desc.append({
            "itemid": it,
            "label": dict((i, l) for i, l, _, _ in HARD_RANGE).get(it, ""),
            "n": len(rs),
            "dirty_range": (min(v for v in vals if v is not None),
                            max(v for v in vals if v is not None)),
            "repaired": sorted(set(reps)),
            "n_peer": sorted(set(int(r["n_peer_readings"]) for r in rs)),
        })
    return desc, set(raw_items), raw_items


def val_example_stats(ds):
    """validation 里注入的原生错误示例的聚合统计（从 validation raw 可观测，不算泄漏）"""
    nat = ds / "validation" / "mimic_native_error_log.csv"
    if not nat.is_file():
        return {"n": 0, "items": [], "new_items": []}
    rows = list(csv.DictReader(nat.open(newline="", encoding="utf-8")))
    items = sorted({int(r["itemid"]) for r in rows})
    _, train_items, _ = train_examples(ds)
    return {"n": len(rows), "items": items,
            "new_items": [i for i in items if i not in train_items]}


def enum_stats(ds, split="test"):
    p = ds / split / "reference" / CE
    if not p.is_file():
        return {}
    want = {str(i) for i, _ in ENUM_ITEMS}
    grp = defaultdict(set)
    cnt = Counter()
    for r in csv.DictReader(p.open(newline="", encoding="utf-8")):
        it = r["itemid"]
        if it not in want:
            continue
        cnt[it] += 1
        grp[(r["subject_id"], r["hadm_id"], r["stay_id"], r["charttime"], it)].add(r["valuenum"])
    multi = Counter()
    for k, v in grp.items():
        if len(v) > 1:
            multi[k[4]] += 1
    return {"rows": cnt, "multi": multi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()
    ds = Path(args.dataset)

    logs = {s: load_log_stats(ds, s) for s in ("train", "validation", "test")}
    nat = {s: native_stats(ds, s) for s in ("train", "validation", "test")}
    desc, train_items, raw_items = train_examples(ds)
    enum = enum_stats(ds, "test")

    hdr = ["itemid", "测量项", "不可能区间"]
    rows_tab = []
    for i, lab, lo, hi in HARD_RANGE:
        rows_tab.append("| %d | %s | [%d, %d] |" % (i, lab, lo, hi))
    tbl = "\n".join(rows_tab)

    ex_lines = []
    for d in desc:
        lo, hi = d["dirty_range"]
        reps = "、".join("%g" % x for x in d["repaired"][:3])
        ex_lines.append(
            "- **itemid %d（%s）**：train raw 里该测量项有 **%d 格**是孤立尖峰，"
            "取值 %g ~ %g；train reference 把它们改成了 **%s**（= 该患者该测量项其余读数的中位数）。"
            % (d["itemid"], d["label"], d["n"], lo, hi, reps))

    appear = sorted(train_items)
    missing = [i for i, _, _, _ in HARD_RANGE if i not in train_items]

    enum_lines = []
    for i, lab in ENUM_ITEMS:
        s = str(i)
        enum_lines.append("- **itemid %d（%s）**：test reference 里 %d 行，其中 **%d 组**是"
                          "「同一 (subject_id, hadm_id, stay_id, charttime, itemid) 出现多条不同取值」"
                          "—— 这是**合法的多值记录**（多个部位 / 多个选项），**不是重复行**，不要去重删除。"
                          % (i, lab, enum["rows"].get(s, 0), enum["multi"].get(s, 0)))

    vstat = val_example_stats(ds)
    ctx = {
        "TBL": tbl,
        "EXAMPLES": "\n".join(ex_lines) if ex_lines else "（本数据集 train 里没有这类示例）",
        "TRAIN_ITEMS": "、".join(str(i) for i in appear) if appear else "（无）",
        "MISSING_ITEMS": "、".join(str(i) for i in missing),
        "N_MISSING": str(len(missing)),
        "ENUM": "\n".join(enum_lines),
        "SENTINELS": "、".join(str(s) for s in SENTINELS),
        "VAL_N": str(vstat["n"]),
        "VAL_ITEMS": "、".join(str(i) for i in vstat["items"]),
        "VAL_NEW_ITEMS": "、".join(str(i) for i in vstat["new_items"]) or "（无）",
        "VAL_NEW_N": str(len(vstat["new_items"])),
        "TRAIN_ERR": "{:,}".format(logs["train"]["n"]) if logs["train"] else "n/a",
        "VAL_ERR": "{:,}".format(logs["validation"]["n"]) if logs["validation"] else "n/a",
        "TEST_ERR": "{:,}".format(logs["test"]["n"]) if logs["test"] else "n/a",
        "TRAIN_SUB": "、".join("%s %s" % (k, v) for k, v in
                               (logs["train"]["sub"].most_common(8) if logs["train"] else [])),
        "VAL_SUB": "、".join("%s %s" % (k, v) for k, v in
                             (logs["validation"]["sub"].most_common(8) if logs["validation"] else [])),
        "TRAIN_NATIVE": str(nat["train"]["n"]) if nat["train"] else "0",
        "VAL_NATIVE": str(nat["validation"]["n"]) if nat["validation"] else "0",
        "TEST_NATIVE": str(nat["test"]["n"]) if nat["test"] else "0",
        "TEST_NATIVE_CAT": "、".join("%s %d" % (k, v) for k, v in
                                     (nat["test"]["per_mode"].most_common() if nat["test"] else [])),
        "TRAIN_OP": "、".join("%s %s" % (k, v) for k, v in
                              (logs["train"]["op"].most_common() if logs["train"] else [])),
        "VAL_OP": "、".join("%s %s" % (k, v) for k, v in
                            (logs["validation"]["op"].most_common() if logs["validation"] else [])),
    }
    print(json.dumps({k: v for k, v in ctx.items()
                      if k in ("TRAIN_NATIVE", "VAL_NATIVE", "TEST_NATIVE",
                               "TEST_NATIVE_CAT", "TRAIN_ITEMS", "N_MISSING")},
                     ensure_ascii=False, indent=1))
    Path(args.out).write_text(TEMPLATE.format(**ctx), encoding="utf-8")
    print("已写出:", args.out)


TEMPLATE = """你是 Data Cleaning Agent。请完成一个基于少量标准示例的 MIMIC-IV 数据清洗任务，最终输出一个名为 K29 的疾病 cohort 结果包（K29 = ICD-10 胃炎和十二指肠炎章节，对应 `K29.0` ~ `K29.9`），围绕该 cohort 做按住院（hadm_id）切片与事件级清洗。

本回合末尾的 `[Validation Harness public contract]` 会给出 train raw / train reference / validation raw / agent workdir 的**真实绝对路径**。请直接使用那里给出的路径，不要去找 `TRAIN_RAW` 之类的占位符字符串。**测试集 raw 在本阶段不提供，也不需要你去找**——你交付的是一份**可重放**的 pipeline；host 会把你最好的一版冻结下来，用 `{{raw_root}}` 指向测试集 raw 跑一次独立重放。

## 工作流契约（务必遵守）

- **每一回合结束时，agent workdir 里必须已经躺着一份完整的结果包**：12 个 csv + `submission.json`。host 会在你的回合结束后**立刻读取并打分**，再把聚合分数与逐文件分数反馈给你。回合结束时没有结果包，这一回合就是 0 分。
- harness 最多给你若干回合，每回合都计分。如果某一回合分数低于之前的最好版本，host 可能把你的 workdir **回滚**到那份最好的文件——所以有更好的产出就立刻写盘，不要只留在推理过程里。
- 每回合的反馈是你的主要信息来源：它会告诉你哪些文件分数低、哪些行对不上、哪些列错了。照反馈改规则、重新跑一遍、再交付。

**重要**：validation 的标准答案（hidden gold）对你不可见，host 会把它的路径 redact 成 `<hidden_gold>`。**不要试图读取 pi_test_scorer.py / evaluation_manifest.json 之类的评分元数据，也不要去猜测测试集标准答案。**

train raw、validation raw 都是按 hadm_id 切片的（不是 subject_id），`keys.csv` 的列是 `hadm_id`。

本轮的核心不是只生成同名文件，而是从 train 示例推导**可重放的**转换规则。开始时请直接读取 contract 里给出的路径。可以枚举它们共同的父目录，但不得读取任何其他数据集或实验目录。

## K29 任务定义

**cohort 选择**：cohort.csv 列出每个入选 hadm 的 K29 诊断信息（`cohort_type` / `cohort_label` / `icd_root` 等列），hospital_expire_flag 作为 in-hospital mortality 标签。

这里有两个必须自己踩出来的陷阱：

1. `diagnoses_icd` 同时含 ICD-9（`icd_version=9`）与 ICD-10（`icd_version=10`）。K29 是 ICD-10 章节码，但本数据集里**相当一部分 hadm 只有 ICD-9 码**。若只按 `LIKE 'K29%'` 过滤会整批漏掉它们。请自己从 train 对照反推 ICD-9 的等价前缀——对照里 cohort 的 hadm 行数会直接告诉你漏了几个。
2. cohort 判定必须在**修完 raw 的诊断码错误之后**做。本数据集的 `hosp/diagnoses_icd.csv` 的 `icd_code` 列被注入了错误，直接拿它去过滤会得到错误的 cohort。

**必须输出的 12 个 csv 文件**：
- 7 个顶层：`admissions.csv`, `cohort.csv`, `demographics.csv`, `diagnoses.csv`, `procedures.csv`, `medications.csv`, `labs.csv`
- 4 个 ICU：`icu/icustays.csv`, `icu/chartevents.csv`, `icu/inputevents.csv`, `icu/outputevents.csv`
- 1 个文本：`multimodal/notes.csv`

## 数据特点（v2 rawerror 关键区别）

train raw、validation raw 都是**注入了错误的 MIMIC-IV 原始数据**：hosp/ 和 icu/ 子目录里**包含多种类型的 cell-level 错误**（值越界、enum 违例、时间序颠倒、跨表不一致等）；note/ 等多模态原始文件是干净原样数据。你的任务是**先修正这些错误**（让 raw 变成可处理的干净 MIMIC），再生成格式化结果包。

具体有哪些错误子类型、有多少错误、如何修正——**这些都需要你通过对照 train raw 与 train reference 自己推导**。

错误注入量级（供你估算工作量，不是硬约束）：
- train: {TRAIN_ERR} 个错误分布在 10 个 hadm 上（{TRAIN_OP}）
- validation: {VAL_ERR} 个错误分布在 20 个 hadm 上（{VAL_OP}）
- 最终测试集: {TEST_ERR} 个错误分布在 8,092 个 hadm 上

**各阶段的规模差 1,000 倍以上，数据量差 140 倍以上**（validation 只有 20 个 hadm，最终测试集有 8,092 个 hadm）：

- 不要用 train 或 validation 的**错误绝对数量**去推断测试集该有多少行需要处理或丢弃。
- 从 train 得到的应该是**每类错误的判定规则**，而不是"这张表大概该丢多少比例的行"。
- **每一类错误的占比在不同阶段之间会剧烈变化**：某一类在 train 里只占个位数百分比，到测试集可能变成绝对主力。所以不要因为某类错误在 train 里"只出现几次"就判定它不重要。
- 反过来，**某个错误模式在 train 和 validation 里都没出现，不代表测试集里没有**。你可以（也应该）从数据本身推断出 train/validation 未覆盖的错误模式，但每一条都必须能用领域常识说明为什么它是错的——不要凭空编造。
- **并非所有表都有注入错误**。raw 下有 28 个表，但错误高度集中在少数几张（chartevents、输入/输出事件、诊断、patients、admissions、icustays 等）。请自己逐表比对确认哪些表是干净的，不要对每个表都套一遍修正逻辑。
- 在 validation 上检验你的规则时，请特别关注：**有没有把不该丢的行丢掉**。丢行会让 key 对不上，比改错值更难挽回。

## 新增错误类：MIMIC 自身的原生错误（不是注入器造成的）

除了上面那些**注入**错误，train/validation raw 里还有另一类错误：**MIMIC 自己就录错的值** —— 任何人都测不出来的数。它们在数据上长得跟注入错误很像（都是"raw 的值不对、reference 的值对"），但成因不同、**修复方式也不同，别用注入错误那套规则去套**。

本数据集 `icu/chartevents.csv` 里这一类一共两类：

### 类 1：生理不可能区间（孤立尖峰）

下表是本数据集 `icu/chartevents.csv` 里所有带**生理不可能区间**的测量项。
**值落在区间外、且值不等于 0** 的，就是 MIMIC 自己录错的原生错误。

| itemid | 测量项 | 不可能区间 |
|---|---|---|
{TBL}

区间背后的通用规律（写代码时如果想按 label 归类也可以）：收缩压 [40,400]、舒张压 [20,250]、平均压 [20,250]、心率 [20,300]、呼吸频率 [0,100]、体温 [25,45]、中心静脉压 [-20,80]、肺动脉收缩压 [0,150]、肺动脉平均压 [0,100]、PTT [8,200]。

⚠️ **但"越界"本身还不够**。脓毒症 / 低血容量 / 疼痛也能让心率上 150~180、体温降、血压掉 ⇒ **越出"正常范围"不等于录错了**。所以判错要**同时**满足三条：

1. **值在区间外**（或本身是哨兵码）；
2. **与该患者该测量项的常态偏离 ≥ 20%** —— 常态 = 该患者该项其余读数的中位数。真实的病情造成的偏离都在 ±20% 以内（本项目在另一个任务族上实测：1249 个同类格里，**与自身常态相差不到 ±20% 的格数是 0**；而这些错误格的偏离中位数是 4.3 倍）。**如果某个越界值与该患者常态只差几个百分点，那多半是病情，不是错误 ⇒ 不要动它。**
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

这里"最终干净结果"是指你已经做完所有其它修复之后的那张干净表：cohort 选好了、重复行删掉了、注入错误（量纲放大、哨兵值等）也修回来了。**换句话说：中位数要在 pipeline 的最后一步算**，对你自己已经产出的干净 `icu/chartevents.csv` 再做一遍后处理：

1. 先把前面所有错误修完，产出干净的 chartevents 表；
2. 在这张干净表上，按上表逐行查：itemid 在表里 + valuenum 落在区间外 + valuenum != 0 → 这是原生错误格；
3. 另查一遍：valuenum 是上面那个哨兵码集合里的值 → 也是原生错误格；
4. 对每个这样的格，用**同一张干净表里**该 (subject_id, itemid) 的可用读数的中位数替换；
5. 同一 (subject_id, itemid) 的多个坏格，修复值是**同一个**中位数。

**为什么强调"最后一步"**：如果你在 cohort 筛选 / 去重 / 注入修复**之前**就去算中位数，池子里会混进 ①别的住院的读数 ②重复行 ③量纲放大/哨兵值弄坏的读数（它们落在区间外，会被当成坏值剔掉，于是本该参与中位数的正常读数少了几个）。这三种污染每一种都会让中位数偏一点，结果就和 reference 对不上。

### train 里能对照出来的例子（**请自己 diff train raw 与 train reference 验证下面的规律**）

train 里这一类错误一共 **{TRAIN_NATIVE} 格**，涉及这些 itemid：**{TRAIN_ITEMS}**。

{EXAMPLES}

⭐ **这些例子的共同规律，请务必在 train 对照里自己确认一遍**：

- 这些坏值的形态和注入器的**量纲放大错误**（×100）**很容易混淆** —— raw 里都是"一个明显不合理的数"。
- **唯一的可靠区别在 reference 里**：
  · 注入的量纲放大 ⇒ reference 里是「÷100 之后的正常值」（即那格本来就该是这个数，只是被乘了 100）；
  · **MIMIC 原生错误 ⇒ reference 里是「该患者该测量项其余读数的中位数」**（那格的值 MIMIC 自己就录错了，没有任何线索能还原它）。
- ⇒ 所以**不要用"÷100 之后落不落回区间"来判**（实测这条判据在中心静脉压上会 245/245 全错：CVP 的原生错值 277~361 除以 100 是 2.77~3.61，落在 CVP 的 [-20,80] 里，看着完全合理）。
- ⇒ **判定依据只有三条**：
  1. **这个值越界**（在区间外，或本身是个哨兵码）；
  2. **与该患者该测量项的常态（其余读数的中位数）偏离 ≥ 20%**；
  3. **该患者该测量项还有区间内的读数**（否则是整段平移，真实值不可判定 ⇒ 不动）。
  满足这三条 ⇒ 用中位数修。
- ⚠️ **不要按"形态"识别原生错误**。上面这些例子只是**数值异常的其中几种形态**：真实的原生错误形态很多样 —— 有的是远高于上界的大数（CVP 常见 250~360），有的是只比区间边界略高/略低（舒张压 10~19、收缩压 30 上下），有的是负值，还有的是哨兵码（999999 这类）。**这些形态里，train 只出现了几种，测试集上还有别的，需要你自己从数据里发现并按同样的原则处理**（越界 + 同组其余读数正常 ⇒ 中位数）。
- ⚠️ 还要注意**"先修注入再修原生"的顺序**：如果一个格同时被量纲放大和原生错误两种问题命中，先按注入规则修好，再走中位数填补 —— 否则被注入错误弄坏的读数会污染中位数。

### ⚠️⚠️ 去重必须放在**所有修复之后**（这条会直接影响原生错误格的计分）

本项目在另一个任务族上实测到过一个**很贵的坑**：清洗 pipeline 的顺序是「去重 → 修 charttime → 修 valuenum」，**去重跑在修 charttime 之前**。而修 charttime 会把**原本不同的两行改成同一时间 + 同一 itemid** ⇒ 它们变成新的重复行，但**没有再去重一次**。

后果：在那个任务族上，有 **78 个原生错误格因为"同一 5 键在输出里出现了 2 行"而键对不上**、直接不参与计分（agent 填的值其实是对的，但按 key 对齐取到的是另一行）—— 那 78 格全部是 charttime 类注入被修好之后新产生的重复。

⇒ **请把「去重」放在所有 cell 级修复（含 charttime / itemid 这类 key 列的修复）之后，或者修完 key 列之后再去重一次。** 否则你会把大量本该修对的格变成"键对不上"。

### 重要：train 里只出现过 {TRAIN_ITEMS} 这几个 itemid

train 分片里只有 **{TRAIN_ITEMS}** 这几个测量项出现过原生错误 —— train 只有 10 个 hadm、2,000 行 chartevents，全 train 的 (subject_id, itemid) 组合里够条件的只有个位数，**样本天然稀少，这是数据规模决定的，不是这类错误不重要**。

**其余 {N_MISSING} 个 itemid（{MISSING_ITEMS}）在 train 里一个原生错误都没有**，但 **validation 和 test 上会出现** —— 所以**要按上表整张实现，不要只写 train 里见过的那几个**。

📌 **validation raw 里也有一批这类错误：共 {VAL_N} 格，覆盖 itemid {VAL_ITEMS}，其中 {VAL_NEW_N} 个（{VAL_NEW_ITEMS}）是 train 里完全没出现过的。** 修好它们是 validation 评分的一部分 —— 这正好是检验你有没有把规则**泛化**到 train 没见过的测量项上的地方（train 只有 5 个 itemid 有样本，其余全靠上面的区间表和规则）。

⚠️ 换句话说：**train 只够让你确认「规则长什么样」，够不够你把它写对，要靠 validation 来验。** 区间表和 R2 修复规则是通用的，不要写成只认 train 那几个 itemid 的特判。

### 注意

- **不要对 0 动手**：MIMIC 里 0 常表示"未测到"或"呼吸机暂停"，不是错误。区间判定里 `!= 0` 是硬条件。
- **不要把 labs 超出 MIMIC 自带参考区间（ref_range_lower / ref_range_upper）的值当错误**：那是临床异常值（INR 1.2、贫血 Hgb 11、Cr 2.0），病人真的有这个值。base excess / 尿电解质 / 粪电解质的负值也是正常的。**这一类错误只在 `icu/chartevents.csv` 的 `valuenum` 上，别的表没有。**
- **如果该患者该测量项一个可用读数都没有**（整段都是坏值 —— 这通常是换能器校准一类的整体平移，真实值不可判定）：**保持原样不动**，不要用全局统计量硬凑。
- **不要扩大到所有离群点**：只处理"落在上表区间外"或"是哨兵码"的孤立尖峰。同一患者某项读数整体偏高或偏低是病情，不是错误。
- ⚠️ **枚举型项目同一时刻出现多条不同取值是合法的，不是重复行**：

{ENUM}

  这类项目如果按 `(subject_id, hadm_id, stay_id, charttime, itemid)` 分组会出现多个不同取值，**不要当成重复行删掉** —— 删了就是丢行，key 对不上，比改错值更难挽回。
- train 里这类错误只有几格、validation 里几十格，**但测试集规模大 1,000 倍以上**（测试集的 (subject_id, itemid) 组合数是 train 的几十倍）。请把规则实现成**逐 (subject_id, itemid) 计算**的通用逻辑，不要写成只认上面这几个 itemid 的特判，也不要因为 train 里只出现过几个 itemid 就缩小实现范围。

### 推导与修正流程

**推荐流程**（具体细节由你按 train 对照决定）：

1. **加载 raw** → 解析所有 hosp/icu/*.csv（**note/ 不需要修**，原样处理）
2. **对照 train raw 与 train reference** → 系统性比对每个文件的每行每列：
   - 哪些 cell 值与 reference 不一致？
   - 不一致是否构成有意义的"错误"（不是 schema 差异，不是过滤差异，不是聚合差异）？
   - 这些错误能否归类为已知模式（值越界 / 枚举违例 / 时间序颠倒 / 哨兵值 / 跨表不一致 ...）？
   - **哪些行/哪些 hadm 被选进了 cohort？被选的和没被选的，在数据上有什么系统性的差别？**
3. **建立基线表**（如 valid gender/age 范围、icustays 三元组等）
4. **对每个发现的错误模式写检测 + 修复逻辑**（按你的实现风格；建议先小后大、先 train 自检通过再处理 validation）
5. **在 validation 上检验**：跑一遍同一 pipeline，用 host 每回合给的逐文件分数找出 train 上没覆盖到的模式，把它们补进规则
6. **产出与 train reference 结构一致的 12 个 csv + `submission.json`**

**重要约束**：
- 不要硬编码任何 subject_id / hadm_id / stay_id / note_id / 行号或标准答案
- 不要用重复行、截断行、填充空值的方式拟合任何数据集的规模
- 不能读取 scoring 元数据（pi_test_scorer.py / evaluation_manifest.json 之类）
- 如果发现无法判定的错误（同一 cell 命中多个子类），优先按 reference 同一 (stay_id, itemid, charttime) 行反推正确值
- **replay 契约**：harness 重放时只注入 `{{raw_root}}`、`{{train_reference}}`、`{{output_dir}}` 和可选 `{{workdir}}` 四个参数。`{{raw_root}}` 在训练时指向 train raw、在验证时指向 validation raw、在最终测试时指向测试集 raw。**不得**使用 `{{train_raw}}` 或 `{{validation_raw}}`——重放时它们根本不存在。

必须完成的工作：

1. 比较 train raw 与 train reference 的目录、文件、schema、主键、实体关系、时间字段、行级变化和聚合结果。
2. 在 agent workdir 写出 `train_analysis.json`。对 12 个目标 csv 文件记录：对应 raw 输入表、主键、时间锚点、过滤条件、字段转换、聚合规则，**以及你从 train 对照中推导出来的错误模式清单**（每条至少配一项 train 对照证据）。
3. 基于这些规则实现**可重放**的 pipeline。它必须从 raw/hosp、raw/icu、raw/note 开始，不能把已有结果包当作输入。
4. 在 train raw 上实际运行该 pipeline（**包括错误修正步骤**），与 train reference 比较，记录可解释的 train 自检结果（**train 自检必须通过：与 reference 一致**）。
5. 在 validation raw 上运行同一 pipeline，把每回合 host 给的逐文件分数当作反馈，把 validation 暴露的新模式补进规则。
6. 在 validation raw 上产出完整的 12 个 csv 结果包和 `submission.json`，并确认 replay 能从干净状态复现出同样的包。

输出与复现要求：

- 保持 train reference 的相对目录结构、12 个 csv 文件名、schema、数据类型、主键语义和聚合语义
- summary.json 不参与行对齐评分，但若 train reference 里有它则建议保留同位置同 schema 的输出（不强制）
- 不得硬编码任何 hadm_id、subject_id、stay_id、note_id、行号或标准答案
- 不得用重复行、截断行或填充空值的方式仅为了匹配数据集规模
- 保留分析脚本、最终 pipeline、规则/映射、运行入口和 `train_analysis.json`
- `submission.json` 必须包含 `schema_version=1`、相对 `result_root`，以及 JSON 参数数组形式的 `replay.argv`
- **replay 契约**：只依赖 `{{raw_root}}`、`{{train_reference}}`、`{{output_dir}}` 和可选 `{{workdir}}`；**不得**使用 `{{train_raw}}` 或 `{{validation_raw}}`
- 请在交付前自己至少做一次完整的 train 自检，并在干净目录里验证一次 replay 能跑通

限制：

- 不使用网络
- 不读取、搜索或推断测试集标准答案、其他实验目录、历史 pipeline、其他工作树或项目外数据（也不读取 pi_test_scorer.py / evaluation_manifest.json 之类的评分元数据）
- 已注册的 Skills 可按需读取，但它们不包含隐藏数据或隐藏答案
- 证据不足时采用保守、可解释、由 train 对照支持的规则；先保证训练对照、train 自检通过，再追求 validation 正确性
"""


if __name__ == "__main__":
    main()
