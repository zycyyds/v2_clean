#!/usr/bin/env python3
"""make_n18_mimicraw_prompt.py —— 由 N18 的基线 val prompt 生成 **mimicraw 版**。

基线（N18 v1）原本**没有**「MIMIC 原生错误」这一节 —— N18 母数据集只有注入层。
本脚本把原生错误层（T1 哨兵占位码 + T2 生理不可能区间）的**判定规则**、
**itemid↔区间对照表**、**train 里能对照出来的例子**、**itemid 别名注入节**、
以及**重算的量级三行 + 各 split 原生层规模**写进去。

⚠️ 全部数字都从**产物**实测（manifest / native log / modification_log），
   不写死「100% 符合」这种没验证的话。

用法:
  python3 make_n18_mimicraw_prompt.py --dataset <mimicraw 数据集> \
      --base-prompt <n18_1.6pct_val_prompt_v1.md> --out <输出>
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(2 ** 31)

HERE = Path(__file__).resolve().parent
BUILDER_PATH = HERE / "build_n18_mimic_raw_error.py"

# ── 锚点（必须与基线 prompt 原文完全一致，否则报错）──
A_MAGNITUDE = ("- train: 约 ", "- validation: 约 ", "- 最终测试集: 约 ")
A_FLOW = "### 推导与修正流程"
A_CONSTRAINT = "**重要约束**："
# ⚠️ result_root 的措辞缺口（1.6pct 第 1 轮实测踩到）：基线只说「相对 result_root」，
#    agent 就写了 "."（= workdir 本身）⇒ harness 判「不是独立子目录」⇒ 本轮 0 分。
#    ��轮 1 白跑了一整个回合 ⇒ 措辞必须写死。
A_SUBMIT = ("- `submission.json` 必须包含 `schema_version=1`、相对 `result_root`，"
            "以及 JSON 参数数组形式的 `replay.argv`")
A_SUBMIT_NEW = """- `submission.json` 必须包含 `schema_version=1`、相对 `result_root`，以及 JSON 参数数组形式的 `replay.argv`
  - ⚠️ **`result_root` 必须是一个**「专门的结果子目录」（例如 `result/`）——
    **不能是 `"."`、不能是 workdir 本身、也不能是 workdir 的父目录**。
    harness 会把它当成结果包的根去遍历；写成 `"."` 会被判成
    `submission result_root must be a dedicated subdirectory` ⇒ **本轮直接 0 分**。
  - 换句话说：**18 个 csv 要写进那个子目录里**（不要直接摊在 workdir 根下），
    `submission.json` 里写 `result_root` = 那个子目录的**相对路径**。"""

SPLIT_NAMES = {"train": "train", "validation": "validation", "test": "最终测试集"}


def fmt(n):
    return "{:,}".format(int(n))


def load_builder():
    spec = importlib.util.spec_from_file_location("n18builder", BUILDER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hard_table(builder):
    """d_items 里命中硬区间的 itemid → (label, (lo, hi))（按 itemid 排序）。"""
    di = builder.load_di(builder.D_ITEMS)
    out = {}
    for it in di:
        rg = builder.hard_range(di, it)
        if rg:
            out[it] = (di[it], rg)
    return dict(sorted(out.items()))


def load_native_log(ds: Path, split: str):
    p = ds / split / "mimic_native_error_log.csv"
    if not p.is_file():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def peer_pool(ds: Path, split: str, builder, di, keys_wanted):
    """从 **源 reference**（干净表）复算 (subject_id, itemid) 的可用读数池。

    ⚠️ 必须用**源** reference（不含原生错误层的那份）：构建器算中位数时用的就是它。
    ⚠️ 口径与构建器完全一致（inject_examples 的 usable() + scan_reference）：
      可用 = valuenum 非空、≠ 0、且 classify() 不判为错（哨兵码 / 越界）
    返回 {group: [(行号, 值), ...]}
    """
    pools = defaultdict(list)
    if not keys_wanted:
        return pools
    ref = ds / split / "reference" / "icu" / "chartevents.csv"
    if not ref.is_file():
        return pools
    with ref.open(newline="", encoding="utf-8") as f:
        for L, row in enumerate(csv.DictReader(f)):
            try:
                it = int(row["itemid"])
            except (ValueError, KeyError):
                continue
            g = (row.get("subject_id", ""), it)
            if g not in keys_wanted:
                continue
            v = builder.num(row.get("valuenum"))
            if v is None or v == 0:
                continue
            cat, _ = builder.classify(di, it, v)
            if cat is None:
                pools[g].append((L, v))
    return pools


def median_of(arr):
    s = sorted(arr)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def fmt_val(v):
    f = float(v)
    return ("%.1f" % f) if f == int(f) else ("%g" % f)


def build_examples(ds: Path, src_ds: Path, builder, di):
    """train 分片里能对照出来的原生错误例子（按 (subject_id, itemid) 分组）。

    ⚠️ 每个 (subject_id, itemid) 是**一个独立的组**（中位数按患者算）——
       同一个 itemid 在不同患者上的中位数不一样，不能合并。
    ⚠️ 池子 = **源 reference** 里该组的可用读数，**排除坏格自己**
       （构建器的口径：med = median(us \\ {L_bad})，n_peer = len(rest)）。
       复算方式与构建器一致：坏格 = 该组可用读数里**离中位数最远**的那一个。
    """
    rows = load_native_log(ds, "train")
    if not rows:
        return []
    groups = {}
    for r in rows:
        groups[(r.get("subject_id", ""), int(r["itemid"]))] = r
    pools = peer_pool(src_ds, "train", builder, di, set(groups))
    out = []
    for (subj, it), r in sorted(groups.items(), key=lambda kv: (kv[1]["itemid"], kv[0])):
        us = pools.get((subj, it), [])
        if not us:
            out.append({"itemid": it, "label": r.get("item_label", "") or di.get(it, ""),
                        "n": 1, "dirty": r.get("raw_dirty_value", ""),
                        "repaired": [r.get("repaired_value", "")],
                        "range": r.get("plausibility_range", ""),
                        "mode": r.get("detection_mode", ""),
                        "n_peer": int(r.get("n_peer_readings") or 0),
                        "peer_min": None, "peer_max": None,
                        "median": r.get("repaired_value", ""), "checked": False})
            continue
        vals = [v for _, v in us]
        med_all = median_of(vals)
        L_bad, v_bad = max(us, key=lambda t: abs(t[1] - med_all))
        rest = [v for Lx, v in us if Lx != L_bad]
        med = median_of(rest)
        logged = r.get("repaired_value", "")
        checked = False
        try:
            checked = abs(float(med) - float(logged)) < 1e-6
        except (TypeError, ValueError):
            checked = False
        out.append({
            "itemid": it, "label": r.get("item_label", "") or di.get(it, ""),
            "n": 1, "dirty": r.get("raw_dirty_value", ""),
            "repaired": [logged], "range": r.get("plausibility_range", ""),
            "mode": r.get("detection_mode", ""),
            "n_peer": len(rest),
            "peer_min": min(rest) if rest else None,
            "peer_max": max(rest) if rest else None,
            "median": med, "checked": checked,
        })
    return out


def render_hard_table(hard, seen_in_test):
    """渲染 itemid ↔ 区间对照表（两列并排），并标出 test 上出现过原生错误的。"""
    items = list(hard.items())
    half = (len(items) + 1) // 2
    left, right = items[:half], items[half:]
    lines = ["| itemid | 测量项 | 不可能区间 | itemid | 测量项 | 不可能区间 |",
             "|---|---|---|---|---|---|"]
    for i in range(half):
        li, lr = left[i]
        if i < len(right):
            ri, rr = right[i]
            cells = (f"| {li} | {lr[0]} | [{lr[1][0]}, {lr[1][1]}] | "
                     f"{ri} | {rr[0]} | [{rr[1][0]}, {rr[1][1]}] |")
        else:
            cells = (f"| {li} | {lr[0]} | [{lr[1][0]}, {lr[1][1]}] | "
                     f"| | |")
        lines.append(cells)
    return "\n".join(lines), seen_in_test


def alias_stats(ds: Path, split="test"):
    """实测 itemid 别名注入（`chart_itemid_alias_split`）的条目数与规则符合率。"""
    log = ds / split / "modification_log.csv"
    n = ok = 0
    bad = Counter()
    if not log.is_file():
        return {"n": 0, "ok": 0, "rate": None, "bad": {}}
    with log.open(newline="", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            if (e.get("error_subtype") != "chart_itemid_alias_split"
                    or e.get("column") != "itemid"
                    or e.get("operation") != "cell_update"):
                continue
            dv = (e.get("dirty_value") or "").strip()
            cv = (e.get("clean_value") or "").strip()
            n += 1
            if len(dv) >= 7 and dv[:2] == "99" and dv[2] in "23":
                if "2" + dv[3] + dv[-4:] == cv:
                    ok += 1
                    continue
            bad[dv[:3]] += 1
    return {"n": n, "ok": ok,
            "rate": (100.0 * ok / n) if n else None,
            "bad": dict(bad.most_common(5))}


def injected_sentinels(ds: Path, split="test"):
    """注入层实际用到的哨兵/占位脏值（用来证明「清单不是封闭的」）。"""
    log = ds / split / "modification_log.csv"
    c = Counter()
    if not log.is_file():
        return c
    with log.open(newline="", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            if e.get("error_subtype") != "sentinel_chart_value":
                continue
            if e.get("column") != "valuenum":
                continue
            c[(e.get("dirty_value") or "").strip()] += 1
    return c


def build_magnitude(manifest, nat_by_split):
    out = []
    for sp in ("train", "validation", "test"):
        s = manifest["splits"].get(sp)
        if not s:
            continue
        sub = s.get("subtype_counts", {})
        cc = sum(v for k, v in sub.items()
                 if k not in ("duplicate_row", "label_leakage_feature"))
        dup = sub.get("duplicate_row", 0)
        leak = sub.get("label_leakage_feature", 0)
        nat = nat_by_split.get(sp, {})
        mode = nat.get("by_detection_mode", {})
        t2 = mode.get("T2_impossible", 0) + mode.get("T2_injected_example", 0)
        t1 = mode.get("T1_sentinel", 0) + mode.get("T1_injected_example", 0)
        n_ex = sum(v for k, v in mode.items() if k.endswith("_injected_example"))
        extra = ""
        if n_ex:
            extra = ("（其中 **%s 格是 train/validation 分片里注入的同形态示例**，"
                     "专门用来让你从 train 对照里学到这条规则）" % fmt(n_ex))
        out.append(
            "- %s: 约 %s cell_update + %s duplicate_row + %s label_leakage 插入行 "
            "= %s 个错误分布在 %s 个 subject 上（cell-level %.2f%%，%s 行）"
            % (SPLIT_NAMES[sp], fmt(cc), fmt(dup), fmt(leak), fmt(s["actual_errors"]),
               fmt(s.get("stays_touched") or 0),
               (s.get("cell_level_error_rate_fraction") or 0) * 100,
               fmt(s["total_rows"])))
        out.append(
            "  - 其中 **MIMIC 自身的原生错误 %s 格**（不是注入器造成的）："
            "生理不可能区间 **%s** 格 + 哨兵占位码 **%s** 格%s —— "
            "这是本次新增的错误类，按下面「新增错误类」一节修"
            % (fmt(nat.get("cells", 0)), fmt(t2), fmt(t1), extra))
    return "\n".join(out)


NATIVE_SECTION = """## 新增错误类：MIMIC 自身的原生错误（不是注入器造成的）

除了上面那些**注入**错误，train/validation raw 里还有另一类错误：**MIMIC 自己就录错的值** ——
任何人都测不出来的数。它们在数据上长得跟注入错误很像（都是"raw 的值不对、reference 的值对"），
但成因不同、修复方式也不同，**别用注入错误那套规则去套**。

⚠️ 这一类错误**只出现在 `icu/chartevents.csv` 的 `valuenum` 列**上，别的表没有。

### 类 1：查「生理不可能区间」（只对这 {N_HARD} 个 itemid 生效）

下表是本数据集 `icu/chartevents.csv` 里所有带**生理不可能区间**的测量项。
**值落在区间外、且值不等于 0** 的，就是 MIMIC 自己录错的原生错误。

{HARD_TABLE}

**重要：train 分片里只出现过其中 {N_SEEN} 个 itemid**（{SEEN_LIST}）。
**其余 {N_UNSEEN} 个在 train/validation 里一个原生错误都没有**（因为 train/validation 的患者少），
但 test 上会出现（test 上这些 itemid 的原生错误格数：{TEST_HARD_COUNTS}）——
所以**要按上表整张实现，不要只写 train 里见过的那几个**。

区间背后的通用规律（写代码时如果想按 label 归类也可以）：
{RULE_TEXT}

### 类 2：哨兵占位码（对**所有** itemid 生效）

MIMIC/CareVue 在「这个读数没取到 / 取错了」时会写入一个**占位码**。这些码**根本不是测量值**：

`{SENTINELS}`

**这一类对所有 itemid 生效** —— 不限上面那张表的 {N_HARD} 个。检验类项目（Albumin / Platelet /
PT / INR / Hemoglobin / Hematocrit / WBC / BUN 等）没有「生理不可能区间」可用，
但它们的哨兵码一样是垃圾值。

⚠️ `220000` 和 `220045` 这两个码长得像 itemid，但它们出现在 **`valuenum` 列**里时是错误码，不是 itemid。

⚠️⚠️ **上面这份清单不是封闭的 —— 请务必自己统计一遍 `valuenum` 的取值分布。**
除上面这组 MIMIC 占位码之外，**注入器还会用别的「不是测量值」的码**。
本数据集的注入层就有 **{INJ_SENT} 格**这类错误（脏值 top：{INJ_SENT_TOP}），
其中 **`-1` 和 `-888` 不在上面那份清单里** —— 只认上面那份清单 = 这类错误**一格都修不掉**。
⇒ 请对 `icu/chartevents.csv` 的 `valuenum` 做一次频次统计（例如 `value_counts()`），
把「出现次数极大 + 明显不是测量值」的值**全部**列出来，按同样的原则处理。

**两类原生错误的修复池子不一样**：
- 类 1（越界）的可用读数 = **落在该 itemid 区间内**且 ≠ 0；
- 类 2（哨兵码）的可用读数 = **不是哨兵码**且 ≠ 0（若该 itemid 另有区间，还要落在区间内）。

### 怎么修：用**该患者自己**该测量项的中位数

修复值 = **该 (subject_id, itemid) 在「最终干净结果」里的、该 itemid 的可用读数的中位数**
（偶数个时取中间两个的平均，和 numpy/pandas 的 median 一致）。

这里"最终干净结果"是指你已经做完所有其它修复之后的那张干净表：cohort 选好了、重复行删掉了、
注入错误（量纲放大、哨兵值等）也修回来了。**换句话说：中位数要在 pipeline 的最后一步算**，
对你自己已经产出的干净 `icu/chartevents.csv` 再做一遍后处理：

1. 先把前面所有错误修完，产出干净的 chartevents 表；
2. 在这张干净表上，按上面两张表逐行查：itemid 命中任一表 + valuenum 落在区间外（或等于哨兵码）
   + valuenum != 0 → 这是原生错误格；
3. 对每个这样的格，用**同一张干净表里**该 (subject_id, itemid) 的所有可用读数的中位数替换；
4. 同一 (subject_id, itemid) 的多个坏格，修复值是**同一个**中位数。

⚠️ **坏格自己不在池子里** —— 它的值要么越界、要么是哨兵码，本来就被"可用"这个条件剔掉了
（**不要**再手动把它减掉，那样会重复扣一次）。
⚠️ 反过来：**如果某个读数当前看起来"可用"，但它是别的注入错误造出来的**
（量纲放大 100 倍、哨兵码），它会污染中位数 ⇒ 必须先把注入错误修完再算。

**为什么强调"最后一步"**：如果你在 cohort 筛选 / 去重 / 注入修复**之前**就去算中位数，
池子里会混进 ①别的住院（同一个患者可能有多次住院，其中有些不属于本队列）的读数
②重复行 ③量纲放大/哨兵值弄坏的读数（它们落在区间外，会被当成坏值剔掉，
于是本该参与中位数的正常读数少了几个）。这三种污染每一种都会让中位数偏一点，
结果就和 reference 对不上。

### train 里能对照出来的 {N_EX} 组例子（每个 (患者, itemid) 独立算）

{EXAMPLES}

### 注意

- **不要对 0 动手**：MIMIC 里 0 常表示"未测到"或"呼吸机暂停"，不是错误。区间判定里 `!= 0` 是硬条件。
- **不要把 labs 超出 MIMIC 自带参考区间（ref_range_lower / ref_range_upper）的值当错误**：
  那是临床异常值（INR 1.2、贫血 Hgb 11、Cr 2.0），病人真的有这个值。base excess / 尿电解质 /
  粪电解质的负值也是正常的。**这一类错误只在 `icu/chartevents.csv` 的 `valuenum` 上，别的表没有。**
- **如果该患者该测量项一个可用读数都没有**（整段都是坏值 —— 这通常是换能器校准一类的整体平移，
  真实值不可判定）：**保持原样不动**，不要用全局统计量硬凑。
- **不要扩大到所有离群点**：只处理"落在上表区间外"或"是哨兵码"的孤立尖峰。
  同一患者某项读数整体偏高或偏低是病情，不是错误。
- train/validation 里这类错误只有十几格，**但测试集规模大 100 倍**。请把规则实现成
  **逐 (subject_id, itemid) 计算**的通用逻辑，不要写成只认上面这几个 itemid 的特判。
- **别忘了注入错误要先修**：如果一个格同时被量纲放大（×100）和原生错误两种问题命中，
  先按注入错误的规则修好，再走上面的中位数填补 —— 否则被注入错误弄坏的读数会污染中位数。

"""

# ── v4：把「两类原生错误」从散文升级成**可执行骨架** ──────────────
# 实测（1.6pct 第 2 轮）：agent 的日志里「生理不可能」出现 **0 次** ⇒
# **越界类（T2）整节被跳过**，16 格一格没修；
# 而哨兵码类（T1）修了 11/15 —— 漏掉的 4 格是**池子分组键用错**
# （agent 按 (stay_id, itemid) 分组，真值是 (subject_id, itemid)）。
# ⇒ 散文 + 表格会被忽略，**代码骨架不会**。
NATIVE_CODE_SECTION = """### ⚠️ 这两类**都必须**实现 —— 附可直接用的实现骨架

⚠️⚠️ **不要只实现哨兵码那一类** —— 越界类和哨兵码类**都要判**。
（实测教训：只实现哨兵码的那一版，越界类 **16 格一格没修**。）

```python
# ── MIMIC 原生错误：两类判定 ───────────────────────────────
# 类 2：哨兵占位码（对**所有** itemid 生效）
NATIVE_SENTINELS = {-999.0, -998.0, -997.0, -990.0, 9999.0, 99999.0,
                     999999.0, 9999999.0, 220000.0, 220045.0}
# 类 1：生理不可能区间（**只对下面这些 itemid 生效**；
#        值 != 0 且落在区间外 ⇒ 原生错误）
IMPLAUSIBLE_RANGE = {
    220045: (20, 300),  220050: (40, 400),  220051: (20, 250),
    220052: (20, 250),  220059: (0, 150),   220061: (0, 100),
    220074: (-20, 80),  220179: (40, 400),  220180: (20, 250),
    220181: (20, 250),  220210: (0, 100),   220562: (8, 200),
    223762: (25, 45),   224167: (40, 400),  224643: (20, 250),
    224688: (0, 100),   224689: (0, 100),   224690: (0, 100),
    227242: (20, 250),  227243: (40, 400),  227466: (8, 200),
}


def repair_native_errors(ce):
    \"\"\"在**最终干净表**上跑（pipeline 的最后一步）。
    ⚠️ 池子按 (subject_id, itemid) 分组 —— **不是 (stay_id, itemid)**。
    \"\"\"
    ce = ce.copy()

    def as_int(x):
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return None

    def usable(it, v):
        if v is None or v == 0:            # 0 不是错误（MIMIC 里 0 = 未测到）
            return False
        if v in NATIVE_SENTINELS:         # 哨兵码
            return False
        rg = IMPLAUSIBLE_RANGE.get(it)
        return rg is None or (rg[0] <= v <= rg[1])

    iters = [as_int(x) for x in ce["itemid"]]
    vals = []
    for v in ce["valuenum"]:
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            vals.append(None)

    # ① 标出坏格：哨兵码 或 越界（且 != 0）
    bad = []
    for i, (it, v) in enumerate(zip(iters, vals)):
        if it is None or v is None or v == 0:
            continue
        if v in NATIVE_SENTINELS:
            bad.append(i)
            continue
        rg = IMPLAUSIBLE_RANGE.get(it)
        if rg is not None and not (rg[0] <= v <= rg[1]):
            bad.append(i)

    # ② 池子：同一 (subject_id, itemid) 的所有**可用**读数
    #    坏格自己被自动剔掉 —— 它的值越界 / 是哨兵码，本来就不"可用"
    pools = {}
    for i, r in ce.iterrows():
        it, v = iters[i], vals[i]
        if not usable(it, v):
            continue
        pools.setdefault((r["subject_id"], it), []).append(v)

    # ③ 用中位数替换
    n_fixed = 0
    for i in bad:
        key = (ce.at[i, "subject_id"], iters[i])
        pool = pools.get(key)
        if not pool:
            continue        # 一个可用读数都没有 ⇒ 真值不可判定 ⇒ 保持原样不动
        med = float(pd.Series(pool).median())
        ce.at[i, "valuenum"] = med
        n_fixed += 1
    return ce, n_fixed
```

⚠️ **池子为什么是 (subject_id, itemid) 而不是 (stay_id, itemid)**：
同一个病人可能有多次住院、多次 ICU 入住。按 stay_id 分组会把**别的 stay 的读数**
混进池子（或反过来漏掉本 stay 的），中位数就偏了。
实测：按 (stay_id, itemid) 分组的那一版，有 4 格的修复值是错的。
⇒ **按 (subject_id, itemid) 分组**。

⚠️ **不要假设 raw 的 `value` 文本列是干净的**：
- **注入层**（量纲放大、哨兵码）：`value` 记的是干净数、`valuenum` 被改 ⇒ 可以用 `value` 还原；
- **原生错误层**：MIMIC 把坏数**同时**写进了 `value` 和 `valuenum`（两列一样）
  ⇒ 原生错误**必须独立判定**（靠区间 / 哨兵码），
  **靠「`value` 与 `valuenum` 不一致」是发现不了原生错误的**。

"""

CHECKLIST_SECTION = """### ✅ 提交前的自检清单（**逐条核对**，缺一条就别提交）

- [ ] **两类原生错误都实现了**：哨兵码（对所有 itemid）**和** 越界（对上面那 {N_HARD} 个 itemid）
- [ ] **train 的 {N_T2} 格越界示例全部修好**：{T2_LIST}
- [ ] **train 的 {N_T1} 格哨兵码示例全部修好**：{T1_LIST}
- [ ] **train 自检是逐格比对** `icu/chartevents.csv` 的 `valuenum`（按 5 段 key 对齐，
      逐格列出 raw ≠ reference 的地方），**不是**只看行数 / 聚合统计
      —— ⚠️ 只看聚合统计会把**整类**错误漏掉（这正是本数据集最容易踩的坑）
- [ ] **修复池按 (subject_id, itemid) 分组**（不是 stay_id）
- [ ] 修复值 = 该 (subject_id, itemid) 可用读数的**中位数**（偶数个取中间两个的平均）
- [ ] 该患者该项**一个可用读数都没有**的坏格**保持原样不动**（真值不可判定）

"""

ALIAS_SECTION = """## ⚠️ 另外：`itemid` 列上还有一批 **CareVue 别名注入**（**key 列**，不改就大面积键对不上）

raw 的 `itemid` 被改成了 `992{{2|3}}XXXX` 形式 —— 例如 `220045` → `99220045` / `99230045`。
**test 分片上实测有 {ALIAS_N} 条（占注入错误 {ALIAS_PCT}）**{ALIAS_VERIFY}
⇒ 还原方式：`原 itemid = "2" + alias[3] + alias[-4:]`
（`alias[3]` 是原 itemid 的**第三位**，`alias[-4:]` 是**末四位**；把前两位 `99` 换成 `2`）。

⚠️ **这是 key 列上的错误 ⇒ 必须在「去重」之前修**（去重跑在修 key 之前的话，
修好的行会和别的行撞成重复，或者反过来把该留的行删掉）。
⚠️ `220045` / `227466` 这类**长得像 itemid 的码**在 `valuenum` 列里是哨兵错误码（见下一节），
在 `itemid` 列里才是 itemid —— 别搞混。

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--src-dataset", required=True,
                    help="母数据集（无原生错误层）—— 中位数池子要从它的 reference 复算")
    ap.add_argument("--base-prompt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = Path(args.dataset)
    src_ds = Path(args.src_dataset)
    text = Path(args.base_prompt).read_text(encoding="utf-8")
    manifest = json.loads(
        (ds / "v2_rawerror_n18_final_report.json").read_text(encoding="utf-8"))
    build = {}
    br = ds / "mimic_native_error_build_report.json"
    if br.is_file():
        build = json.loads(br.read_text(encoding="utf-8"))

    builder = load_builder()
    di = builder.load_di(builder.D_ITEMS)
    hard = hard_table(builder)

    nat_by_split = manifest.get("mimic_native_error_layer", {}).get("by_split", {})

    # ---- ① 量级三行（重算 + 原生层规模）----
    lines = text.split("\n")
    hits = []
    for pref in A_MAGNITUDE:
        h = [i for i, l in enumerate(lines) if l.startswith(pref)]
        if len(h) != 1:
            raise SystemExit("FATAL: 量级行 %r 命中 %d 次" % (pref, len(h)))
        hits.append(h[0])
    if hits[1] != hits[0] + 1 or hits[2] != hits[1] + 1:
        raise SystemExit("FATAL: 量级三行不连续：%s" % hits)
    lines[hits[0]:hits[2] + 1] = build_magnitude(manifest, nat_by_split).split("\n")
    text = "\n".join(lines)

    # ---- ② 原生错误节 ----
    train_rows = load_native_log(ds, "train")
    val_rows = load_native_log(ds, "validation")
    test_rows = load_native_log(ds, "test")
    train_items = {int(r["itemid"]) for r in train_rows}
    val_items = {int(r["itemid"]) for r in val_rows}
    test_hard = Counter(int(r["itemid"]) for r in test_rows if int(r["itemid"]) in hard)
    seen = sorted(train_items & set(hard))
    unseen = sorted(set(hard) - set(train_items) - set(val_items))

    table, _ = render_hard_table(hard, test_hard)
    rule_text = "、".join(
        "%s [%d,%d]" % (k, v[0], v[1])
        for k, v in builder.HARD_BY_LABEL.items())
    sentinels = build.get("sentinels") or sorted(builder.SENTINELS)
    sent_txt = "、".join(
        (fmt_val(s) if float(s) != int(float(s)) else str(int(float(s))))
        for s in sentinels)

    exs = build_examples(ds, src_ds, builder, di)
    ex_lines = []
    n_checked = 0
    for e in exs:
        rng = ("合理区间 %s" % e["range"]) if e["range"] not in ("sentinel_code", "") \
            else "该 itemid 没有硬区间，属哨兵码类"
        peer = ""
        if e["peer_min"] is not None:
            peer = ("该患者该测量项的**其它** %d 个可用读数都在 %s~%s 之间，"
                    "**中位数 %s**"
                    % (e["n_peer"], fmt_val(e["peer_min"]), fmt_val(e["peer_max"]),
                       fmt_val(e["median"])))
        else:
            peer = "该患者该测量项没有其它可用读数"
        rep = "、".join(e["repaired"])
        kind = "越界" if e["mode"].startswith("T2") else "哨兵占位码"
        n_checked += 1 if e["checked"] else 0
        ex_lines.append(
            "- **itemid `%d`（%s，%s类）**：train raw 里该患者该测量项有 %d 个读数是 **%s**"
            "（%s）；%s。train reference 把这个坏值改成了 **%s**。"
            % (e["itemid"], e["label"] or "label 未知", kind, e["n"],
               e["dirty"], rng, peer, rep))
    examples_text = "\n".join(ex_lines) if ex_lines else "（train 分片里没有可比对的例子）"

    inj_s = injected_sentinels(ds, "test")
    inj_top = "、".join("%s（%s 条）" % (k if k else "(空)", fmt(v))
                        for k, v in inj_s.most_common(6))
    inj_sent_total = sum(inj_s.values())

    al = alias_stats(ds, "test")
    sub = manifest["splits"]["test"].get("subtype_counts", {})
    alias_n = sub.get("chart_itemid_alias_split", 0)
    alias_total = manifest["splits"]["test"].get("actual_errors", 0) or 1
    alias_verify = ""
    if al["n"]:
        alias_verify = ("；实测其中 **%s 条（%.1f%%）**能被「`2` + alias[3] + alias[-4:]」"
                        "还原成一个真实存在的 itemid"
                        % (fmt(al["ok"]), al["rate"]))

    native = NATIVE_SECTION.format(
        N_HARD=len(hard),
        HARD_TABLE=table,
        N_SEEN=len(seen),
        SEEN_LIST=" / ".join(str(i) for i in seen) if seen else "（无）",
        N_UNSEEN=len(unseen),
        TEST_HARD_COUNTS="、".join("%s %s" % (i, fmt(test_hard[i]))
                                   for i in sorted(test_hard)) or "（无）",
        RULE_TEXT=rule_text,
        SENTINELS=sent_txt,
        INJ_SENT=fmt(inj_sent_total),
        INJ_SENT_TOP=inj_top,
        N_EX=len(exs),
        EXAMPLES=examples_text,
    )
    alias = ALIAS_SECTION.format(
        ALIAS_N=fmt(alias_n),
        ALIAS_PCT="%.1f%%" % (100.0 * alias_n / alias_total),
        ALIAS_VERIFY=alias_verify,
    )

    # ---- ④ 代码骨架 + ⑤ 自检清单（v4）----
    def render_examples(rows, prefix):
        """把 train 原生示例按 (itemid, 脏值) 聚合成清单。"""
        c = Counter()
        for r in rows:
            if not r.get("detection_mode", "").startswith(prefix):
                continue
            c[(r.get("itemid", ""), r.get("raw_dirty_value", ""))] += 1
        parts = []
        for (it, dv), n in sorted(c.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
            parts.append("itemid %s（脏值 %s）%s"
                         % (it, dv, ("×%d" % n) if n > 1 else ""))
        return "、".join(parts), sum(c.values())

    t2_list, n_t2 = render_examples(train_rows, "T2")
    t1_list, n_t1 = render_examples(train_rows, "T1")
    NATIVE_CODE = NATIVE_CODE_SECTION
    checklist = CHECKLIST_SECTION.format(
        N_HARD=len(hard), N_T2=n_t2, N_T1=n_t1,
        T2_LIST=t2_list, T1_LIST=t1_list)

    if A_FLOW not in text:
        raise SystemExit("FATAL: 找不到锚点 %r" % A_FLOW)
    text = text.replace(A_FLOW, alias + native + NATIVE_CODE + A_FLOW, 1)

    # ---- ⑥ 提交前自检清单（放在「重要约束」之前）----
    if A_CONSTRAINT not in text:
        raise SystemExit("FATAL: 找不到锚点 %r" % A_CONSTRAINT)
    text = text.replace(A_CONSTRAINT, checklist + A_CONSTRAINT, 1)

    # ---- ⑤ result_root 措辞加固 ----
    if A_SUBMIT not in text:
        raise SystemExit("FATAL: 找不到锚点 %r" % A_SUBMIT)
    text = text.replace(A_SUBMIT, A_SUBMIT_NEW, 1)

    Path(args.out).write_text(text, encoding="utf-8")
    print("已写出:", args.out, "(%d bytes)" % len(text.encode("utf-8")))
    print("  硬区间 itemid: %d 个（train 见过 %d / test 上有原生错误的 %d）"
          % (len(hard), len(seen), len(test_hard)))
    print("  train 原生例子: %d 组（中位数与 log 复算一致 %d 组）"
          % (len(exs), n_checked))
    print("  itemid 别名注入(test): %s 条，规则符合率 %s"
          % (fmt(al["n"]), ("%.2f%%" % al["rate"]) if al["rate"] is not None else "n/a"))
    print("  注入层哨兵脏值 top: %s" % inj_top)


if __name__ == "__main__":
    main()
