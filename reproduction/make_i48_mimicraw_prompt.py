#!/usr/bin/env python3
"""make_i48_mimicraw_prompt.py —— 由 i48 的 mimicraw prompt 生成 **v5**（加固版）。

相对上一版（v3 / v4）加了 4 件事，每一件都对应一个实测出来的问题：

  1. **T1 哨兵占位码类**（原生错误类的第二类，对**所有** itemid 生效）
     —— 上一版只教了「生理不可能区间」（T2）。
     实测 i48 的 test 上 T1 有 34 格（1.6pct 档）。
  2. **「哨兵码清单不是封闭的」** + 要求 agent 自己统计 `valuenum` 取值分布
     —— ICU5000 上实测：prompt 把哨兵码清单写成封闭集合时，
        agent 会漏掉注入层用的 `-1` / `-888`，4.7pct 档因此有 **281 万格**
        原生/注入哨兵码一格没修（b_dirty 263 万）。i48 的 `sentinel_chart_value`
        注入量级是 **107 万（1.6pct）/ 771 万（4.7pct）**，同样的坑必须提前堵。
  3. **itemid 别名注入一节**（`chart_itemid_alias_split`）
     —— 上一版完全没有。i48 的 test 上有 **12.3 万（1.6pct）/ 39.7 万（4.7pct）**
        条 itemid 被改成 `992{2|3}XXXX`，不改就大面积键对不上。
  4. **原生错误层的规模数字**（train/validation 各 11 格、test 多少格）
     —— 上一版只说「train/validation 里这类错误只有十几格」，没给确切数。

用法:
  python3 make_i48_mimicraw_prompt.py --dataset <mimicraw2 数据集> \
      --base-prompt <v3/v4 prompt> --out <输出>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SENTINELS = [-999, -998, -997, -990, 9999, 99999, 999999, 9999999,
              220000, 220045]

# ── 锚点（必须与基线 prompt 里的原文完全一致，否则报错）──
A_JUDGE = "### 怎么判：查「生理不可能区间」"
A_JUDGE_NEW = "### 类 1：查「生理不可能区间」（只对这 20 个 itemid 生效）"
A_REPAIR = "### 怎么修：用**该患者自己**该测量项的中位数"
A_NATIVE_HEAD = "## 新增错误类：MIMIC 自身的原生错误（不是注入器造成的）"
A_MAGNITUDE = ("- train: 约 ", "- validation: 约 ", "- 最终测试集: 约 ")
A_TAIL = "### 推导与修正流程"

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

T1_SECTION = """### 类 2：哨兵占位码（对**所有** itemid 生效）

MIMIC/CareVue 在「这个读数没取到 / 取错了」时会写入一个**占位码**。这些码**根本不是测量值**：

`{SENTINELS}`

**这一类对所有 itemid 生效** —— 不限上面那张表的 20 个。检验类项目（Albumin / Platelet /
PT / INR / Hemoglobin / Hematocrit / WBC / BUN 等）没有「生理不可能区间」可用，
但它们的哨兵码一样是垃圾值。

⚠️ `220000` 和 `220045` 这两个码长得像 itemid，但它们出现在 **`valuenum` 列**里时是错误码，不是 itemid。

⚠️⚠️ **上面这份清单不是封闭的 —— 请务必自己统计一遍 `valuenum` 的取值分布。**
除上面这组 MIMIC 占位码之外，**注入器还会用别的「不是测量值」的码**
（常见形态是 `-1` / `-888` / `-1000` 这类负数或固定的整数）。
它们在 raw 里会**成千上万次重复出现**，而且**通常不在上面这份清单里**
（本数据集的注入层就有 **{INJ_SENT} 格**这类错误，量级上百万）。
⇒ 请对 `icu/chartevents.csv` 的 `valuenum` 做一次频次统计（例如 `value_counts()`），
把「出现次数极大 + 明显不是测量值」的值**全部**列出来，按同样的原则处理。
⇒ **只认上面那份清单 = 这类注入错误一格都修不掉。**

**两类原生错误的修复池子不一样**：
- 类 1（越界）的可用读数 = **落在该 itemid 区间内**且 ≠ 0；
- 类 2（哨兵码）的可用读数 = **不是哨兵码**且 ≠ 0。

"""


def fmt(n):
    return "{:,}".format(int(n))


def find_magnitude_block(text: str):
    """定位量级三行（train / validation / 最终测试集）。"""
    lines = text.split("\n")
    idx = []
    for pref in A_MAGNITUDE:
        hit = [i for i, l in enumerate(lines) if l.startswith(pref)]
        if len(hit) != 1:
            raise SystemExit("FATAL: 量级行 %r 命中 %d 次" % (pref, len(hit)))
        idx.append(hit[0])
    if idx[1] != idx[0] + 1 or idx[2] != idx[1] + 1:
        raise SystemExit("FATAL: 量级三行不连续：%s" % idx)
    return idx[0], idx[2]


def build_magnitude(manifest, nat_by_split):
    """用实测数字重写量级三行 + 追加原生层规模。"""
    out = []
    names = {"train": "train", "validation": "validation", "test": "最终测试集"}
    for sp in ("train", "validation", "test"):
        s = manifest["splits"].get(sp)
        if not s:
            continue
        sub = s.get("subtype_counts", {})
        cc = sum(v for k, v in sub.items()
                 if k not in ("duplicate_row", "label_leakage_feature"))
        dup = sub.get("duplicate_row", 0)
        leak = sub.get("label_leakage_feature", 0)
        total = s["actual_errors"]
        rate = s["cell_level_error_rate_fraction"]
        rows = s["total_rows"]
        nsub = s.get("stays_touched")
        nat = nat_by_split.get(sp, {})
        out.append(
            "- %s: 约 %s cell_update + %s duplicate_row + %s label_leakage 插入行 "
            "= %s 个错误分布在 %s 个 subject 上（cell-level %.2f%%，%s 行）"
            % (names[sp], fmt(cc), fmt(dup), fmt(leak), fmt(total),
               fmt(nsub) if nsub is not None else "?", rate * 100, fmt(rows)))
        if nat:
            by_mode = nat.get("by_detection_mode", {})
            out.append(
                "  - 其中 **MIMIC 原生错误（不是注入器造成的）%s 格**"
                "（类 1 越界 %s + 类 2 哨兵码 %s）—— 这是本次新增的错误类，"
                "按下一节的规则修"
                % (fmt(nat.get("cells", 0)),
                   fmt(by_mode.get("T2_impossible", 0)),
                   fmt(by_mode.get("T1_sentinel", 0))))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="mimicraw2 数据集目录")
    ap.add_argument("--base-prompt", required=True, help="基线 prompt（v3 / v4）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ds = Path(args.dataset)
    text = Path(args.base_prompt).read_text(encoding="utf-8")

    manifest = json.loads(
        (ds / "v2_rawerror_i48_final_report.json").read_text(encoding="utf-8"))
    nat = manifest.get("mimic_native_error_layer", {}).get("by_split", {})

    # ---- ① 量级三行 ----
    lo, hi = find_magnitude_block(text)
    lines = text.split("\n")
    lines[lo:hi + 1] = build_magnitude(manifest, nat).split("\n")
    text = "\n".join(lines)

    # ---- ② 类 1 / 类 2 标题 ----
    if A_JUDGE not in text:
        raise SystemExit("FATAL: 找不到锚点 %r" % A_JUDGE)
    text = text.replace(A_JUDGE, A_JUDGE_NEW, 1)

    # ---- ③ 插入 T1 哨兵码节 ----
    if A_REPAIR not in text:
        raise SystemExit("FATAL: 找不到锚点 %r" % A_REPAIR)
    sub = manifest["splits"]["test"].get("subtype_counts", {})
    inj_sent = sub.get("sentinel_chart_value", 0)
    t1 = T1_SECTION.format(
        SENTINELS="、".join(str(s) for s in SENTINELS),
        INJ_SENT=fmt(inj_sent))
    text = text.replace(A_REPAIR, t1 + A_REPAIR, 1)

    # ---- ④ itemid 别名注入节 ----
    if A_NATIVE_HEAD not in text:
        raise SystemExit("FATAL: 找不到锚点 %r" % A_NATIVE_HEAD)
    alias_n = sub.get("chart_itemid_alias_split", 0)
    alias_total = manifest["splits"]["test"].get("actual_errors", 0) or 1
    alias_pct = 100.0 * alias_n / alias_total
    # 规则符合率由 i48_alias_verify.py 实测（不写死「100%」这种没验证的话）
    verify = ""
    vf = Path(args.dataset) / "i48_alias_verify.json"
    if vf.is_file():
        v = json.loads(vf.read_text(encoding="utf-8"))
        if v.get("n"):
            verify = ("；实测其中 **%s 条（%.1f%%）**能被「`2` + alias[3] + alias[-4:]」"
                      "还原成一个真实存在的 itemid"
                      % (fmt(v["rule_ok"]), 100.0 * v["rule_ok"] / v["n"]))
    alias = ALIAS_SECTION.format(
        ALIAS_N=fmt(alias_n), ALIAS_PCT="%.1f%%" % alias_pct, ALIAS_VERIFY=verify)
    text = text.replace(A_NATIVE_HEAD, alias + A_NATIVE_HEAD, 1)

    Path(args.out).write_text(text, encoding="utf-8")
    print("已写出:", args.out, "(%d bytes)" % len(text.encode("utf-8")))
    print("  原生层规模: %s" % {k: v.get("cells") for k, v in nat.items()})
    print("  itemid 别名注入(test): %s" % fmt(alias_n))


if __name__ == "__main__":
    main()
