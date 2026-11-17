#!/usr/bin/env python3
"""把 mimicrawerror 数据集 log 里 `clean_value` 的**字符串形式**改成「raw / 原始 MIMIC 的形式」。

## 背景（两档 log 的唯一差异）
- **4.7pct**（v2 重建引擎）：log 的 `clean_value` = **MIMIC 原始串**（`'160'`）
  —— 与该档 raw 里未注入格的写法一致
- **1.6pct**（legacy 引擎）：log 的 `clean_value` = **float repr**（`'160.0'`）
  —— 与该档 gold（reference）的写法一致

⇒ 老口径（`--string-compare`，拿 log 的 `clean_value` 当判据）在两档上的行为完全不同，
   绝对值不可跨档比。本脚本把 1.6pct 改成与 4.7pct **同一套约定**。

## 转换规则（严格版）
`clean_value` **恰为 float repr 的整数** ⇒ 写成整数形式（`'160.0'` → `'160'`）；否则原样。
判据：`float(s)` 是整数且 `str(float(s)) == s`。
- `'160.0'` → 转；`'160'` → 不转（本就是整数形式）
- `'04280'` → **不转**（保留前导零；它是键列，见下）

（依据：MIMIC 原始 CSV 的整数不带 `.0` —— 在 4.7pct 的 log 上实测 **99.948%** 吻合。）
**实测：该规则对 4.7pct 的 log 零改动** ⇒ 与 4.7pct 约定完全兼容（幂等）；
对 1.6pct 改 `valuenum` 3,205,425 条 + `amount` 31,566 条。

## 只动注入器自己的条目
`error_subtype` 以 `mimic_native` 开头的条目（构建器追加的 873 条原生错误 + train/val 示例）
**不动** —— 它们的 `clean_value` 由构建器用 `repr(float(v))` 写，两档本就一致。

## 不动 `dirty_value`
它必须与 raw 里实际写入的字符串一致（否则 `apply_modification_log.py` 重建 raw 会变）；
实测两档的 `dirty_value` 都已与 raw 一致（1.6pct 抽样 300/300 全同）⇒ 无需改。

## ⚠️ 不动**键列**（重要）
评分器的 `RECOVER_CLEAN_KEY` 会拿 log 的 `clean_value` **按精确字符串**去还原被注入破坏的
key，再去 **agent 输出**里查这一行 ⇒ 所以键列的 `clean_value` 必须与 **agent 写出的那一列**
逐字节一致，否则 key 还原失败、该行被记成 `row_not_in_gold` 被排除。
实测：`diagnoses_icd.csv` 的键列含 `icd_code`，它的 `clean_value` 里有 **1,743 条
前导零形式**（如 `'04280'`）—— 宽松转换会把它们抹成 `'4280'`，
于是这 **1,743 行**的 key 还原失效（被排除、离开评估集）。
⇒ ⇒ **本脚本跳过所有键列**（各文件的键列见 RAW_TO_OUTPUT，与评分器一致），
且转换规则本身也保留前导零（双重保险）。

用法：
    python3 fix_k29_log_rawform.py --dataset <数据集目录>            # 只报告
    python3 fix_k29_log_rawform.py --dataset <数据集目录> --fix      # 真改
只输出聚合，不打印行级内容。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import Counter
from pathlib import Path

csv.field_size_limit(2 ** 31)

# 各文件的键列 —— 与 score_only/error_metrics/compute_error_metrics.py 的 RAW_TO_OUTPUT 一致。
# 键列的 clean_value 参与 key 还原，**不能改**。
KEY_COLUMNS = {
    "hosp/patients.csv": {"subject_id"},
    "hosp/admissions.csv": {"hadm_id"},
    "hosp/diagnoses_icd.csv": {"subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"},
    "icu/icustays.csv": {"stay_id"},
    "icu/chartevents.csv": {"subject_id", "hadm_id", "stay_id", "charttime", "itemid"},
    "icu/inputevents.csv": {"subject_id", "hadm_id", "stay_id", "starttime", "itemid"},
    "icu/outputevents.csv": {"subject_id", "hadm_id", "stay_id", "charttime", "itemid"},
}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def conv_key(e):
    """条目的唯一定位（同一 (文件, 行, 列) 可能有多条注入，取行号+列+序号更稳）"""
    return "%s|%s|%s" % (e.get("raw_file", ""), e.get("raw_row_index", ""),
                         e.get("column", ""))


def eligible(e):
    """这条要不要参与转换？（注入器自己的条目、非键列）"""
    if e.get("operation") != "cell_update":
        return False
    if e.get("error_subtype", "").startswith("mimic_native"):
        return False
    rf = e.get("raw_file", "")
    col = e.get("column", "")
    keycols = KEY_COLUMNS.get(rf)
    if keycols and col in keycols:
        return False
    return True


def pick(eligible_keys, ratio, seed):
    """确定性地选出 ratio 比例的条目（用 blake2b 哈希排序 ⇒ 可复现、与顺序无关）。"""
    if ratio >= 1.0:
        return set(eligible_keys)
    if ratio <= 0.0:
        return set()
    scored = []
    for k in eligible_keys:
        h = hashlib.blake2b(("%s|%s" % (seed, k)).encode("utf-8"), digest_size=8).digest()
        scored.append((h, k))
    scored.sort()
    n = int(round(len(scored) * ratio))
    return set(k for _, k in scored[:n])


def to_int_form(s):
    """**只**把「确为 float repr 的整数」改成整数形式；其它一律原样。

    判据：float(s) 是整数，且 ``str(float(s)) == s``（即 s 就是这个数的 float repr）
      - '160.0'  -> str(160.0)=='160.0'  ✔ 转成 '160'
      - '160'    -> str(160.0)=='160.0' != '160'   ⇒ 不转（本就是整数形式）
      - '04280'  -> str(4280.0)=='4280.0' != '04280' ⇒ 不转（**保留前导零**）

    为什么必须这么严：
      - 宽松规则（只要是整数就转）会把 '04280' 改成 '4280'，
        而 diagnoses_icd 的 icd_code **是键列**，评分器用它还原被破坏的 key ——
        改了就会还原失败（该行被记成 row_not_in_gold 而排除出评估集）。
        实测：这样会破坏 1,743 行。
      - 严格规则对 **4.7pct 的 log 零改动**（该档 log 已是原始串形式），
        即本规则与 4.7pct 的约定完全兼容；对 1.6pct 只改 valuenum / amount。
    """
    v = fnum(s)
    if v is None:
        return s
    if v != v or v in (float("inf"), float("-inf")):
        return s                       # NaN / inf：不是有限整数，原样
    if v != int(v) or abs(v) >= 1e15:
        return s
    if str(v) != s:
        return s
    return str(int(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", default="test,train,validation")
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="只把这么比例的可转换条目改成整数形式（0~1）。"
                         "确定性抽样（blake2b 哈希排序），--seed 控制选哪些")
    ap.add_argument("--seed", default="k29-mimicraw")
    args = ap.parse_args()
    ds = Path(args.dataset)

    for split in args.splits.split(","):
        log_path = ds / split / "modification_log.csv"
        if not log_path.is_file():
            print("=== %s：没有 log，跳过" % split)
            continue

        # ---- 第一遍：读全部行，收集可转换候选的 key ----
        with log_path.open(newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            header = rd.fieldnames
            rows = list(rd)

        keys = []
        for e in rows:
            if not eligible(e):
                continue
            cv = e.get("clean_value")
            if to_int_form(cv) != cv:
                keys.append(conv_key(e))
        chosen = pick(keys, args.ratio, args.seed)

        # ---- 第二遍：按比例写回 ----
        stat = Counter()
        for e in rows:
            if not eligible(e):
                if e.get("operation") == "cell_update" and \
                        e.get("error_subtype", "").startswith("mimic_native"):
                    stat["原生条目（不动）"] += 1
                elif e.get("operation") == "cell_update":
                    stat["键列条目（不动）"] += 1
                else:
                    stat["非 cell_update 条目（不动）"] += 1
                continue
            cv = e.get("clean_value")
            new = to_int_form(cv)
            if new == cv:
                stat["注入器条目：本来就是整数形式/非数值"] += 1
            elif conv_key(e) in chosen:
                stat["注入器条目：**已转**成整数形式"] += 1
                if args.fix:
                    e["clean_value"] = new
            else:
                stat["注入器条目：保留浮点形式（未转）"] += 1

        print("=== %s ===  ratio=%.4f  (候选 %s 条，选中 %s 条)"
              % (split, args.ratio, f"{len(keys):,}", f"{len(chosen):,}"))
        for k, v in stat.most_common():
            print("   %-36s %12s" % (k, f"{v:,}"))
        if args.fix:
            tmp = log_path.with_suffix(".csv.fixing")
            with tmp.open("w", newline="", encoding="utf-8") as fout:
                w = csv.DictWriter(fout, fieldnames=header, lineterminator="\n")
                w.writeheader()
                for e in rows:
                    w.writerow(e)
            os.replace(tmp, log_path)
            print("   已写回：%s" % log_path)
        print()


if __name__ == "__main__":
    main()
