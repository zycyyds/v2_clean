#!/usr/bin/env python3
"""i48_alias_verify.py —— 实测 i48 的 itemid 别名注入规则是否成立。

背景：`chart_itemid_alias_split` 把 raw 的 `itemid` 改成 `992{2|3}XXXX` 形式。
ICU5000 上实测还原规则是「原 itemid = "2" + alias[3] + alias[-4:]」，
**i48 上必须自己验一遍**（不能照抄）。

做法：扫 modification_log 里该 subtype 的条目 —— log 同时记了
      `dirty_value`（被改成的 alias）和 `clean_value`（原 itemid，权威来源）
      ⇒ 直接核对还原规则与 clean_value 是否一致。
只出聚合。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

csv.field_size_limit(2 ** 31)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ds = Path(args.dataset)

    log = ds / args.split / "modification_log.csv"
    n = 0
    ok = 0
    bad = Counter()
    examples = {}
    with log.open(newline="", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            if e.get("error_subtype") != "chart_itemid_alias_split":
                continue
            if e.get("column") != "itemid":
                continue
            if e.get("operation") != "cell_update":
                continue
            dv = (e.get("dirty_value") or "").strip()
            cv = (e.get("clean_value") or "").strip()
            n += 1
            if len(dv) >= 7 and dv[:2] == "99" and dv[2] in "23":
                rec = "2" + dv[3] + dv[-4:]
                if rec == cv:
                    ok += 1
                    if len(examples) < 6 and dv not in examples:
                        examples[dv] = cv
                    continue
            bad[dv[:3] if not dv.startswith("99") else dv[:2] + "x"] += 1

    res = {
        "dataset": ds.name, "split": args.split,
        "n": n, "rule_ok": ok,
        "rule_ok_rate": round(ok / n, 6) if n else None,
        "rule": "original = '2' + alias[3] + alias[-4:]",
        "bad_shapes": dict(bad.most_common(5)),
        "examples": examples,
    }
    print("=== i48 itemid 别名注入规则实测（%s / %s）===" % (ds.name, args.split))
    print("  条目数 %s" % f"{n:,}")
    print("  规则符合 %s（%.2f%%）" % (f"{ok:,}", 100.0 * ok / max(1, n)))
    if bad:
        print("  不符合的形态 top: %s" % dict(bad.most_common(5)))
    print("  样例: %s" % examples)
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print("  写出:", args.out)


if __name__ == "__main__":
    main()
