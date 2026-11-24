#!/usr/bin/env python3
"""n18_alias_rule_audit.py —— 审计 prompt 里那条 itemid 别名还原规则
「原 itemid = "2" + alias[3] + alias[-4:]」**到底对不对**。

背景：N18 原生层里「输出缺行」的格，100% 的根因是 **raw 里那一行的 itemid 是别名形态**
（992xxxxx），agent 没能还原它 ⇒ 这些格必然修不了。

查两件事：
  ① 对**全量**别名注入（test 分片的 modification_log 里登记的），
     这条规则还原出来的 itemid **是不是就是真值**？
     （prompt 里只说「能还原成一个真实存在的 itemid」—— 那不等于还原对了）
  ② 对**漏掉的那几格**，它的别名属于哪种形态、规则把它还原成了什么。

只输出聚合统计（别名形态 / 命中率），不输出病人级标识与逐格医学值。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

csv.field_size_limit(2 ** 31)


def restore(alias):
    """prompt 里的规则：原 itemid = "2" + alias[3] + alias[-4:]"""
    if len(alias) < 5:
        return None
    return "2" + alias[3] + alias[-4:]


def restore_alt(alias):
    """另一条自然规则：把前两位 99 换成 2（对 9922xxxx 成立，对 9923xxxx 不成立）"""
    if len(alias) < 5:
        return None
    return "2" + alias[2:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    ds, res, sp = Path(args.dataset), Path(args.result), args.split

    # ---------- ① 全量别名注入：规则命中率 ----------
    mod = ds / sp / "modification_log.csv"
    shape = Counter()
    rule_hit = Counter()
    n_alias = 0
    if mod.is_file():
        with mod.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("error_subtype") != "chart_itemid_alias_split":
                    continue
                if r.get("column") != "itemid":
                    continue
                n_alias += 1
                dirty = (r.get("dirty_value", "") or "").strip()
                clean = (r.get("clean_value", "") or "").strip()
                if not dirty.startswith("99"):
                    shape["非 99 前缀"] += 1
                    continue
                shape["前缀形态 " + dirty[:4]] += 1
                got = restore(dirty)
                got2 = restore_alt(dirty)
                rule_hit["prompt 规则 对"] += (got == clean)
                rule_hit["prompt 规则 错"] += (got != clean)
                rule_hit["换 99→2 规则 对"] += (got2 == clean)
                rule_hit["换 99→2 规则 错"] += (got2 != clean)
                if got == clean and got2 != clean:
                    rule_hit["只有 prompt 规则对（99→2 会错）"] += 1
                if got != clean and got2 == clean:
                    rule_hit["只有 99→2 规则对"] += 1
    print("=== ① 全量别名注入（test 分片 modification_log）===")
    print("  登记条目数: %s" % n_alias)
    for k, v in shape.most_common():
        print("    %-14s : %s" % (k, "{:,}".format(v)))
    for k, v in rule_hit.most_common():
        print("    %-26s : %s" % (k, "{:,}".format(v)))
    if n_alias:
        ok = rule_hit["prompt 规则 对"]
        print("  ⇒ prompt 规则还原**正确**的比例: %.2f%%"
              % (100.0 * ok / max(1, rule_hit["prompt 规则 对"] + rule_hit["prompt 规则 错"])))

    # ---------- ② 漏掉的那几格：别名形态 ----------
    nat = list(csv.DictReader(
        (ds / sp / "mimic_native_error_log.csv").open(newline="", encoding="utf-8")))
    agent_keys = set()
    for r in csv.DictReader((res / "icu" / "chartevents.csv").open(
            newline="", encoding="utf-8")):
        agent_keys.add((r.get("subject_id"), r.get("hadm_id"), r.get("stay_id"),
                        r.get("charttime"), r.get("itemid")))
    missing = [r for r in nat
               if (r["subject_id"], r["hadm_id"], r["stay_id"],
                   r["charttime"], r["itemid"]) not in agent_keys]
    want = {(r["subject_id"], r["hadm_id"], r["stay_id"]) for r in missing}
    rows_by_stay = {}
    with (ds / sp / "raw" / "icu" / "chartevents.csv").open(
            newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sk = (r.get("subject_id"), r.get("hadm_id"), r.get("stay_id"))
            if sk in want:
                rows_by_stay.setdefault(sk, []).append(r)

    print()
    print("=== ② 漏掉的 %d 格：别名形态与规则还原结果 ===" % len(missing))
    forms = Counter()
    verdicts = Counter()
    for r in missing:
        sk = (r["subject_id"], r["hadm_id"], r["stay_id"])
        true_it = r["itemid"]
        rows = rows_by_stay.get(sk, [])
        # 只认「还原后正好等于真值」的那一行 = 这一格对应的那条别名行
        hit = [x for x in rows
               if x.get("itemid", "").startswith("99")
               and (restore(x["itemid"]) == true_it
                    or restore_alt(x["itemid"]) == true_it)]
        if not hit:
            verdicts["该 stay 里找不到对应真值的别名行"] += 1
            continue
        for x in hit:
            a = x["itemid"]
            forms["前缀形态 " + a[:4]] += 1
            got, got2 = restore(a), restore_alt(a)
            verdicts["prompt 规则(2+a[3]+a[-4:]) 还原 == 真值"] += (got == true_it)
            verdicts["prompt 规则 还原 != 真值"] += (got != true_it)
            verdicts["换 99→2 规则 还原 == 真值"] += (got2 == true_it)
    for k, v in forms.most_common():
        print("    %-14s : %s" % (k, "{:,}".format(v)))
    for k, v in verdicts.most_common():
        print("    %-38s : %s" % (k, "{:,}".format(v)))


if __name__ == "__main__":
    main()
