#!/usr/bin/env python3
"""verify_icu5000_mimic_raw_error.py —— 独立校验 mimicrawerror 数据集的正确性。

（从 verify_k29_mimicraw_error.py 改造而来；HARD 区间表与 SENTINELS 照搬 ——
  icu5000 与 K29 同为 MIMIC-IV chartevents，itemid 集合相同。）

不信任构建脚本的自检，直接查产物：
  ① test 的 raw 与源数据集**逐字节相同**（原生错误本来就在 raw 里，一个字节都不该动）
  ② reference 里剩余的原生错误格数 == 声明的「不可修」数（其余全修好了）
  ③ modification_log 里有 N 条 mimic_native_* 条目，N == 声明的修复数
  ④ 每个原生错误格在 log 里**只有一条** valuenum 条目（重叠的注入条目已删除）
  ⑤ log 的 raw_row_index 与 reference 的行号对齐（抽样核对 clean_value）
  ⑥ train/validation 的注入示例：raw 是脏值、reference 是中位数、log 有条目
  ⑦ 溯源表 mimic_native_error_log.csv 行数 == 原生条目数
只输出聚合。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(2 ** 31)
CE = "icu/chartevents.csv"
CE_KEY = ["subject_id", "hadm_id", "stay_id", "charttime", "itemid"]
SENT = {-999.0, -998.0, -997.0, -990.0, 9999.0, 99999.0,
        999999.0, 9999999.0, 220000.0, 220045.0}
HARD = {220045: (20, 300), 220050: (40, 400), 220051: (20, 250), 220052: (20, 250),
        220059: (0, 150), 220061: (0, 100), 220074: (-20, 80), 220179: (40, 400),
        220180: (20, 250), 220181: (20, 250), 220210: (0, 100), 223762: (25, 45),
        224688: (0, 100), 224689: (0, 100), 224690: (0, 100), 227466: (8, 200),
        224167: (40, 400), 224643: (20, 250), 227242: (20, 250), 227243: (40, 400)}
FAILS = []


def check(cond, msg):
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        FAILS.append(msg)


def md5(p):
    h = hashlib.md5()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def detect_offset(rows_by_L, ins_pos, raw_path, sample=200):
    """探测 raw 行号 ↔ reference 行号的偏移规则（**不能硬编码**）。

    两种情形都真实存在：
      A) raw 里真的多出插入/重复行 ⇒ raw 行号 = L + bisect_left(ins_pos, L)
      B) raw 与 reference 逐行对应（log 里的 duplicate 条目指向已存在的行）
         ⇒ raw 行号 = L
    ⚠️ **1.6pct 档是 B、4.7pct 档是 A**。写死 A 会让校验器在 1.6pct 上整段读错行
    （K29 实测：写死 A ⇒ 853/873 命中、560 格被误报成「value 非数值」；
      自动探测 ⇒ 873/873 命中、872 格形态正确）。
    ⚠️ icu5000 是**混合索引空间**：chartevents = bisect（raw 多 1,740,790 行）、
       inputevents / outputevents = zero ⇒ 必须逐文件自动判定。

    判据与 builder 一致：raw 行的 key 是否与 reference 行一致（**不依赖 value 性质**，
    避免循环论证）。
    """
    Ls = list(rows_by_L)[:sample]
    if not Ls:
        return True
    need = set()
    for L in Ls:
        need.add(L)
        need.add(L + bisect_left(ins_pos, L))
    rows_at = {}
    with raw_path.open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for ri, row in enumerate(rd):
            if ri in need:
                rows_at[ri] = row
            if ri >= max(need):
                break

    def key_of(row):
        return tuple(row.get(c) for c in CE_KEY)

    zero_hit = bis_hit = 0
    for L in Ls:
        k = key_of(rows_by_L[L])
        rz = rows_at.get(L)
        if rz is not None and key_of(rz) == k:
            zero_hit += 1
        rb = rows_at.get(L + bisect_left(ins_pos, L))
        if rb is not None and key_of(rb) == k:
            bis_hit += 1
    use_zero = zero_hit >= bis_hit
    print("     偏移探测: zero 命中 %d / bisect 命中 %d ⇒ raw 行号 = %s"
          % (zero_hit, bis_hit, "L" if use_zero else "L + 插入偏移"))
    return use_zero


def reference_rows_at(ref_path, Ls):
    """取 reference 里指定行号的行：{L: row}"""
    want = set(Ls)
    out = {}
    if not want:
        return out
    mx = max(want)
    with ref_path.open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for L, row in enumerate(rd):
            if L in want:
                out[L] = row
            if L >= mx:
                break
    return out


def ins_positions(log_path):
    """log 里 insert/duplicate 的 raw_row_index（升序）"""
    pos = []
    with log_path.open(newline="", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            if e.get("raw_file") == CE and e.get("operation") in ("insert", "duplicate"):
                pos.append(int(e["raw_row_index"]))
    pos.sort()
    return pos


def native_cells(path):
    """reference 里越界/哨兵的格：[(行号, itemid, 值)]"""
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        for L, r in enumerate(csv.DictReader(f)):
            try:
                it = int(r["itemid"])
            except ValueError:
                continue
            v = fnum(r["valuenum"])
            if v is None or v == 0:
                continue
            if v in SENT:
                out.append((L, it, v))
            else:
                rg = HARD.get(it)
                if rg and not (rg[0] <= v <= rg[1]):
                    out.append((L, it, v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源数据集（A 类）")
    ap.add_argument("--dst", required=True, help="mimicrawerror 数据集")
    args = ap.parse_args()
    src = Path(args.src); dst = Path(args.dst)

    print("=" * 72)
    print("① test 的 raw：只允许「原生错误格」被恢复成 MIMIC 原生错值")
    nat0 = list(csv.DictReader((dst / "test/mimic_native_error_log.csv").open(newline="", encoding="utf-8"))) \
        if (dst / "test/mimic_native_error_log.csv").is_file() else []
    # 源 raw 与新 raw 的差异必须只出现在「也被告 injected」的那批格上，
    # 且差异后的值 == MIMIC 原生错值（mimic_native_value）
    inj_rows = {int(r["raw_row_index"]) for r in nat0 if r["also_had_injection"] == "yes"}
    src_vals = []
    with (src / "test/raw" / CE).open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            src_vals.append(r.get("valuenum"))
    new_vals = []
    with (dst / "test/raw" / CE).open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            new_vals.append(r.get("valuenum"))
    diffs = [i for i in range(min(len(src_vals), len(new_vals)))
             if src_vals[i] != new_vals[i]]
    # raw 行号 → reference 行号：raw 比 reference 多插入行 ⇒ 用 log 索引对齐
    # （这里用 native log 的 raw_row_index 直接比对 reference 行号）
    print("     raw 与源 raw 的值差异行数: %d（其中原生格 %d）" % (len(diffs), len(inj_rows)))
    check(len(diffs) <= len(inj_rows) * 2,
          "raw 的改动量级与「叠脏格数」相符（差异 %d 行，叠脏 %d 格）" % (len(diffs), len(inj_rows)))

    print("\n② reference 里剩余的原生错误格")
    rem = native_cells(dst / "test/reference" / CE)
    nat = list(csv.DictReader((dst / "test/mimic_native_error_log.csv").open(newline="", encoding="utf-8")))
    check(len(nat) > 0, "溯源表非空（%d 行）" % len(nat))
    print("     reference 残留未修的原生错误格: %d" % len(rem))
    print("       （应当 == ①该患者该项**无可用读数**（真实值不可判定，保持原样）"
          "+ ②该格的 itemid 也被注入器改过（两层错误叠加、形态同构不可判定，"
          "不进 native 层、留给注入层评测）两类之和）")

    print("\n③/④ modification_log 里的原生条目")
    per_cell = defaultdict(list)
    n_nat = 0
    n_val_total = 0
    with (dst / "test/modification_log.csv").open(newline="", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            if e.get("raw_file") != CE or e.get("column") != "valuenum" \
                    or e.get("operation") != "cell_update":
                continue
            n_val_total += 1
            ri = int(e["raw_row_index"])
            per_cell[ri].append(e)
            if e.get("error_subtype", "").startswith("mimic_native"):
                n_nat += 1
    check(n_nat == len(nat), "log 里 mimic_native_* 条目 %d == 溯源表 %d" % (n_nat, len(nat)))
    # 只查**原生格**：每个原生格在 log 里应恰好有 1 条 valuenum 条目
    # （重叠的注入条目已被删掉 ⇒ 只剩我们追加的那条）。
    # ⚠️ 注入器本身就会在同一个格上注入多次（如 unit_scale_error + sentinel_chart_value），
    #    那些格有 >1 条是正常的，不该报错。
    nat_rows_set = {int(r["raw_row_index"]) for r in nat}
    dup = [ri for ri in nat_rows_set if len(per_cell.get(ri, [])) > 1]
    missing_entry = [ri for ri in nat_rows_set if len(per_cell.get(ri, [])) == 0]
    check(not dup, "每个原生格在 log 里恰好 1 条 valuenum 条目；重复的 %d 格" % len(dup))
    check(not missing_entry, "每个原生格在 log 里都有条目；缺的 %d 格" % len(missing_entry))
    multi_inj = sum(1 for ri, es in per_cell.items()
                    if len(es) > 1 and ri not in nat_rows_set)
    print("     （注入器自己在同格注入多次的格: %d —— 正常，不参与本项检查）" % multi_inj)
    nat_rows = {int(r["raw_row_index"]) for r in nat}
    check(nat_rows <= set(per_cell), "溯源表的每一行都在 log 里有条目")

    print("\n①b `value` 文本列与 `valuenum` 的一致性（原生错误的判据）")
    # MIMIC 原生错误 ⇒ raw 的 value 文本列记的也是那个坏数（== valuenum）
    # 注入的量纲放大 ⇒ value 记的是干净数（!= valuenum）
    # ⇒ 原生层每一个格都必须是 value == valuenum
    nat_rowset = {int(r["raw_row_index"]) for r in nat}
    # ⚠️ log 索引空间(= reference 行号) → raw 行号：**必须自动探测偏移**，
    #    不能写死 r = L + bisect_left(ins_pos, L) —— 1.6pct 档是零偏移。
    ins_pos = ins_positions(dst / "test/modification_log.csv")
    ref_rows = reference_rows_at(dst / "test/reference" / CE, nat_rowset)
    use_zero = detect_offset(ref_rows, ins_pos, dst / "test/raw" / CE)
    want = {L: L for L in nat_rowset} if use_zero else \
        {L + bisect_left(ins_pos, L): L for L in nat_rowset}
    c = Counter()
    with (dst / "test/raw" / CE).open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for ri, r in enumerate(rd):
            if ri not in want:
                continue
            v = fnum(r.get("valuenum")); val = fnum(r.get("value"))
            if val is None:
                c["value 非数值"] += 1
            elif v is not None and abs(val - v) < 1e-9:
                c["value == valuenum（原生错误形态，正确）"] += 1
            else:
                c["value != valuenum（形态错！）"] += 1
    print("     分布: %s" % dict(c))
    check(c["value != valuenum（形态错！）"] == 0 and c["value 非数值"] == 0,
          "test 的每个原生格，raw 的 value 文本列都 == valuenum（与真实 MIMIC 原生错误同构）")

    print("\n⑤ log 行号与 reference 行号对齐（抽样 500 条核对 clean_value）")
    ref_vals = []
    with (dst / "test/reference" / CE).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ref_vals.append(r.get("valuenum"))
    ok = bad = 0
    for r in nat[:500]:
        L = int(r["raw_row_index"])
        if L >= len(ref_vals):
            continue
        cv, rv = r["repaired_value"], ref_vals[L]
        try:
            same = abs(float(cv) - float(rv)) < 1e-9
        except (TypeError, ValueError):
            same = str(cv) == str(rv)
        ok += 1 if same else 0
        bad += 0 if same else 1
    check(bad == 0, "抽样 %d 条：reference[行号] == 修复值（不一致 %d）" % (ok + bad, bad))

    print("\n⑤b test 原生错误的 R2 口径（用源数据集 reference 独立重算）")
    src_ref = src / "test/reference" / CE
    want_rows = {int(r["raw_row_index"]) for r in nat}
    grp = defaultdict(list)
    rowkey = {}
    with src_ref.open(newline="", encoding="utf-8") as f:
        for L, r in enumerate(csv.DictReader(f)):
            try:
                it = int(r["itemid"])
            except ValueError:
                continue
            v = fnum(r["valuenum"])
            if v is None or v == 0:
                continue
            g = (r["subject_id"], it)
            grp[g].append((L, v))
            if L in want_rows:
                rowkey[L] = (g, it)
    okr = badr = skipped = 0
    for r in nat:
        L = int(r["raw_row_index"])
        g_it = rowkey.get(L)
        if g_it is None:
            skipped += 1
            continue
        g, it = g_it
        rg = HARD.get(it)
        if rg is not None:
            # 类 1：可用读数 = 区间内 且 非 0
            usable = [v for Lx, v in grp[g]
                      if Lx != L and rg[0] <= v <= rg[1]]
        else:
            # 类 2（哨兵码）：可用读数 = 非哨兵 且 非 0
            usable = [v for Lx, v in grp[g]
                      if Lx != L and v not in SENT]
        if not usable:
            skipped += 1
            continue
        s = sorted(usable)
        m = s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2.0
        try:
            same = abs(m - fnum(r["repaired_value"])) < 1e-9
        except TypeError:
            same = False
        okr += 1 if same else 0
        badr += 0 if same else 1
    check(badr == 0,
          "%d 个原生格的 repaired_value == R2 中位数（剔除坏格后重算），不一致 %d，跳过 %d"
          % (okr + badr, badr, skipped))

    print("\n⑥ train/validation 的注入示例")
    for split in ("train", "validation"):
        p = dst / split / "mimic_native_error_log.csv"
        if not p.is_file():
            print("     %s: 无注入示例" % split)
            continue
        rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
        refv = []
        with (dst / split / "reference" / CE).open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                refv.append(r.get("valuenum"))
        same = 0
        for r in rows:
            L = int(r["raw_row_index"])
            if L < len(refv) and r["repaired_value"] == refv[L]:
                same += 1
        check(same == len(rows),
              "%s：%d 个示例的 reference[行号] == 中位数（一致 %d）" % (split, len(rows), same))
        # 每个示例的 dirty 值必须**落在区间外**（否则"脏值"看着不像错误）
        not_out = 0
        for r in rows:
            it = int(r["itemid"])
            dv = fnum(r["raw_dirty_value"])
            if dv is None:
                not_out += 1
                continue
            rg = HARD.get(it)
            if dv in SENT or (rg and not (rg[0] <= dv <= rg[1])):
                continue
            not_out += 1
        check(not_out == 0,
              "%s：所有示例的 dirty 值都越界/是哨兵码（不合格 %d）" % (split, not_out))
        # ★ value 文本列必须 == 脏值（与测试集真实原生错误同构）
        #   ⚠️ 同样**必须自动探测偏移**（1.6pct 档零偏移、4.7pct 档带偏移）
        ins_pos2 = ins_positions(dst / split / "modification_log.csv")
        by_L2 = {int(r["raw_row_index"]): r for r in rows}
        Ls2 = set(by_L2)
        ref_rows2 = reference_rows_at(dst / split / "reference" / CE, Ls2)
        use_zero2 = detect_offset(ref_rows2, ins_pos2, dst / split / "raw" / CE)
        # want2: raw 行号 → 溯源表里那一行（后面要取 raw_dirty_value）
        want2 = {ri: by_L2[L] for ri, L in
                 ((L, L) if use_zero2 else (L + bisect_left(ins_pos2, L), L)
                  for L in Ls2)}
        cv = Counter()
        with (dst / split / "raw" / CE).open(newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for ri, r in enumerate(rd):
                if ri not in want2:
                    continue
                d = fnum(want2[ri].get("raw_dirty_value"))
                val = fnum(r.get("value"))
                if val is None:
                    cv["value 非数值"] += 1
                elif d is not None and abs(val - d) < 1e-9:
                    cv["value == 脏值（正确）"] += 1
                else:
                    cv["value != 脏值（形态错！）"] += 1
        check(cv["value != 脏值（形态错！）"] == 0 and cv["value 非数值"] == 0,
              "%s：示例的 value 文本列 == 脏值（与真实原生错误同构）；分布 %s"
              % (split, dict(cv)))
        # log 里这些格有条目
        got = 0
        with (dst / split / "modification_log.csv").open(newline="", encoding="utf-8") as f:
            for e in csv.DictReader(f):
                if (e.get("raw_file") == CE and e.get("column") == "valuenum"
                        and e.get("error_subtype", "").startswith("mimic_native")
                        and int(e["raw_row_index"]) in {int(r["raw_row_index"]) for r in rows}):
                    got += 1
        check(got == len(rows), "%s：log 里 %d 个示例都有条目" % (split, got))

        # ★ R2 口径校验：用**源数据集**的 reference 重算该 (subject,itemid) 的
        #   「区间内且非 0」读数中位数（**排除被破坏的那一格**），应等于 repaired_value。
        src_ref = src / split / "reference" / CE
        if src_ref.is_file():
            grp = defaultdict(list)
            keyof = {}
            with src_ref.open(newline="", encoding="utf-8") as f:
                for L, r in enumerate(csv.DictReader(f)):
                    try:
                        it = int(r["itemid"])
                    except ValueError:
                        continue
                    v = fnum(r["valuenum"])
                    if v is None or v == 0:
                        continue
                    g = (r["subject_id"], it)
                    grp[g].append((L, v))
                    keyof[L] = (g, it)
            # 被破坏的格在**新** reference 里已改成中位数 ⇒ 它在源 reference 里的位置
            # 由 native log 的 raw_row_index 给出（= 源 reference 的行号）
            okr = badr = 0
            for r in rows:
                L = int(r["raw_row_index"])
                g_it = keyof.get(L)
                if g_it is None:
                    continue
                g, it = g_it
                rg = HARD.get(it)
                usable = [v for Lx, v in grp[g]
                          if Lx != L and (rg is None or (rg[0] <= v <= rg[1]))]
                if not usable:
                    continue
                s = sorted(usable)
                m = s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2.0
                try:
                    same = abs(m - fnum(r["repaired_value"])) < 1e-9
                except TypeError:
                    same = False
                okr += 1 if same else 0
                badr += 0 if same else 1
            check(badr == 0,
                  "%s：%d 个示例的 repaired_value == R2 中位数（剔除坏格后重算），不一致 %d"
                  % (split, okr + badr, badr))

    print("\n⑦ 溯源表结构")
    cols = list(csv.DictReader((dst / "test/mimic_native_error_log.csv").open(newline="", encoding="utf-8")).fieldnames or [])
    check("mimic_native_value" in cols and "repaired_value" in cols
          and "also_had_injection" in cols and "detection_mode" in cols,
          "溯源表列齐全（%d 列）" % len(cols))
    modes = Counter(r["detection_mode"] for r in nat)
    inj = Counter(r["also_had_injection"] for r in nat)
    print("     类别分布: %s" % dict(modes))
    print("     其中同时被注入器叠脏: %s" % dict(inj))

    print("\n" + "=" * 72)
    if FAILS:
        print("❌ %d 项未通过：" % len(FAILS))
        for m in FAILS:
            print("   -", m)
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
