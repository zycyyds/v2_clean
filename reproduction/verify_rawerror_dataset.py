#!/usr/bin/env python3
"""校验注入错误数据集。

10 项检查：
  1. 日志可重放      拿干净 baseline 重新 apply 一遍日志，与成品 raw 逐格比对
  2. 计数自洽        actual == cell_update + duplicate + insert
  3. 无自然噪声      error_class 只出现 class1/2/3/4，0 条 class0_baseline_gap
  4. 18 型全覆盖
  5. 错误率达标      actual / 全部 raw cells 落在目标 ±0.05pt
  6. 干净模态零污染   raw/ed/ raw/ecg/ raw/note/ 的 md5 与 baseline 相同
  7. 索引位移语义    chartevents 上按位移公式能定位到脏值
  8. 退化注入已修    future_window_leakage 是真实未来时间；stay_hadm_mismatch 是别的 hadm_id
  9. demographics 是 cohort 切片（不是全库主表）
 10. K29 未被触碰

用法:
    python3 verify_rawerror_dataset.py <dataset_key> [<split> ...]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import sys
import tempfile
from array import array
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
csv.field_size_limit(sys.maxsize)

ROOT = Path("/Users/mac/PycharmProjects/v2_clean")
CLEAN_ROOT = ROOT / "datasets/常规数据集"
OUT_ROOT = ROOT / "datasets/注入错误数据集"

CELL_RATES = {
    "4.7pct": {"train": 0.0233, "validation": 0.0089, "test": 0.0472},
    "1.6pct": {"train": 0.0130, "validation": 0.0224, "test": 0.0158},
    # 低密度档：绝对条数 400/800/1800，换算成率（分母 = 各 split 干净基线的
    # 全部 raw cell 数，实测 K29：112,984 / 728,158 / 265,583,987）。
    "1800": {"train": 400 / 112984, "validation": 800 / 728158, "test": 1800 / 265583987},
}
# 低密度档的目标**绝对条数**（错误率在 1e-5 量级时，±0.0005 的率容差形同虚设，
# 必须直接卡条数）
TARGET_COUNTS = {
    "1800": {"train": 400, "validation": 800, "test": 1800},
}
INJECTED_TABLES = [
    "hosp/patients.csv", "hosp/admissions.csv", "hosp/diagnoses_icd.csv",
    "icu/icustays.csv", "icu/chartevents.csv", "icu/inputevents.csv",
    "icu/outputevents.csv",
]
CLEAN_MODALITIES = {"n18": ["ed", "note"], "i48": ["ecg"], "icu5000": ["ecg", "note"]}

DATASETS = {
    "n18": ("N18_ckd_ED_notes_eval_10_20_rest",
            "N18_ckd_ED_v2_rawerror_test{tag}_490M",
            {"train": "train", "validation": "val", "test": "test"}),
    "i48": ("I48_atrial_fibrillation_ECG_eval_10_20_rest",
            "I48_atrial_fibrillation_v2_rawerror_test{tag}_411M",
            {"train": "train", "validation": "val", "test": "test"}),
    "icu5000": ("icu_5000_clean_eval_10_20_rest",
                 "icu_5000_v2_rawerror_test{tag}_614M",
                 {"train": "train", "validation": "val", "test": "test"}),
    "k29": ("K29_clean_eval_10_20_rest",
            "K29_clean_v2_rawerror_10train_400_800_{tag}",
            {"train": "train", "validation": "val", "test": "test"}),
}


def open_any(p: Path, mode="r"):
    if str(p).endswith(".gz"):
        return gzip.open(p, mode + "t", encoding="utf-8", newline="")
    return open(p, mode, encoding="utf-8", newline="")


def md5(p: Path) -> str:
    return md5_bytes(p)


def md5_bytes(p: Path) -> str:
    """对解压后的内容算 md5（跨 .gz / 明文可比）。必须用二进制模式。"""
    h = hashlib.md5()
    if str(p).endswith(".gz"):
        fh = gzip.open(p, "rb")
    else:
        fh = open(p, "rb")
    with fh as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_cells(raw_dir: Path) -> int:
    total = 0
    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file() or p.name == ".DS_Store":
            continue
        if not (p.name.endswith(".csv") or p.name.endswith(".csv.gz")):
            continue
        with open_any(p) as f:
            rd = csv.reader(f)
            cols = next(rd, [])
            total += len(cols) * sum(1 for _ in rd)
    return total


def scan_log(log_path: Path, sample_per_file: int = 5, special_cap: int = 100):
    """流式扫 modification_log。

    test split 的日志有 2,500 万条 —— 全量 list[dict] 要 15 GB，直接 OOM。
    这里只留四样东西：
      - 四个计数器
      - 每个文件的 insert/duplicate 行号（array('q')，给位移公式用）
      - 每个文件的前 sample_per_file 条 cell_update（给检查 7 用）
      - future_window_leakage / stay_hadm_mismatch 的前若干条（给检查 8 用）
    """
    by_subtype, by_class, by_op, by_file = (Counter() for _ in range(4))
    pos = defaultdict(lambda: array("q"))
    samples = defaultdict(list)
    special: dict[str, list] = {}
    total = 0
    with log_path.open(newline="", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            total += 1
            rel = e["raw_file"]
            by_subtype[e["error_subtype"]] += 1
            by_class[e["error_class"]] += 1
            by_op[e["operation"]] += 1
            by_file[rel] += 1
            try:
                idx = int(e["raw_row_index"])
            except (ValueError, TypeError):
                continue
            op = e["operation"]
            if op in ("insert", "duplicate"):
                pos[rel].append(idx)
            else:
                if len(samples[rel]) < sample_per_file:
                    samples[rel].append(e)
            if e["error_subtype"] in ("future_window_leakage", "stay_hadm_mismatch"):
                # 每型单独限额 —— 日志按 subtype 成段排，一个总额度会被
                # 先出现的那个型吃光，另一个型一条都留不下
                bucket = special.setdefault(e["error_subtype"], [])
                if len(bucket) < special_cap:
                    bucket.append(e)
    # array 没有 sort()，先 list 排序再转回 array（8 字节/个）
    for rel in list(pos):
        pos[rel] = array("q", sorted(pos[rel]))
    return {"total": total, "by_subtype": by_subtype, "by_class": by_class,
            "by_op": by_op, "by_file": by_file, "pos": pos,
            "samples": samples, "special": special}


def probe_rows(raw_file: Path, wanted: set[int], columns: list[str]):
    """流式读 csv，只抓 wanted 里的行号。

    返回 (header, {行号: {列: 值}})。整表读进 list[list] 在 chartevents 上
    要 6 GB 内存 —— 改成只留需要的行。
    """
    header: list[str] = []
    out: dict[int, dict[str, str]] = {}
    if not wanted:
        return header, out
    with open_any(raw_file) as f:
        rd = csv.reader(f)
        header = next(rd, [])
        idxs = {c: header.index(c) for c in columns if c in header}
        for i, row in enumerate(rd):
            if i in wanted:
                out[i] = {c: row[j] for c, j in idxs.items() if j < len(row)}
                if len(out) == len(wanted):
                    break
    return header, out


def check_split(key: str, split: str, clean_name: str, out_name: str, sd,
                 rates: dict, preset: str = "4.7pct") -> dict:
    res = {}
    clean_split = CLEAN_ROOT / clean_name / sd[split]
    out_split = OUT_ROOT / out_name / split
    raw_src = clean_split / "raw"
    raw_out = out_split / "raw"
    log_path = out_split / "modification_log.csv"

    print("\n" + "=" * 88)
    print("### %s / %s" % (key, split))
    print("=" * 88)

    if not log_path.is_file():
        print("  [FAIL] 日志不存在")
        return {"ok": False}

    scan = scan_log(log_path)
    by_subtype = scan["by_subtype"]
    by_class = scan["by_class"]
    by_op = scan["by_op"]
    by_file = scan["by_file"]
    pos_by_file = scan["pos"]
    samples = scan["samples"]
    special = scan["special"]
    total = scan["total"]

    # ---- 2. 计数自洽
    n_cu, n_dup, n_ins = by_op["cell_update"], by_op["duplicate"], by_op["insert"]
    res["2_count_selfconsistent"] = (n_cu + n_dup + n_ins == total)
    print("  [2] 计数自洽: cell_update=%s dup=%s insert=%s 合计=%s vs 日志 %s  %s"
          % ("{:,}".format(n_cu), "{: ,}".format(n_dup) if False else "{:,}".format(n_dup),
             "{:,}".format(n_ins), "{:,}".format(n_cu + n_dup + n_ins),
             "{:,}".format(total),
             "OK" if res["2_count_selfconsistent"] else "FAIL"))

    # ---- 3. 无自然噪声
    n_class0 = by_class.get("class0_baseline_gap", 0)
    res["3_no_natural_noise"] = (n_class0 == 0)
    print("  [3] 无自然噪声: class0_baseline_gap = %s  %s"
          % ("{:,}".format(n_class0), "OK" if res["3_no_natural_noise"] else "FAIL"))
    print("      error_class 分布: %s" % dict(by_class))

    # ---- 4. 18 型全覆盖
    res["4_18_subtypes"] = (len(by_subtype) == 18)
    print("  [4] subtype 覆盖: %d/18  %s" % (len(by_subtype),
                                            "OK" if res["4_18_subtypes"] else "FAIL"))

    # ---- 5. 错误率
    cells = count_cells(raw_out)
    rate = total / cells if cells else 0
    want = rates[split]
    res["5_rate"] = abs(rate - want) <= 0.0005
    print("  [5] 错误率: %d / %s = %.4f%%  (目标 %.2f%%)  %s"
          % (total, "{:,}".format(cells), rate * 100, want * 100,
             "OK" if res["5_rate"] else "FAIL"))

    # ---- 5b. 绝对条数（低密度档必备：率在 1e-5 量级时，±0.0005 的率容差形同虚设）
    want_n = TARGET_COUNTS.get(preset, {}).get(split)
    if want_n is not None:
        res["5b_target_count"] = (total == want_n)
        print("  [5b] 绝对条数: %d  (目标 %d)  %s"
              % (total, want_n, "OK" if total == want_n else "FAIL"))

    # ---- 6. 干净模态零污染
    # 有 CLEAN_MODALITIES 的（n18/i48/icu5000）：整个 modality 目录必须逐字节相同。
    # 没有的（K29）：raw 是 28 张 MIMIC 源表铺在一棵树上，没有「干净的模态目录」，
    # 干净的表是**除 7 张被注入表以外的所有表** —— 逐文件 md5 比对。
    clean_ok = True
    mods = CLEAN_MODALITIES.get(key, [])
    if mods:
        for mod in mods:
            src_dir = raw_src / mod
            dst_dir = raw_out / mod
            if not src_dir.is_dir():
                continue
            for p in sorted(src_dir.rglob("*")):
                if not p.is_file() or p.name == ".DS_Store":
                    continue
                rel = p.relative_to(src_dir)
                q = dst_dir / rel
                if not q.is_file():
                    q = dst_dir / rel.with_suffix("")
                if not q.is_file() or md5_bytes(p) != md5_bytes(q):
                    clean_ok = False
                    print("      !! %s/%s 内容不一致" % (mod, rel))
    else:
        n_clean = 0
        for p in sorted(raw_src.rglob("*")):
            if not p.is_file() or p.name == ".DS_Store":
                continue
            rel = p.relative_to(raw_src)
            if rel.as_posix() in INJECTED_TABLES:
                continue
            n_clean += 1
            q = raw_out / rel
            if not q.is_file() or md5_bytes(p) != md5_bytes(q):
                clean_ok = False
                print("      !! %s 内容不一致" % rel)
        print("      （比对 %d 张未被注入的表）" % n_clean)
    res["6_clean_modalities"] = clean_ok
    print("  [6] 干净模态零污染 (%s): %s"
          % (",".join(mods) or "所有未被注入的表",
             "OK" if clean_ok else "FAIL"))

    # ---- 7. 索引位移语义
    # 位移公式（评估器口径）：raw_idx = log_idx + bisect_left(pos, log_idx)
    # pos 是该文件内 insert/duplicate 的 raw_row_index【带重复的】升序列表
    shift_ok = True
    for rel in INJECTED_TABLES:
        s_entries = samples.get(rel) or []
        if not s_entries:
            continue
        p = raw_out / rel
        if not p.is_file():
            shift_ok = False
            continue
        pl = pos_by_file.get(rel) or array("q")
        wanted: dict[int, dict] = {}
        cols: set[str] = set()
        for e in s_entries:
            li = int(e["raw_row_index"])
            wanted[li + bisect_left(pl, li)] = e
            cols.add(e["column"])
        header, got = probe_rows(p, set(wanted), sorted(cols))
        for idx, e in wanted.items():
            col = e["column"]
            if col not in header:
                continue
            if idx not in got:
                shift_ok = False
                print("      !! %s 目标行 %d 不存在" % (rel, idx))
                continue
            v = got[idx].get(col, "")
            if v != e["dirty_value"]:
                shift_ok = False
                print("      !! %s 行 %d 列 %s: 期望 %r 实得 %r"
                      % (rel, int(e["raw_row_index"]), col,
                         e["dirty_value"][:30], v[:30]))
    res["7_index_shift"] = shift_ok
    print("  [7] 索引位移语义: %s" % ("OK" if shift_ok else "FAIL"))

    # ---- 8. 退化注入已修：直接看 raw 里那个 cell 变成什么了
    fw = special.get("future_window_leakage", [])
    sh = special.get("stay_hadm_mismatch", [])
    fw_ok = bool(fw)
    for e in fw[:50]:
        dv = e["dirty_value"]
        # 必须是真实未来时间戳，不能是空串 / 字面量 / epoch
        if not dv or len(dv) < 19 or dv[4] != "-" or dv.startswith("1970"):
            fw_ok = False
            break
    # 再抽查 raw 里 charttime 确实变成了那个未来时间
    fw_raw_ok = fw_ok
    if fw:
        rel = fw[0]["raw_file"]
        p = raw_out / rel
        pl = pos_by_file.get(rel) or array("q")
        wanted = {}
        for e in fw[:20]:
            li = int(e["raw_row_index"])
            wanted[li + bisect_left(pl, li)] = e
        header, got = probe_rows(p, set(wanted), ["charttime"])
        for idx, e in wanted.items():
            if "charttime" not in header:
                continue
            if idx not in got or got[idx].get("charttime", "") != e["dirty_value"]:
                fw_raw_ok = False
                break
    sh_ok = bool(sh)
    for e in sh[:50]:
        dv = e["dirty_value"]
        if not dv or dv == "(error generating)" or len(dv) < 5 or not dv.isdigit():
            sh_ok = False
            break
    # raw 里 hadm_id 确实变了
    sh_raw_ok = sh_ok
    if sh:
        rel = sh[0]["raw_file"]
        p = raw_out / rel
        pl = pos_by_file.get(rel) or array("q")
        wanted = {}
        for e in sh[:20]:
            li = int(e["raw_row_index"])
            wanted[li + bisect_left(pl, li)] = e
        header, got = probe_rows(p, set(wanted), ["hadm_id"])
        for idx, e in wanted.items():
            if "hadm_id" not in header:
                continue
            if idx not in got or got[idx].get("hadm_id", "") != e["dirty_value"]:
                sh_raw_ok = False
                break
    res["8_degraded_fixed"] = fw_ok and fw_raw_ok and sh_ok and sh_raw_ok
    print("  [8] 退化注入已修:")
    print("      future_window_leakage=%s  值=%r  raw 一致=%s  %s"
          % ("{:,}".format(by_subtype.get("future_window_leakage", 0)),
             fw[0]["dirty_value"] if fw else None,
             fw_raw_ok, "OK" if (fw_ok and fw_raw_ok) else "FAIL"))
    print("      stay_hadm_mismatch=%s  值=%r  raw 一致=%s  %s"
          % ("{:,}".format(by_subtype.get("stay_hadm_mismatch", 0)),
             sh[0]["dirty_value"] if sh else None,
             sh_raw_ok, "OK" if (sh_ok and sh_raw_ok) else "FAIL"))

    # ---- 9. demographics 是 cohort 切片
    # 判据要跟**干净 baseline** 对齐。ICU5000 干净版 test 自己就缺 6 个
    # subject（cohort 4,983 vs demographics 4,977）—— 那是上游建数据集时的
    # 缺口，不是注入造成的。所以这里查两件事：
    #   a) 注入产物的 reference/demographics.csv 与干净版 md5 相同（我们没碰过）
    #   b) cohort 与 demographics 的行数差跟干净版**一样**（没引入新缺口）
    demo = out_split / "reference/demographics.csv"
    cohort = out_split / "reference/cohort.csv"
    clean_ref = CLEAN_ROOT / clean_name / sd[split] / "reference"
    demo_ok = False
    detail = ""
    if demo.is_file() and cohort.is_file():
        with cohort.open(newline="", encoding="utf-8") as f:
            n_subj = len({(r.get("subject_id") or "").strip()
                          for r in csv.DictReader(f) if (r.get("subject_id") or "").strip()})
        with demo.open(newline="", encoding="utf-8") as f:
            n_demo = sum(1 for _ in csv.DictReader(f))
        clean_demo = clean_ref / "demographics.csv"
        clean_cohort = clean_ref / "cohort.csv"
        n_demo_clean = n_subj_clean = None
        if clean_demo.is_file():
            with clean_demo.open(newline="", encoding="utf-8") as f:
                n_demo_clean = sum(1 for _ in csv.DictReader(f))
        if clean_cohort.is_file():
            with clean_cohort.open(newline="", encoding="utf-8") as f:
                n_subj_clean = len({(r.get("subject_id") or "").strip()
                                    for r in csv.DictReader(f)
                                    if (r.get("subject_id") or "").strip()})
        same_md5 = clean_demo.is_file() and md5_bytes(clean_demo) == md5_bytes(demo)
        same_gap = (n_demo_clean is not None and n_subj_clean is not None
                    and (n_subj - n_demo) == (n_subj_clean - n_demo_clean))
        demo_ok = same_md5 and same_gap
        detail = ("干净版 cohort %s / demographics %s（差 %s）→ 注入版差 %s%s"
                  % ("{:,}".format(n_subj_clean or 0), "{:,}".format(n_demo_clean or 0),
                     (n_subj_clean - n_demo_clean) if n_demo_clean is not None else "?",
                     n_subj - n_demo, "（与干净版一致）" if same_gap else "（与干净版不一致！）"))
        print("  [9] demographics cohort 切片: %s 行 vs cohort %s 个 subject  %s"
              % ("{:,}".format(n_demo), "{:,}".format(n_subj),
                 "OK" if demo_ok else "FAIL"))
        print("      %s" % detail)
        print("      reference/demographics.csv 与干净版 md5 %s"
              % ("相同（未被动过）" if same_md5 else "不同 ← 被改过！"))
    res["9_demographics_slice"] = demo_ok

    # ---- 1. 日志可重放
    replay_ok = True
    replay_detail = ""
    with tempfile.TemporaryDirectory(dir="/Users/mac/.claude/jobs/910902cd/tmp") as td:
        tmp_raw = Path(td) / "raw"
        tmp_raw.mkdir(parents=True)
        for rel in INJECTED_TABLES:
            src = raw_src / rel
            if not src.is_file():
                src = raw_src / (rel + ".gz")
            if src.is_file():
                dst = tmp_raw / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.name.endswith(".gz"):
                    with gzip.open(src, "rt", encoding="utf-8", newline="") as fi, \
                            open(dst, "w", encoding="utf-8", newline="") as fo:
                        shutil.copyfileobj(fi, fo)
                else:
                    shutil.copy2(src, dst)
        from apply_modification_log import apply_modification_log
        apply_modification_log(tmp_raw, log_path)
        for rel in INJECTED_TABLES:
            a, b = tmp_raw / rel, raw_out / rel
            if not a.is_file() or not b.is_file():
                continue
            if md5(a) != md5(b):
                replay_ok = False
                replay_detail += " %s" % rel
    res["1_replay"] = replay_ok
    print("  [1] 日志可重放: %s%s"
          % ("OK" if replay_ok else "FAIL" + replay_detail,
             "" if replay_ok else replay_detail))

    res["all_ok"] = all(v for k, v in res.items() if k != "all_ok")
    print("  → %s" % ("全部通过" if res["all_ok"] else "有失败项"))

    # 落一份机读结果，共享报告要引用
    res["dataset"] = key
    res["split"] = split
    res["target_rate"] = rates[split]
    res["actual_rate"] = rate
    res["log_entries"] = total
    res["total_cells"] = cells
    res["n_subtypes_covered"] = len(by_subtype)
    res["natural_noise_count"] = n_class0
    res["by_subtype"] = dict(by_subtype)
    res["by_file"] = dict(by_file)
    res["by_op"] = dict(by_op)
    (out_split / "verify_report.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(DATASETS))
    ap.add_argument("splits", nargs="*", default=None)
    ap.add_argument("--rates", choices=sorted(CELL_RATES), default="4.7pct")
    args = ap.parse_args()
    clean_name, out_base, sd = DATASETS[args.dataset]
    out_name = out_base.format(tag=args.rates)
    rates = CELL_RATES[args.rates]
    splits = args.splits or ("train", "validation", "test")
    all_ok = True
    for sp in splits:
        r = check_split(args.dataset, sp, clean_name, out_name, sd, rates, args.rates)
        all_ok = all_ok and r.get("all_ok", False)
    print("\n" + "=" * 88)
    print("总判定: %s" % ("全部通过" if all_ok else "存在失败项"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
