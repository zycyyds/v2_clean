#!/usr/bin/env python3
"""按 cell 级错误率给 MIMIC 任务数据集注入错误。

与旧的 rebuild_{n18,i48,icu5000}_v2.py（三份近乎逐字复制的脚本）不同，
这里是**参数化**的一个 driver：数据集配置走 DATASETS 表。

注入引擎固定用 build_v2_error_configs.build_modification_log（注入路径），
**不走** reconstruct_mod_log_highrate.py 的 diff 路径 —— 后者会把 raw 与
reference 之间天然就有的差异（浮点渲染、行集错位）记成 class0_baseline_gap
「错误」，1.58pct 数据集 87.7% 的错误就是这么来的。

用法:
    python3 build_rawerror_by_ratio.py n18            # 三个 split 全跑
    python3 build_rawerror_by_ratio.py n18 test       # 只跑 test
    python3 build_rawerror_by_ratio.py n18 --dry-run  # 只算目标不写盘
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from apply_modification_log import apply_modification_log          # noqa: E402
from build_v2_error_configs import build_modification_log, validate_stats  # noqa: E402

ROOT = Path("/Users/mac/PycharmProjects/v2_clean")
CLEAN_ROOT = ROOT / "datasets/常规数据集"
OUT_ROOT = ROOT / "datasets/注入错误数据集"

ENGINE_VERSION = "fixed-v1"

# 两套口径，都对齐 K29 对应版本的实测 cell 级错误率
# （分母 = 该 split 全部 raw 表的 cell 总数，含未注入的干净表）。
#
#   4.7pct 版 ← K29_v2_rawerror_test4.7pct_14.1M
#   1.6pct 版 ← K29_v2_rawerror_test1.6pct_4.2M
#
# 注意 K29 的 1.6pct 版走的是 diff 重建路径，87.7% 的条目是 class0 自然噪声
# （raw 与 gold 天然就有的差异被误记成错误）。本 driver 两个口径都走注入引擎，
# 所以 1.6pct 版是 0 自然噪声的。
RATE_PRESETS = {
    "4.7pct": {
        "rates": {"train": 0.0233, "validation": 0.0089, "test": 0.0472},
        "tag": "4.7pct",
    },
    "1.6pct": {
        "rates": {"train": 0.0130, "validation": 0.0224, "test": 0.0158},
        "tag": "1.6pct",
    },
    # 低密度档：**绝对条数**（不是比例）。
    # 与 N18/I48/ICU5000 的 *_v2_rawerror_10train_400_800_1800 同规格，
    # 但那三个是 legacy 引擎造的（future_window_leakage 退化成清空 charttime、
    # stay_hadm_mismatch 写成字面量），这版走 fixed-v1 引擎。
    "1800": {
        "counts": {"train": 400, "validation": 800, "test": 1800},
        "tag": "1800",
    },
}

DATASETS = {
    "n18": {
        "clean": "N18_ckd_ED_notes_eval_10_20_rest",
        "out_base": "N18_ckd_ED_v2_rawerror_test{tag}_490M",
        "split_key": "subject_id",
        "split_dir": {"train": "train", "validation": "val", "test": "test"},
    },
    "i48": {
        "clean": "I48_atrial_fibrillation_ECG_eval_10_20_rest",
        "out_base": "I48_atrial_fibrillation_v2_rawerror_test{tag}_411M",
        "split_key": "subject_id",
        "split_dir": {"train": "train", "validation": "val", "test": "test"},
    },
    "icu5000": {
        "clean": "icu_5000_clean_eval_10_20_rest",
        "out_base": "icu_5000_v2_rawerror_test{tag}_614M",
        "split_key": "stay_id",
        "split_dir": {"train": "train", "validation": "val", "test": "test"},
    },
    # K29（胃炎/十二指肠炎）。切分键是 hadm_id（split_manifest.json 的
    # split_strategy = deterministic_random_split_by_hadm_id，seed 20260819），
    # 干净数据的 keys.csv 首列也是 hadm_id ⇒ read_keys 自动读对。
    # 注意 split_key 在引擎里其实是死参数，真正的「哪些行算在范围内」
    # 由 all_keys（= keys.csv 首列）决定。
    "k29": {
        "clean": "K29_clean_eval_10_20_rest",
        "out_base": "K29_clean_v2_rawerror_10train_400_800_{tag}",
        "split_key": "hadm_id",
        "split_dir": {"train": "train", "validation": "val", "test": "test"},
    },
}

SPLITS = ("train", "validation", "test")


# ----------------------------------------------------------------------
# 基础 IO
# ----------------------------------------------------------------------
def open_any(path: Path, mode: str = "r"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return open(path, mode, encoding="utf-8", newline="")


def count_raw_cells(raw_dir: Path) -> tuple[int, int]:
    """返回 (总 cell 数, 总行数)。

    必须用 csv.reader 数**记录数**而不是物理行 —— note/discharge.csv 的
    text 字段含嵌入换行，物理行比记录数多 330 倍（K29 test: 2,055,293
    物理行 vs 6,219 条记录）。旧报告就是在这里算错的。
    """
    total_cells = 0
    total_rows = 0
    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file() or p.name == ".DS_Store":
            continue
        if not (p.name.endswith(".csv") or p.name.endswith(".csv.gz")):
            continue
        with open_any(p) as f:
            rd = csv.reader(f)
            cols = next(rd, [])
            n = 0
            for _ in rd:
                n += 1
        total_cells += len(cols) * n
        total_rows += n
    return total_cells, total_rows


def decompress_copytree(src: Path, dst: Path) -> None:
    """拷贝 raw 目录，.csv.gz 一律解压成明文 .csv。"""
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.rglob("*")):
        if not p.is_file() or p.name == ".DS_Store":
            continue
        rel = p.relative_to(src)
        target = dst / rel
        if p.name.endswith(".gz"):
            target = target.with_suffix("")       # 去掉 .gz
            target.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(p, "rt", encoding="utf-8", newline="") as fin, \
                    open(target, "w", encoding="utf-8", newline="") as fout:
                shutil.copyfileobj(fin, fout)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)


def read_keys(split_dir: Path) -> list[str]:
    kf = split_dir / "keys.csv"
    if not kf.is_file():
        return []
    with kf.open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        col = rd.fieldnames[0]
        return [(r.get(col) or "").strip() for r in rd
                if (r.get(col) or "").strip()]


def find_table(raw_dir: Path, rel: str) -> Path | None:
    """找一张表 —— 干净数据集里有的用 .csv、有的用 .csv.gz（ICU5000 是 .gz）。

    之前三个 map 构造器只试明文路径，ICU5000 上全部静默返回空集：
    subject_id_lookup 空 → accept_set 里只有 stay_id → patients / admissions
    这些按 subject_id 组织的表 eligible=0 → 5 个 subtype 直接没注入；
    icustays_outtime_map 空 → future_window_leakage 退化成「清空 charttime」；
    cohort_hadm_map 空 → stay_hadm_mismatch 退化成写字面量。
    """
    p = raw_dir / rel
    if p.is_file():
        return p
    p_gz = raw_dir / (rel + ".gz")
    if p_gz.is_file():
        return p_gz
    return None


def build_subject_lookup(raw_dir: Path) -> set[str]:
    subs: set[str] = set()
    for rel in ("hosp/admissions.csv", "icu/icustays.csv"):
        p = find_table(raw_dir, rel)
        if p is None:
            continue
        with open_any(p) as f:
            rd = csv.DictReader(f)
            if "subject_id" not in (rd.fieldnames or []):
                continue
            for r in rd:
                v = (r.get("subject_id") or "").strip()
                if v:
                    subs.add(v)
    return subs


def build_icustays_outtime_map(raw_dir: Path) -> dict[str, str]:
    """stay_id -> outtime，供 future_window_leakage 用。

    不传这个 map 的话，chartevents 里没有 outtime 列，_shift_time 会解析失败
    返回空串 —— charttime 被整列清空，那不是「未来窗口泄漏」而是「时间戳被
    摧毁」，agent 无法从 train 学到如何识别。
    """
    out: dict[str, str] = {}
    p = find_table(raw_dir, "icu/icustays.csv")
    if p is None:
        return out
    with open_any(p) as f:
        for r in csv.DictReader(f):
            sid = (r.get("stay_id") or "").strip()
            ot = (r.get("outtime") or "").strip()
            if sid and ot:
                out[sid] = ot
    return out


def build_cohort_hadm_map(raw_dir: Path) -> dict[str, list[str]]:
    """stay_id -> [hadm_id]，供 stay_hadm_mismatch 真做跨表错位。

    不传这个 map 的话，dirty_value 拿不到别的 hadm_id，hadm_id 列会被写成
    字面量 "(error generating)"。
    """
    out: dict[str, list[str]] = {}
    p = find_table(raw_dir, "icu/icustays.csv")
    if p is None:
        return out
    with open_any(p) as f:
        for r in csv.DictReader(f):
            sid = (r.get("stay_id") or "").strip()
            hid = (r.get("hadm_id") or "").strip()
            if sid and hid:
                out.setdefault(sid, []).append(hid)
    return out


def deterministic_seed(dataset_key: str, split: str) -> int:
    """确定性 seed。

    旧的 rebuild_*.py 用 hash((NAME, split)) —— Python 字符串 hash 受
    PYTHONHASHSEED 随机化，每次进程都可能不同，所以现有三个数据集的注入
    不可复现（ICU5000 的 train/validation 甚至被重跑成另一套错误集，
    报告却没跟着更新）。
    """
    h = hashlib.md5(f"{dataset_key}|{split}|{ENGINE_VERSION}".encode()).hexdigest()
    return int(h, 16) & 0xFFFFFFFF


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def run_split(key: str, cfg: dict, split: str, preset: dict, dry_run: bool = False) -> dict:
    clean_dir = CLEAN_ROOT / cfg["clean"] / cfg["split_dir"][split]
    out_name = cfg["out_base"].format(tag=preset["tag"])
    out_dir = OUT_ROOT / out_name / split
    raw_src = clean_dir / "raw"
    if not raw_src.is_dir():
        print(f"  [SKIP] {split}: 目标 raw 不存在 {raw_src}")
        return {}

    print(f"\n{'=' * 78}\n=== {key} / {split}\n{'=' * 78}", flush=True)

    # 1. 目标：按 cell 率算（或按绝对条数，低密度档用）
    total_cells, total_rows = count_raw_cells(raw_src)
    if "counts" in preset:
        target = int(preset["counts"][split])
        rate = target / total_cells
        mode = f"绝对条数 {target:,} 条"
    else:
        rate = preset["rates"][split]
        target = int(round(total_cells * rate))
        mode = f"cell 率 {rate * 100:.4f}%"
    print(f"  全部 raw: {total_rows:,} 行 / {total_cells:,} cells")
    print(f"  目标 {mode}  →  target {target:,} 条", flush=True)
    if dry_run:
        return {"target_errors": target, "total_cells": total_cells, "total_rows": total_rows}

    # 2. 拷贝 raw（解压 gz）+ reference/keys
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    decompress_copytree(raw_src, out_dir / "raw")
    ref_src = clean_dir / "reference"
    if ref_src.is_dir():
        shutil.copytree(ref_src, out_dir / "reference")
    keys_src = clean_dir / "keys.csv"
    if keys_src.is_file():
        shutil.copy2(keys_src, out_dir / "keys.csv")
    print(f"  raw 已拷贝并解压: {out_dir / 'raw'}", flush=True)

    raw_dst = out_dir / "raw"
    log_path = out_dir / "modification_log.csv"

    # 3. 建三张 map
    all_keys = read_keys(clean_dir)
    subject_id_lookup = build_subject_lookup(raw_src)
    icustays_outtime_map = build_icustays_outtime_map(raw_src)
    cohort_hadm_map = build_cohort_hadm_map(raw_src)
    print(f"  keys={len(all_keys)}  subjects={len(subject_id_lookup)}  "
          f"stays(outtime)={len(icustays_outtime_map)}  stays(hadm)={len(cohort_hadm_map)}",
          flush=True)

    # 4. 生成日志
    # 硬校验：map 空 = 上次 ICU5000 那种静默退化（.gz 没找到 →
    # 5 个 subtype eligible=0、charttime 被清空、hadm_id 写字面量）。
    # 与其产出一个「看起来成功」的数据集，不如当场报错。
    problems = []
    if not all_keys:
        problems.append("keys.csv 为空")
    if not subject_id_lookup:
        problems.append("subject_id_lookup 为空")
    if not icustays_outtime_map:
        problems.append("icustays_outtime_map 为空（future_window_leakage 会退化成清空 charttime）")
    if not cohort_hadm_map:
        problems.append("cohort_hadm_map 为空（stay_hadm_mismatch 会写成字面量）")
    if problems:
        raise RuntimeError(f"{key}/{split} 前置 map 构造失败：{'; '.join(problems)}")

    t0 = time.time()
    stats = build_modification_log(
        split,
        target,
        raw_src,                       # 在**未注入**的 baseline raw 上取样
        log_path,
        split_key=cfg["split_key"],
        all_keys=all_keys,
        random_seed=deterministic_seed(key, split),
        must_cover_all_subtypes=True,
        icustays_outtime_map=icustays_outtime_map,
        cohort_hadm_map=cohort_hadm_map,
        subject_id_lookup=subject_id_lookup,
        target_rate=rate,              # 按「注入后」的 cell 数算率，与 K29 口径一致
        baseline_cells=total_cells,
    )
    build_seconds = time.time() - t0
    print(f"  日志生成 {build_seconds:.0f}s  actual={stats['actual_count']:,}  "
          f"subtypes={stats['n_subtypes_covered']}  "
          f"shortfall={stats.get('allocation_shortfall', 0):,}", flush=True)

    # 5. 应用到 raw
    t1 = time.time()
    applied = apply_modification_log(raw_dst, log_path)
    apply_seconds = time.time() - t1
    print(f"  应用 {apply_seconds:.0f}s  cell_update={applied['n_cell_updates']:,} "
          f"dup={applied['n_duplicates']:,} insert={applied['n_inserts']:,} "
          f"conflict={applied.get('n_conflicts', 0):,}", flush=True)

    # 6. 校验
    # 分母口径 = **注入后**的 raw cell 总数（K29 4.72pct 的口径：
    # 14,123,558 / 299,037,066 = 4.723%）。duplicate / insert 会新增行，
    # 所以分母比 baseline 大；用 baseline 会算出虚高的错误率。
    final_cells = stats.get("projected_final_cells") or total_cells
    report = {
        "target_errors": target,
        "actual_errors": stats["actual_count"],
        "cell_level_error_rate": round(stats["actual_count"] / final_cells, 6),
        "total_cells": final_cells,
        "baseline_total_cells": total_cells,
        "total_rows": total_rows,
        "subtype_counts": stats["subtype_counts"],
        "file_counts": stats["file_counts"],
        "applied": applied,
        "build_seconds": round(build_seconds, 1),
        "apply_seconds": round(apply_seconds, 1),
        "seed": deterministic_seed(key, split),
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", choices=sorted(DATASETS), help="数据集 key")
    ap.add_argument("split", nargs="?", choices=SPLITS, help="只跑某一个 split")
    ap.add_argument("--rates", choices=sorted(RATE_PRESETS), default="4.7pct",
                    help="错误率口径预设")
    ap.add_argument("--dry-run", action="store_true", help="只算目标，不写盘")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    preset = RATE_PRESETS[args.rates]
    out_name = cfg["out_base"].format(tag=preset["tag"])
    splits = (args.split,) if args.split else SPLITS
    reports = {}
    for split in splits:
        r = run_split(args.dataset, cfg, split, preset, dry_run=args.dry_run)
        if r:
            reports[split] = r

    if not args.dry_run:
        out_dir = OUT_ROOT / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"v2_rawerror_{args.dataset}_final_report.json"
        # 一次只跑一个 split 时不能把上一次的 split 冲掉 —— 合并写
        report = {
            "name": out_name,
            "baseline": cfg["clean"],
            "injection_engine_version": ENGINE_VERSION,
            "rate_preset": args.rates,
            "cell_rates": preset.get("rates")
            or {sp: (preset["counts"][sp] / max(
                (r["total_cells"] for r in reports.values() if r and sp in r), default=1))
                for sp in SPLITS if sp in reports},
            "splits": {},
        }
        if p.is_file():
            try:
                old = json.loads(p.read_text(encoding="utf-8"))
                if (old.get("name") == out_name
                        and old.get("injection_engine_version") == ENGINE_VERSION
                        and old.get("rate_preset") == args.rates):
                    report["splits"] = old.get("splits", {})
            except (json.JSONDecodeError, OSError):
                pass
        report["splits"].update(reports)
        p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n报告已写出: {p}（splits: {sorted(report['splits'])}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
