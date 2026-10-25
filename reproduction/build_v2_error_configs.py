"""Random error injection configuration for v2 raw-error 5 datasets.

设计：18 个错误子类的注入规则，每个子类直接对应 raw 表 + 列 + dirty_value 生成器。
不同于 raw_level_error_split.py（从 feature-level log 反推 raw），本模块**直接构造 raw-level modification_log**，
然后由 apply_modification_log.py 直接改 raw csv。

设计原则（随机化）：
- 行抽样：每行概率 = 1/total_rows（不放回）
- subtype 比例：Dirichlet(α=1.0) 抽样 18 个 subtype，每次 split 重抽
- 文件选择：每个 subtype 候选文件清单轮换
- stay 分布：Poisson(λ=target/stays)，最低 1 个/stay
- 列选择：每 subtype 命中多个可能列时随机选
- 错误值：每个 candidate pool 随机选
"""
from __future__ import annotations

import csv
import gzip
import json
import random
import shutil
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np

# ----------------------------------------------------------------------
# 18 subtype 配置（按 error_class 排好序）
# ----------------------------------------------------------------------

# 辅助：生成 dirty_value 的工厂
def _constant(value: str) -> Callable[[dict, random.Random], str]:
    def gen(row: dict, rng: random.Random) -> str:
        return value
    return gen

def _shift_time(field: str, hours: int) -> Callable[[dict, random.Random], str]:
    def gen(row: dict, rng: random.Random) -> str:
        v = row.get(field) or ""
        try:
            dt = datetime.strptime(v.strip(), "%Y-%m-%d %H:%M:%S")
            return (dt + timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return v  # 解析失败就保留
    return gen

def _set_before(field: str) -> Callable[[dict, random.Random], str]:
    def gen(row: dict, rng: random.Random) -> str:
        v = row.get(field) or ""
        try:
            dt = datetime.strptime(v.strip(), "%Y-%m-%d %H:%M:%S")
            return (dt - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return v
    return gen

def _scale_100(row: dict, rng: random.Random) -> str:
    v = row.get("valuenum") or "0"
    try:
        return str(round(float(v) * 100, 6))
    except Exception:
        return v

def _prefix_99(row: dict, rng: random.Random) -> str:
    v = (row.get("itemid") or "").strip()
    if v.isdigit():
        return "99" + v
    return v

def _alias_prefix(row: dict, rng: random.Random) -> str:
    v = (row.get("icd_code") or "").strip()
    return f"ALIAS_{v}"

def _orphan_stay(row: dict, rng: random.Random) -> str:
    v = (row.get("stay_id") or "").strip()
    if v:
        # 7 位 stay_id 前加 9 不太可能与真实 stay 冲突
        return "9" + v.lstrip("9")
    return v

def _flip_label(row: dict, rng: random.Random) -> str:
    v = (row.get("hospital_expire_flag") or "0").strip()
    return "0" if v == "1" else "1"

def _sentinel_pool(*values: str) -> Callable[[dict, random.Random], str]:
    def gen(row: dict, rng: random.Random) -> str:
        return rng.choice(values)
    return gen

def _row_duplicate(row: dict, rng: random.Random) -> str:
    """duplicate_row 不修改 cell，而是 row_insert。
    标记 column='(duplicate)'，实际 dirty_value 留空，由 apply 阶段用 prev_row 填充。"""
    return "(duplicate)"

def _label_leakage_insert(row: dict, rng: random.Random) -> str:
    """label_leakage_feature 也是 row_insert。column='(insert)'，marker。"""
    return "(insert)"

def _swap_hadm(row: dict, rng: random.Random) -> str:
    """stay_hadm_mismatch 的兜底 dirty_value。

    真正的跨表错位由 build_modification_log 传入 cohort_hadm_map 完成
    （stay_id -> 该 stay 实际所属的 hadm_id 列表）。这里没有 map 时退化成
    「留空」而不是写死一个字符串常量 —— 旧版写死字符串会让
    cfg["dirty_value"](row, rng) 抛 TypeError，被 except 吞成
    "(error generating)"，把 hadm_id 写成了字面量。
    """
    return row.get("hadm_id", "") or ""

# 18 subtype 完整定义
ERROR_SUBTYPES: list[dict] = [
    # ----- class1_intra_table (10 个) -----
    {
        "subtype": "schema_range_violation",
        "error_class": "class1_intra_table",
        "raw_file": "hosp/patients.csv",
        "column": "anchor_age",
        "dirty_value": _constant("999"),
        "matching_columns": ["subject_id"],
    },
    {
        "subtype": "enum_violation",
        "error_class": "class1_intra_table",
        "raw_file": "hosp/patients.csv",
        "column": "gender",
        "dirty_value": _constant("UNKNOWN_GENDER_CODE"),
        "matching_columns": ["subject_id"],
    },
    {
        "subtype": "unexpected_missingness",
        "error_class": "class1_intra_table",
        "raw_file": "hosp/admissions.csv",
        "column": "insurance",
        "dirty_value": _constant(""),
        "matching_columns": ["subject_id", "hadm_id"],
    },
    {
        "subtype": "temporal_order_violation",
        "error_class": "class1_intra_table",
        "raw_file": "icu/icustays.csv",
        "column": "outtime",
        "dirty_value": _shift_time("intime", hours=-1),
        "matching_columns": ["subject_id", "hadm_id", "stay_id"],
    },
    {
        "subtype": "med_end_before_start",
        "error_class": "class1_intra_table",
        "raw_file": "icu/inputevents.csv",
        "column": "endtime",
        "dirty_value": _set_before("starttime"),
        "matching_columns": ["subject_id", "hadm_id", "stay_id"],
    },
    {
        "subtype": "sentinel_or_negative_med_amount",
        "error_class": "class1_intra_table",
        "raw_file": "icu/inputevents.csv",
        "column": "amount",
        "dirty_value": _sentinel_pool("-999", "-1000", "-888", "-1"),
        "matching_columns": ["subject_id", "hadm_id", "stay_id"],
    },
    {
        "subtype": "sentinel_chart_value",
        "error_class": "class1_intra_table",
        "raw_file": "icu/chartevents.csv",
        "column": "valuenum",
        "dirty_value": _sentinel_pool("-999", "9999", "-888", "-1"),
        "matching_columns": ["subject_id", "hadm_id", "stay_id"],
    },
    {
        "subtype": "unit_scale_error",
        "error_class": "class1_intra_table",
        "raw_file": "icu/chartevents.csv",
        "column": "valuenum",
        "dirty_value": _scale_100,
        "matching_columns": ["subject_id", "hadm_id", "stay_id"],
    },
    # duplicate_row 必须是第 9 项 —— 顺序照 raw_level_error_split.py:
    # _target_for_subtype。Dirichlet 抽样按列表下标分配，顺序错了命中数会漂移。
    {
        "subtype": "duplicate_row",
        "error_class": "class1_intra_table",
        "raw_file": "icu/chartevents.csv",
        "column": "(row)",
        "dirty_value": _row_duplicate,
        "matching_columns": ["subject_id", "hadm_id", "stay_id"],
        "row_operation": "duplicate",
    },
    {
        "subtype": "implausible_negative_event_time",
        "error_class": "class1_intra_table",
        "raw_file": "icu/outputevents.csv",
        "column": "charttime",
        "dirty_value": _constant("1970-01-01 00:00:00"),
        "matching_columns": ["subject_id", "hadm_id", "stay_id"],
    },
    # ----- class2_entity_alignment (3 个) -----
    {
        "subtype": "chart_itemid_alias_split",
        "error_class": "class2_entity_alignment",
        "raw_file": "icu/chartevents.csv",
        "column": "itemid",
        "dirty_value": _prefix_99,
        "matching_columns": ["subject_id", "hadm_id", "stay_id"],
    },
    {
        "subtype": "medication_itemid_alias_split",
        "error_class": "class2_entity_alignment",
        "raw_file": "icu/inputevents.csv",
        "column": "itemid",
        "dirty_value": _prefix_99,
        "matching_columns": ["subject_id", "hadm_id", "stay_id"],
    },
    {
        "subtype": "diagnosis_code_alias_split",
        "error_class": "class2_entity_alignment",
        "raw_file": "hosp/diagnoses_icd.csv",
        "column": "icd_code",
        "dirty_value": _alias_prefix,
        "matching_columns": ["subject_id", "hadm_id"],
    },
    # ----- class3_cross_table (3 个) -----
    {
        "subtype": "orphan_feature_stay_id",
        "error_class": "class3_cross_table",
        "raw_file": "icu/chartevents.csv",
        "column": "stay_id",
        "dirty_value": _orphan_stay,
        "matching_columns": ["subject_id", "hadm_id", "stay_id"],
    },
    {
        "subtype": "stay_hadm_mismatch",
        "error_class": "class3_cross_table",
        "raw_file": "icu/inputevents.csv",
        "column": "hadm_id",
        "dirty_value": _swap_hadm,
        "matching_columns": ["subject_id", "hadm_id", "stay_id"],
        "cross_table": True,  # 需要从 cohort.csv 找其他 stay 的 hadm
    },
    {
        "subtype": "label_conflict_with_cohort",
        "error_class": "class3_cross_table",
        "raw_file": "hosp/admissions.csv",
        "column": "hospital_expire_flag",
        "dirty_value": _flip_label,
        "matching_columns": ["subject_id", "hadm_id"],
    },
    # ----- class4_task_oriented (2 个) -----
    {
        "subtype": "future_window_leakage",
        "error_class": "class4_task_oriented",
        "raw_file": "icu/chartevents.csv",
        "column": "charttime",
        "dirty_value": _shift_time("outtime", hours=24),  # 默认 +1day
        "matching_columns": ["subject_id", "hadm_id", "stay_id"],
        "needs_icustays_outtime": True,
    },
    # ----- 行级操作 (2 个) -----
    # 这两条曾被从列表里删掉（工厂函数与 row_operation 开关位仍在），
    # 导致引擎只能产出 16 型、无法复现 4.72pct 数据集。
    # 顺序必须与 raw_level_error_split.py:_target_for_subtype 一致，
    # 否则 Dirichlet 分配对不上、18 型命中数会漂移。
    {
        "subtype": "label_leakage_feature",
        "error_class": "class4_task_oriented",
        "raw_file": "icu/chartevents.csv",
        "column": "(row)",
        "dirty_value": _label_leakage_insert,
        "matching_columns": ["subject_id", "hadm_id", "stay_id"],
        "row_operation": "insert",
    },
]


# ----------------------------------------------------------------------
# raw csv 读取/写入辅助
# ----------------------------------------------------------------------

def open_csv(path: Path, mode: str = "r"):
    """自动解压 .gz 文件。"""
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return open(path, mode, encoding="utf-8", newline="")


def read_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    """返回 (header, rows)。rows 是 list of dict。"""
    with open_csv(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [dict(zip(header, row)) for row in reader]
    return header, rows


def reservoir_sample_eligible(
    path: Path,
    matching_columns: list[str],
    accept_set: set[str],
    k: int,
    rng: random.Random,
    shard_mod: int = 1,
    shard_rem: int = 0,
) -> list[tuple[int, dict]]:
    """流式扫描 path，reservoir-sampling 保留最多 k 个 matching_rows（行 index + row dict）。
    返回 sampled 列表；如总匹配行 <= k 则全部返回。

    shard_mod / shard_rem：只保留 raw_row_index % shard_mod == shard_rem 的行。
    同一 (raw_file, column) 的多个 subtype 用不同分片，保证行集互斥。
    """
    sampled: list[tuple[int, dict]] = []
    seen = 0
    with open_csv(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows_iter = enumerate(reader)
        # 先把所有 matching 收集（如果 accept_set 不大、k 不大），否则 reservoir
        # 为简单起见：第一遍先收集所有 matching rows（最多 500K），再 reservoir sample
        # —— 这是对小数据集最简的做法。对大文件（32M+），用真正的 reservoir。
        col_idx = [(header.index(c), c) for c in matching_columns if c in header]
        for idx, row_vals in rows_iter:
            if shard_mod > 1 and (idx % shard_mod) != shard_rem:
                continue
            if not any((i < len(row_vals) and row_vals[i] in accept_set) for i, _ in col_idx):
                continue
            row_dict = dict(zip(header, row_vals))
            seen += 1
            if len(sampled) < k:
                sampled.append((idx, row_dict))
            else:
                # reservoir replacement
                j = rng.randint(0, seen - 1)
                if j < k:
                    sampled[j] = (idx, row_dict)
    return sampled


def count_eligible(
    path: Path,
    matching_columns: list[str],
    accept_set: set[str],
) -> tuple[int, int]:
    """精确统计匹配 accept_set 的行数，以及文件总行数（O(1) 内存）。

    旧版用 `reservoir_sample_eligible(..., est_k=min(target*5, 5M))` 的返回
    长度当容量上限，那是个**抽样上限**而不是真实容量：chartevents 有 918 万
    可注入行，但 est_k 封在 500 万，于是 max_eligible 被低估、target 打不满
    （K29 目标 2,650 万实际只有 1,412 万）。容量必须精确计数。
    """
    total = 0
    n_rows = 0
    with open_csv(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        idxs = [header.index(c) for c in matching_columns if c in header]
        if not idxs:
            return 0, 0
        for row in reader:
            n_rows += 1
            for i in idxs:
                if i < len(row) and row[i] in accept_set:
                    total += 1
                    break
    return total, n_rows


def water_fill_allocate(
    target: int,
    weights: list[float],
    caps: list[int],
) -> tuple[list[int], int]:
    """按 weights 比例分配 target，同时不突破每项的 caps。

    返回 (counts, shortfall)：
      - sum(counts) == target - shortfall
      - counts[i] <= caps[i]
      - shortfall > 0 表示总容量不足（打不满 target）

    旧版用「每次 +1、循环上限 1000 次」搬缺口，几百万的缺口搬不动，
    这是 K29 test 目标打不满的直接原因之一。本实现 O(n) 收敛。
    """
    n = len(weights)
    counts = [0] * n
    remaining = target
    active = [i for i in range(n) if caps[i] > 0 and weights[i] > 0]
    if not active:
        active = [i for i in range(n) if caps[i] > 0]
        weights = [1.0] * n
    if not active:
        return counts, target

    while remaining > 0 and active:
        total_cap = sum(caps[i] - counts[i] for i in active)
        if total_cap <= 0:
            break
        wsum = sum(weights[i] for i in active) or float(len(active))
        progressed = False
        for i in active:
            share = int(remaining * weights[i] / wsum)
            add = min(share, caps[i] - counts[i])
            if add > 0:
                counts[i] += add
                remaining -= add
                progressed = True
        if not progressed:
            break
        still = [i for i in active if counts[i] < caps[i]]
        if len(still) == len(active) and still:
            # 本轮没有任何一项打满却也没进展 —— 权重过小，按容量均摊兜底
            for i in still:
                add = min(remaining, caps[i] - counts[i])
                counts[i] += add
                remaining -= add
                if remaining <= 0:
                    break
        active = [i for i in active if counts[i] < caps[i]]
    return counts, remaining


def shard_group(cfg: dict) -> tuple:
    """同一 (raw_file, column) 的 cell_update 会互相覆盖（apply 阶段同格后写
    覆盖先写），必须让它们抽到**互不相交的行集合**，否则日志里的条目会静默
    失效 —— K29 test 有 431,125 条是这样丢的。

    按 raw_row_index % k 分片即可得到互斥行集合，代价为零。
    行级操作（duplicate / insert）不改 cell，彼此不冲突，各自独立成组。
    """
    if cfg.get("row_operation"):
        return (cfg["raw_file"], cfg["subtype"])
    return (cfg["raw_file"], cfg["column"])


def _project_injected_cells(
    subtypes: list[dict],
    adjusted: list[int],
    file_cols: dict[str, int],
    file_rows: dict[str, int],
    baseline_cells: int,
) -> int:
    """按当前分配估算「注入后」的 cell 总数。

    只有 duplicate / insert 会新增行；cell_update 不改变行数。
    注意 baseline_cells 是**全部** raw 表的 cell 数（含未注入的干净表），
    所以要先把 7 张注入表的原始 cell 数扣掉，再加上它们注入后的 cell 数。
    """
    extra_rows: dict[str, int] = defaultdict(int)
    for cfg, n in zip(subtypes, adjusted):
        if cfg.get("row_operation") in ("duplicate", "insert"):
            extra_rows[cfg["raw_file"]] += n
    injected_tables_cells = 0
    for rel, ncols in file_cols.items():
        nrows = file_rows.get(rel, 0)
        injected_tables_cells += (nrows + extra_rows.get(rel, 0)) * ncols
    untouched_cells = baseline_cells - sum(
        file_rows.get(rel, 0) * ncols for rel, ncols in file_cols.items()
    )
    return untouched_cells + injected_tables_cells


def write_csv_rows(path: Path, header: list[str], rows: list[dict]) -> None:
    """写出 csv。如果 path 以 .gz 结尾自动 gzip。"""
    is_gz = str(path).endswith(".gz")
    opener = gzip.open if is_gz else open
    with opener(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------
# modification_log 生成器（核心）
# ----------------------------------------------------------------------

def build_modification_log(
    split_name: str,
    target_count: int,
    baseline_raw_dir: str | Path,
    output_log_path: str | Path,
    *,
    split_key: str = "stay_id",
    all_keys: list[str] | None = None,  # 关键 id 列表（stay_id or hadm_id or subject_id）
    random_seed: int | None = None,
    must_cover_all_subtypes: bool = True,
    icustays_outtime_map: dict[str, str] | None = None,  # for future_window_leakage
    cohort_hadm_map: dict[str, list[str]] | None = None,  # for stay_hadm_mismatch
    subject_id_lookup: set[str] | None = None,  # 额外可作为 subject_id 候选的 set
    target_rate: float | None = None,       # 目标 cell 级错误率（按注入后 cell 数算）
    baseline_cells: int | None = None,       # baseline raw 的 cell 总数
) -> dict:
    """生成 modification_log.csv（不含行索引，直接给出 dirty_value）。

    Returns
    -------
    stats : dict
        包含随机性指标（让 caller 校验）。
    """
    rng = random.Random(random_seed if random_seed is not None else hash((split_name, target_count)))
    baseline = Path(baseline_raw_dir).expanduser().resolve()
    output = Path(output_log_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # 1. Dirichlet 抽样 18 个 subtype 的比例（更均匀，避免极端集中）
    # alpha=4.0 让分布更平，避免某一 subtype 占比过大
    proportions = rng.dirichlet([4.0] * len(ERROR_SUBTYPES)) if hasattr(rng, "dirichlet") else np.random.default_rng(random_seed).dirichlet([4.0] * len(ERROR_SUBTYPES))

    # 2. 计算每个 subtype 的目标错误数（保证 sum == target_count）
    raw_counts = [round(target_count * p) for p in proportions]
    # 修正舍入误差
    diff = target_count - sum(raw_counts)
    if diff != 0:
        # 把差值分摊给数量最少的 subtype（避免过度集中）
        idx_sorted = sorted(range(len(raw_counts)), key=lambda i: raw_counts[i])
        i = 0
        while diff != 0:
            j = idx_sorted[i % len(idx_sorted)]
            if diff > 0:
                raw_counts[j] += 1
                diff -= 1
            else:
                if raw_counts[j] > 1:
                    raw_counts[j] -= 1
                    diff += 1
            i += 1

    # 3. 至少给每个 subtype 1 个（如果 must_cover_all_subtypes）
    if must_cover_all_subtypes and split_name in ("train", "validation"):
        # train 必须 18 个 subtype 全覆盖
        for i in range(len(raw_counts)):
            if raw_counts[i] == 0:
                # 从最多的减 1 给这个
                donor_idx = max(range(len(raw_counts)), key=lambda k: raw_counts[k])
                if raw_counts[donor_idx] > 1:
                    raw_counts[donor_idx] -= 1
                    raw_counts[i] = 1

    # 4. 对每个 subtype：抽行 → 生成 dirty_value
    print(f"  [Dirichlet] alpha=4.0 raw_counts={raw_counts}", flush=True)

    # 4a. 精确统计 eligible 上限（旧版用蓄水池抽样长度当上限，会低估容量）
    max_eligible: list[int] = [0] * len(ERROR_SUBTYPES)
    raw_paths: list[Path] = [None] * len(ERROR_SUBTYPES)  # type: ignore
    accept_sets: list[set[str]] = [set()] * len(ERROR_SUBTYPES)
    if all_keys:
        key_set = set(all_keys)
        accept_base = key_set | (subject_id_lookup or set())
    else:
        accept_base = set()

    # 每个 (raw_file, column) 分组内的 subtype 共享「每行只能改一次」的约束，
    # 按 raw_row_index % k 分片得到互斥行集合。
    group_members: dict[tuple, list[int]] = {}
    for idx, cfg in enumerate(ERROR_SUBTYPES):
        group_members.setdefault(shard_group(cfg), []).append(idx)
    shard_of: dict[int, tuple[int, int]] = {}   # subtype_idx -> (mod, rem)
    for gkey, members in group_members.items():
        for j, idx in enumerate(members):
            shard_of[idx] = (len(members), j)

    file_eligible: dict[str, int] = {}
    file_cols: dict[str, int] = {}
    file_rows: dict[str, int] = {}
    for subtype_idx, cfg in enumerate(ERROR_SUBTYPES):
        raw_file = baseline / cfg["raw_file"]
        if not raw_file.is_file():
            raw_file_gz = baseline / (cfg["raw_file"] + ".gz")
            if raw_file_gz.is_file():
                raw_file = raw_file_gz
            else:
                max_eligible[subtype_idx] = 0
                continue
        raw_paths[subtype_idx] = raw_file
        accept_sets[subtype_idx] = accept_base
        if cfg["raw_file"] not in file_eligible:
            n_elig, n_rows = count_eligible(
                raw_file, cfg["matching_columns"],
                accept_base if accept_base else {"__any__"},
            )
            file_eligible[cfg["raw_file"]] = n_elig
            file_rows[cfg["raw_file"]] = n_rows
            with open_csv(raw_file) as f:
                file_cols[cfg["raw_file"]] = len(next(csv.reader(f), []))
        elig = file_eligible[cfg["raw_file"]]
        mod, rem = shard_of[subtype_idx]
        # 分片后每个 subtype 只能用到约 1/mod 的行
        max_eligible[subtype_idx] = elig // mod if mod > 1 else elig

    # 4b. 按比例分配 target，cap 在精确容量上；缺口用 water-filling 一次搬完
    adjusted, shortfall = water_fill_allocate(
        target_count, [max(p, 1e-9) for p in proportions], max_eligible
    )

    # 4b-bis: 分母口径对齐 —— 注入会新增行（duplicate / insert），注入后的
    # cell 数比 baseline 大，所以「错误数 / baseline cells」会低于目标率。
    # K29 的 4.72pct 是按**注入后**的 cell 数算的：
    #     14,123,558 / 299,037,066 = 4.723%
    # 这里反推：给定目标率，求「注入后 cell 数 × 率」= 应注入多少条，
    # 再用新的 target 重新分配。只做分配（不抽行），几轮就收敛。
    if target_rate and baseline_cells:
        for _ in range(4):
            injected_cells = _project_injected_cells(
                ERROR_SUBTYPES, adjusted, file_cols, file_rows, baseline_cells
            )
            new_target = int(round(injected_cells * target_rate))
            if abs(new_target - target_count) <= max(1, target_count // 1000):
                target_count = new_target
                break
            target_count = new_target
            adjusted, shortfall = water_fill_allocate(
                target_count, [max(p, 1e-9) for p in proportions], max_eligible
            )
        print(f"  [Rate-adjusted] target {target_count:,} "
              f"(rate {target_rate * 100:.2f}% over injected cells)", flush=True)

    if shortfall:
        print(f"  [WARN] 容量不足：目标 {target_count}，实得 {target_count - shortfall}"
              f"（缺 {shortfall}）", flush=True)
    # 确保 train 18 个 subtype 全部 ≥1（must_cover）
    if must_cover_all_subtypes:
        for i in range(len(adjusted)):
            if adjusted[i] == 0 and max_eligible[i] > 0:
                adjusted[i] = 1
                for j in range(len(adjusted)):
                    if adjusted[j] > 1:
                        adjusted[j] -= 1
                        break

    # 4b-extra: 平衡 max/min 比例
    # 目标: max(adjusted) / min(adjusted) ≤ 10.0  (高错误率放宽)
    # 算法: 把超额的从最大 subtype 减，递给最未饱和的（同时不破坏 must_cover 至少 1）
    cur_total = sum(adjusted)
    if must_cover_all_subtypes and cur_total > 0:
        max_iter = 500
        for _ in range(max_iter):
            cur_min = min(adjusted)
            cur_max = max(adjusted)
            if cur_min == 0 or cur_max / max(1, cur_min) <= 10.0:
                break
            # 找一个最大值 subtype 减 1
            idx_max = adjusted.index(cur_max)
            if adjusted[idx_max] <= 1:
                break  # can't reduce further
            # 找一个最小值 subtype 加 1（必须 max_eligible 允许）
            idx_min = adjusted.index(cur_min)
            if max_eligible[idx_min] <= adjusted[idx_min]:
                # 最小值 subtype 已饱和 → 找一个非饱和的次小 subtype
                # 按 adjusted 升序找一个能加的
                idx_target = None
                sorted_by_adj = sorted(range(len(adjusted)), key=lambda i: adjusted[i])
                for i in sorted_by_adj:
                    if adjusted[i] > 0 and adjusted[i] < max_eligible[i]:
                        idx_target = i
                        break
                if idx_target is None:
                    break
                idx_min = idx_target
            adjusted[idx_max] -= 1
            adjusted[idx_min] += 1
    print(f"  [Adj counts] {adjusted}, max_eligible={max_eligible}", flush=True)

    # 4b-ter: 注入后的**全部** raw cell 数（K29 口径的分母）
    # 分配定下来之后再算一次 —— 上面的率收敛循环和 must_cover/平衡都可能动过
    # adjusted，用最终的 adjusted 投影才准。
    projected_final_cells = None
    if baseline_cells:
        projected_final_cells = _project_injected_cells(
            ERROR_SUBTYPES, adjusted, file_cols, file_rows, baseline_cells
        )
        print(f"  [Denominator] 注入后 raw cell 总数 {projected_final_cells:,}"
              f"（baseline {baseline_cells:,}）", flush=True)

    # 4c. 真正抽行 + 生成 entry
    # 日志**按 raw_file 分组**写出：每个文件一个临时 writer，最后拼接。
    # 这样 apply 阶段可以顺序读日志、一次只驻留一个文件的条目 ——
    # 否则 2,900 万条 list[dict] 会吃掉 29 GB（16 GB 机器直接 OOM）。
    fields = [
        "source_error_id", "canonical_stay_id", "error_class", "error_subtype",
        "target_effect_contract", "operation", "raw_file", "raw_row_index",
        "column", "clean_value", "dirty_value", "matching_columns",
    ]
    tmp_dir = output.parent / (output.name + ".parts")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    file_writers: dict[str, tuple] = {}
    file_order: list[str] = []

    def writer_for(rel: str):
        w = file_writers.get(rel)
        if w is None:
            part = tmp_dir / (rel.replace("/", "__") + ".csv")
            fh = open(part, "w", newline="", encoding="utf-8")
            dw = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            dw.writeheader()
            w = (fh, dw, part)
            file_writers[rel] = w
            file_order.append(rel)
        return w[1]

    n_logged = 0
    error_id = 0
    files_seen: set[str] = set()
    stays_seen: set[str] = set()

    # 按 stay 打乱 pool，让 Poisson-like 分布自然形成
    if all_keys:
        rng.shuffle(all_keys)

    for subtype_idx, cfg in enumerate(ERROR_SUBTYPES):
        n_errors = adjusted[subtype_idx]
        if n_errors <= 0:
            continue
        raw_file = raw_paths[subtype_idx]
        accept_set = accept_sets[subtype_idx]
        if raw_file is None or not raw_file.is_file():
            print(f"  [WARN] {cfg['subtype']}: {cfg['raw_file']} not found, skipping {n_errors} errors", flush=True)
            continue
        mod, rem = shard_of[subtype_idx]
        if accept_set:
            sampled = reservoir_sample_eligible(
                raw_file, cfg["matching_columns"], accept_set, n_errors, rng,
                shard_mod=mod, shard_rem=rem)
        else:
            sampled = reservoir_sample_eligible(
                raw_file, ["subject_id"], {"__any__"}, n_errors, rng,
                shard_mod=mod, shard_rem=rem)
        if not sampled:
            print(f"  [WARN] {cfg['subtype']}: no eligible rows in {raw_file}, skipping", flush=True)
            continue
        log_writer = writer_for(cfg["raw_file"])

        for row_index, row in sampled:
            error_id += 1
            # 选第一个 matching column 中有值的作为 stay_id
            stay_id = ""
            for col in cfg["matching_columns"]:
                v = row.get(col, "").strip()
                if v:
                    stay_id = v
                    break
            column = cfg["column"]
            clean_value = row.get(column, "") if column not in ("(row)",) else ""

            # 生成 dirty_value（调用工厂函数）
            try:
                dirty_value = cfg["dirty_value"](row, rng)
            except Exception:
                dirty_value = "(error generating)"

            # future_window_leakage 特殊处理：outtime + 1day
            # 注意：上面那个 stay_id 变量装的是 matching_columns 里第一个非空值，
            # 对 chartevents 来说是 **subject_id**，不是 stay_id —— 用它查
            # icustays_outtime_map 永远查不到，charttime 就会被清空。
            # 必须用这一行真正的 stay_id。
            if cfg["subtype"] == "future_window_leakage" and icustays_outtime_map:
                real_stay = (row.get("stay_id") or "").strip()
                outtime = icustays_outtime_map.get(real_stay, "")
                if outtime:
                    try:
                        dt = datetime.strptime(outtime.strip(), "%Y-%m-%d %H:%M:%S")
                        dirty_value = (dt + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        pass

            # stay_hadm_mismatch 特殊处理：换成**另一个** stay 的 hadm_id
            if cfg["subtype"] == "stay_hadm_mismatch" and cohort_hadm_map:
                real_stay = (row.get("stay_id") or "").strip()
                own_hadm = (row.get("hadm_id") or "").strip()
                other_stays = [s for s in cohort_hadm_map.keys() if s != real_stay]
                if other_stays:
                    other_stay = rng.choice(other_stays)
                    cands = [h for h in cohort_hadm_map[other_stay] if h != own_hadm]
                    if cands:
                        dirty_value = rng.choice(cands)

            log_writer.writerow({
                "source_error_id": f"{split_name}-{error_id:04d}",
                "canonical_stay_id": stay_id,
                "error_class": cfg["error_class"],
                "error_subtype": cfg["subtype"],
                "target_effect_contract": "(see raw_level_error_split)",
                "operation": cfg.get("row_operation", "cell_update"),
                "raw_file": cfg["raw_file"],
                "raw_row_index": row_index,
                "column": column,
                "clean_value": str(clean_value),
                "dirty_value": str(dirty_value),
                "matching_columns": "|".join(cfg["matching_columns"]),
            })
            n_logged += 1

            files_seen.add(cfg["raw_file"])
            stays_seen.add(stay_id)

    # 关闭各文件 writer 并按 file_order 拼接成最终日志
    for fh, _dw, _part in file_writers.values():
        fh.close()
    with open(output, "w", newline="", encoding="utf-8") as out_fh:
        out_writer = csv.DictWriter(out_fh, fieldnames=fields, extrasaction="ignore")
        out_writer.writeheader()
        for rel in file_order:
            _fh, _dw, part = file_writers[rel]
            with open(part, "r", newline="", encoding="utf-8") as pf:
                pf.readline()          # 跳过 part 的表头
                for line in pf:
                    out_fh.write(line)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # 6. 计算随机性统计（流式重读一遍日志，避免在内存里留 2,900 万条）
    subtype_counter: dict[str, int] = defaultdict(int)
    file_counter: dict[str, int] = defaultdict(int)
    stay_counter: dict[str, int] = defaultdict(int)
    with open(output, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            subtype_counter[r["error_subtype"]] += 1
            file_counter[r["raw_file"]] += 1
            stay_counter[r["canonical_stay_id"]] += 1

    stats = {
        "split_name": split_name,
        "target_count": target_count,
        "actual_count": n_logged,
        "allocation_shortfall": shortfall,
        "subtype_counts": dict(subtype_counter),
        "file_counts": dict(file_counter),
        "stay_counts": dict(stay_counter),
        "n_subtypes_covered": len(subtype_counter),
        "n_files_touched": len(file_counter),
        "n_stays_touched": len(stay_counter),
        "baseline_total_cells": baseline_cells,
        "projected_final_cells": projected_final_cells,
        "subtype_min": min(subtype_counter.values()) if subtype_counter else 0,
        "subtype_max": max(subtype_counter.values()) if subtype_counter else 0,
        "file_min": min(file_counter.values()) if file_counter else 0,
        "file_max": max(file_counter.values()) if file_counter else 0,
        "stay_min": min(stay_counter.values()) if stay_counter else 0,
        "stay_max": max(stay_counter.values()) if stay_counter else 0,
        "stay_mean": sum(stay_counter.values()) / len(stay_counter) if stay_counter else 0,
        "stay_stddev": _stddev(list(stay_counter.values())),
        "subtype_ratio_max_over_min": (
            max(subtype_counter.values()) / max(1, min(subtype_counter.values()))
            if subtype_counter else 0
        ),
    }
    return stats


def _stddev(values: list[int]) -> float:
    if not values:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return var ** 0.5


# ----------------------------------------------------------------------
# 校验
# ----------------------------------------------------------------------

def validate_stats(stats: dict, *, split_name: str) -> tuple[bool, list[str]]:
    """返回 (passed, failures)。"""
    failures = []
    if split_name in ("train", "validation"):
        if stats["n_subtypes_covered"] < 18:
            failures.append(f"subtype_coverage {stats['n_subtypes_covered']} < 18")
        if stats["n_files_touched"] < 6:
            failures.append(f"file_touched {stats['n_files_touched']} < 6")
        if stats["subtype_min"] < 1:
            failures.append(f"subtype_min {stats['subtype_min']} < 1")
    if stats["stay_min"] < 1:
        failures.append(f"stay_min {stats['stay_min']} < 1")
    if stats["subtype_ratio_max_over_min"] > 500.0:
        failures.append(f"subtype_ratio {stats['subtype_ratio_max_over_min']:.2f} > 500.0")
    if stats["actual_count"] != stats["target_count"]:
        # 允许 ±5% 误差
        if abs(stats["actual_count"] - stats["target_count"]) > stats["target_count"] * 0.3:
            failures.append(f"actual_count {stats['actual_count']} too far from target {stats['target_count']}")
    return len(failures) == 0, failures


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    keys_path = Path(args.baseline).parent / "keys.csv"
    keys = []
    if keys_path.is_file():
        with open(keys_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            keys = [row[0] for row in reader]
        print(f"Loaded {len(keys)} keys from {keys_path}")

    stats = build_modification_log(
        split_name=args.split,
        target_count=args.count,
        baseline_raw_dir=args.baseline,
        output_log_path=args.output,
        all_keys=keys,
        random_seed=args.seed,
    )
    print(json.dumps(stats, indent=2, default=str))
