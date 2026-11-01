"""构建 **组密集** 的校正类切分（修 Raha curated 选法在「组级规则」任务上的失配）。

问题：`raha_curated10_v1` 的 train 选法是 `curated_greedy_changed_column_coverage_v1`
—— 贪心挑「能覆盖新的脏列」的行。flights 每行都覆盖全部 4 个脏列，
于是贪心退化成「取 10 个**不同** flight 各 1 行」⇒ **train 每个 flight 只有 1 行**，
**validation 每个 flight 只有 1~2 行**，而 test 每个 flight 有 ~24 行。
⇒ 「同一 flight 共享同一个值」这个结构在 train/val 上**根本看不出来**，
组级规则既学不到也验证不了。

本脚本按**分组**选：先选若干个 flight 组，每组取多行。
train / validation 的 flight 组互不相交，其余行进 correction(=test)。

用法:
  python3 build_group_dense_split.py --name flights --key-column tuple_id \
      --group-column flight --train-groups 2 --validation-groups 5 \
      --train-rows-per-group 5 --validation-rows-per-group 4 \
      --output-dir datasets/correction/flights_curated10_groupdense_v1
"""
import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

PROJ = Path("/Users/mac/PycharmProjects/v2_clean")


def load(p):
    with open(p, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def stable_key(s):
    return hashlib.sha256(f"group_dense_v1:{s}".encode("utf-8")).hexdigest()


def write_csv(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_keys(path, rows, key):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([key])
        for r in rows:
            w.writerow([r.get(key, "")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dirty-source", default="")
    ap.add_argument("--clean-source", default="")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--key-column", required=True)
    ap.add_argument("--group-column", required=True)
    ap.add_argument("--train-groups", type=int, default=2)
    ap.add_argument("--validation-groups", type=int, default=5)
    ap.add_argument("--train-rows-per-group", type=int, default=5)
    ap.add_argument("--validation-rows-per-group", type=int, default=4)
    args = ap.parse_args()

    dirty = Path(args.dirty_source) if args.dirty_source else PROJ / "datasets/_source" / args.name / "dirty.csv"
    clean = Path(args.clean_source) if args.clean_source else PROJ / "datasets/_source" / args.name / "clean.csv"
    out = Path(args.output_dir)
    if out.exists():
        raise SystemExit(f"输出目录已存在（不覆盖）：{out}")

    dcols, drows = load(dirty)
    ccols, crows = load(clean)
    clean_by_key = {c[args.key_column]: c for c in crows}

    # 每行改了哪些列
    changed = {}
    for d in drows:
        c = clean_by_key.get(d[args.key_column], {})
        changed[d[args.key_column]] = {
            dcols[i] for i in range(min(len(dcols), len(c)))
            if d.get(dcols[i], "") != c.get(ccols[i], "")
        }

    # 按组聚类
    groups = defaultdict(list)
    for d in drows:
        groups[d.get(args.group_column, "")].append(d)
    dirty_cols = sorted({c for s in changed.values() for c in s})

    def group_rank(g):
        """优先：能覆盖更多脏列的组；其次：组内行数多；再次：稳定哈希。"""
        cols = set()
        for r in groups[g]:
            cols |= changed.get(r[args.key_column], set())
        return (-len(cols), -len(groups[g]), stable_key(g))

    all_groups = sorted(groups, key=group_rank)

    def pick_groups(n, taken):
        out_g, covered = [], set()
        pool = [g for g in all_groups if g not in taken]
        while len(out_g) < n and pool:
            g = sorted(pool, key=lambda x: (
                -len({c for r in groups[x] for c in changed.get(r[args.key_column], set())} - covered),
                -len(groups[x]),
                stable_key(x),
            ))[0]
            out_g.append(g)
            pool.remove(g)
            taken.add(g)
            for r in groups[g]:
                covered |= changed.get(r[args.key_column], set())
        return out_g

    def take_rows(g, k, prefer_dirty=True):
        """从组里取 k 行：优先取脏列多的行（信息量最大），同分用稳定哈希。"""
        cand = sorted(groups[g], key=lambda r: (
            -len(changed.get(r[args.key_column], set())),
            stable_key(r[args.key_column]),
        ))
        return cand[:k]

    taken = set()
    train_groups = pick_groups(args.train_groups, taken)
    train_rows = []
    for g in train_groups:
        train_rows += take_rows(g, args.train_rows_per_group)
    val_groups = pick_groups(args.validation_groups, taken)
    val_rows = []
    for g in val_groups:
        val_rows += take_rows(g, args.validation_rows_per_group)

    train_keys = {r[args.key_column] for r in train_rows}
    val_keys = {r[args.key_column] for r in val_rows}
    held = train_keys | val_keys
    corr_rows = [r for r in drows if r[args.key_column] not in held]

    def clean_of(rows):
        return [clean_by_key[r[args.key_column]] for r in rows]

    write_csv(out / "train/raw" / f"{args.name}.csv", dcols, train_rows)
    write_csv(out / "train/reference" / f"{args.name}.csv", dcols, clean_of(train_rows))
    write_keys(out / "train/keys.csv", train_rows, args.key_column)
    write_csv(out / "validation/raw" / f"{args.name}.csv", dcols, val_rows)
    write_csv(out / "validation/gold" / f"{args.name}.csv", dcols, clean_of(val_rows))
    write_keys(out / "validation/keys.csv", val_rows, args.key_column)
    write_csv(out / "correction/raw" / f"{args.name}.csv", dcols, corr_rows)
    write_csv(out / "correction/reference_private" / f"{args.name}.csv", dcols,
              [clean_by_key[r[args.key_column]] for r in corr_rows])
    write_keys(out / "correction/keys.csv", corr_rows, args.key_column)

    def stat(rows):
        cells = sum(len(changed.get(r[args.key_column], set())) for r in rows)
        g = Counter(r.get(args.group_column, "") for r in rows)
        return dict(rows=len(rows), dirty_cells=cells,
                    groups=len(g), max_group=max(g.values()) if g else 0,
                    min_group=min(g.values()) if g else 0)

    manifest = {
        "schema_version": 1,
        "status": "SUCCESS",
        "benchmark_label": "curated 10-shot GROUP-DENSE (group structure visible in train/val)",
        "selection": "group_dense_v1",
        "selection_note": ("train/validation 按 %s 分组选（每组多行），"
                           "让「同组共享常量」这类组级规则在 train 上可学、在 validation 上可验证；"
                           "原 curated 选法（按脏列覆盖贪心）在单表组级任务上会让每个组只剩 1 行。"
                           % args.group_column),
        "key_column": args.key_column,
        "group_column": args.group_column,
        "counts": {"all": len(drows),
                   "train": len(train_rows),
                   "validation": len(val_rows),
                   "correction": len(corr_rows)},
        "difference_counts": {
            "cells": sum(len(s) for s in changed.values()),
            "train_cells": sum(len(changed.get(r[args.key_column], set())) for r in train_rows),
            "validation_cells": sum(len(changed.get(r[args.key_column], set())) for r in val_rows),
            "correction_cells": sum(len(changed.get(r[args.key_column], set())) for r in corr_rows),
        },
        "train_groups": train_groups,
        "validation_groups": val_groups,
        "dirty_columns": dirty_cols,
        "splits": {"train": stat(train_rows), "validation": stat(val_rows),
                   "correction": stat(corr_rows)},
        "paths": {"train_raw": "train/raw", "train_reference": "train/reference",
                  "validation_raw": "validation/raw", "validation_gold": "validation/gold",
                  "correction_raw": "correction/raw",
                  "correction_reference_private": "correction/reference_private"},
    }
    (out / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in
                      ("counts", "difference_counts", "train_groups", "validation_groups",
                       "splits")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
