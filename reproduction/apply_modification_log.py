"""Apply modification_log to raw csv files (流式处理，支持大文件)。

支持操作：
- cell_update：修改某行某列的值
- duplicate：复制某行，紧接其后插入
- insert：在某行后插入新行（用于 label_leakage_feature）

流式：每行读完立即处理 + 写，不强制 load 全表到内存。
用临时文件 + atomic rename 保证中断安全。
"""
from __future__ import annotations

import csv
import gzip
import shutil
from pathlib import Path


def open_csv(path: Path, mode: str = "r", newline: str = ""):
    """自动解压 .gz 文件。"""
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8", newline=newline)
    return open(path, mode, encoding="utf-8", newline=newline)


def apply_modification_log(raw_dir: str | Path, log_path: str | Path) -> dict:
    """流式应用 modification_log 到 raw_dir 下的 csv 文件。

    日志按 raw_file 分组（同文件的条目连续出现），因此这里顺序读日志、
    一次只驻留一个文件的条目 —— 2,900 万条日志一次性 list[dict] 会吃掉 29 GB。

    Returns
    -------
    stats : dict
    """
    raw_dir = Path(raw_dir).expanduser().resolve()
    log_path = Path(log_path).expanduser().resolve()

    n_cell_updates = 0
    n_duplicates = 0
    n_inserts = 0
    n_skipped = 0
    n_conflicts = 0
    file_stats: dict[str, int] = {}

    def process_file(relative: str, file_entries: list[tuple]) -> tuple[int, int, int, int]:
        """处理单个文件。返回 (cell_updates, duplicates, inserts, conflicts)。

        file_entries 是紧凑元组而不是原始 dict —— ICU5k test 的 chartevents
        有 1,500 万条日志，dict 形式要 15 GB，16 GB 机器直接 OOM。
        元组形式：(0, idx, column, dirty_value) / (1, idx, _) / (2, idx, dirty_value)
        """
        file_path = raw_dir / relative
        if not file_path.is_file():
            file_path_gz = raw_dir / (relative + ".gz")
            if file_path_gz.is_file():
                file_path = file_path_gz
            else:
                print(f"  [WARN] {relative} not found, skipping {len(file_entries)} entries")
                return 0, 0, 0, len(file_entries)

        # 建索引：row_index -> (column, dirty_value)
        update_idx: dict[int, dict[str, str]] = {}
        n_conflict = 0
        for e in file_entries:
            if e[0] != 0:
                continue
            idx, col, dv = e[1], e[2], e[3]
            slot = update_idx.setdefault(idx, {})
            if col in slot:
                # 同一 (行, 列) 被两个 subtype 命中 —— 后写覆盖先写，先写的那条静默失效
                n_conflict += 1
            slot[col] = dv

        # row_index -> insertion
        insert_after: dict[int, list[tuple]] = {}
        for e in file_entries:
            if e[0] != 2:
                continue
            insert_after.setdefault(e[1], []).append(e)

        # duplicate after: row_index -> count
        dup_after: dict[int, int] = {}
        for e in file_entries:
            if e[0] != 1:
                continue
            dup_after[e[1]] = dup_after.get(e[1], 0) + 1

        # 流式：写临时文件 → atomic 替换
        original_row_count = 0
        new_row_count = 0
        file_cell_updates = 0
        file_duplicates = 0
        file_inserts = 0

        # 写到 .tmp 文件，再 atomic rename（中断安全）
        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp_apply")
        try:
            with open_csv(file_path) as fin, open(tmp_path, "w") as fout:
                reader = csv.reader(fin)
                writer = csv.writer(fout)
                header = next(reader)
                writer.writerow(header)
                for row_idx, row_vals in enumerate(reader):
                    original_row_count += 1
                    # cell updates
                    if row_idx in update_idx:
                        upd = update_idx[row_idx]
                        for ci, col in enumerate(header):
                            if col in upd:
                                while len(row_vals) <= ci:
                                    row_vals.append("")
                                row_vals[ci] = upd[col]
                                file_cell_updates += 1

                    writer.writerow(row_vals)
                    new_row_count += 1

                    # duplicate after this row
                    if row_idx in dup_after:
                        for _ in range(dup_after[row_idx]):
                            writer.writerow(row_vals)
                            new_row_count += 1
                            file_duplicates += 1

                    # insert after this row
                    if row_idx in insert_after:
                        for ins_e in insert_after[row_idx]:
                            new_row = list(row_vals)
                            if "itemid" in header:
                                ci = header.index("itemid")
                                while len(new_row) <= ci:
                                    new_row.append("")
                                new_row[ci] = "999001"
                            if "valuenum" in header:
                                ci = header.index("valuenum")
                                while len(new_row) <= ci:
                                    new_row.append("")
                                new_row[ci] = (ins_e[3] or "1") if len(ins_e) > 3 else "1"
                            if "value" in header:
                                ci = header.index("value")
                                while len(new_row) <= ci:
                                    new_row.append("")
                                new_row[ci] = "1"
                            writer.writerow(new_row)
                            new_row_count += 1
                            file_inserts += 1

            # atomic rename
            shutil.move(str(tmp_path), str(file_path))
        except Exception:
            # 清理 tmp
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        file_stats[relative] = file_cell_updates + file_duplicates + file_inserts
        if new_row_count != original_row_count:
            print(f"  {relative}: rows {original_row_count} -> {new_row_count} (delta={new_row_count - original_row_count})")
        return file_cell_updates, file_duplicates, file_inserts, n_conflict

    # 顺序读日志：同文件的条目连续出现，一次只驻留一个文件
    # 条目转成紧凑元组（见 process_file 的说明），避免 1,500 万个 dict
    current_rel = None
    current_entries: list[tuple] = []
    with open(log_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for e in reader:
            rel = e["raw_file"]
            if rel != current_rel:
                if current_rel is not None:
                    cu, du, ins, cf = process_file(current_rel, current_entries)
                    n_cell_updates += cu
                    n_duplicates += du
                    n_inserts += ins
                    n_conflicts += cf
                current_rel = rel
                current_entries = []
            try:
                idx = int(e["raw_row_index"])
            except (ValueError, TypeError):
                n_skipped += 1
                continue
            op = e["operation"]
            if op == "cell_update":
                current_entries.append((0, idx, e["column"], e["dirty_value"]))
            elif op == "duplicate":
                current_entries.append((1, idx, "", ""))
            else:   # insert
                current_entries.append((2, idx, "", e["dirty_value"]))
        if current_rel is not None:
            cu, du, ins, cf = process_file(current_rel, current_entries)
            n_cell_updates += cu
            n_duplicates += du
            n_inserts += ins
            n_conflicts += cf
    n_skipped += n_conflicts

    return {
        "n_cell_updates": n_cell_updates,
        "n_duplicates": n_duplicates,
        "n_inserts": n_inserts,
        "n_skipped": n_skipped,
        "n_conflicts": n_conflicts,
        "by_file": file_stats,
        "n_files": len(file_stats),
    }


if __name__ == "__main__":
    import argparse
    import json
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--log", required=True)
    args = p.parse_args()
    stats = apply_modification_log(args.raw_dir, args.log)
    print(json.dumps(stats, indent=2))