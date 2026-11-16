#!/usr/bin/env python3
"""给 K29 1.6pct mimicrawerror 版生成 dataset manifest。

harness 的 `dataset_manifest` 只被检查「存在」并记进 run_manifest（路径 + sha256），
不影响评分 —— 但如果直接把 4.7pct 那份复制过来，里面记的 cell_level_error_rate 是
4.7pct 的值，档位信息就错了。

这里从**权威来源** `datasets/v2_rawerror_validation_report_1.6pct.json` 里
K29_v2_rawerror_test1.6pct_4.2M 那一条取真值，字段结构对齐 4.7pct 的
`v2_rawerror_k29_rebuild_final_report.json`（cell_level_error_rate 用**百分数**，
和 4.7pct 那份同单位，便于两个档位直接比）。
"""
from __future__ import annotations

import json
from pathlib import Path

PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
SRC_NAME = "K29_v2_rawerror_test1.6pct_4.2M"
REPORT = PROJ / "datasets/v2_rawerror_validation_report_1.6pct.json"
OUT = (PROJ / "datasets/注入错误数据集/k29"
       / "K29_v2_rawerror_test1.6pct_4.2M_mimicrawerror"
       / "v2_rawerror_k29_rebuild_final_report.json")
LOG = (PROJ / "datasets/注入错误数据集/k29"
       / SRC_NAME / "test/modification_log.csv")


def main() -> None:
    entries = json.loads(REPORT.read_text(encoding="utf-8"))
    entry = next(e for e in entries if e["name"] == SRC_NAME)
    t = entry["splits"]["test"]
    applied = t["applied"]

    rate_pct = round(t["cell_level_error_rate"] * 100, 2)
    ratio = t["subtype_ratio"]
    failures = []
    if ratio > 500.0:
        failures.append("subtype_ratio %.2f > 500.0" % ratio)
    gap = abs(t["actual_errors"] - t["target_errors"]) / max(1, t["target_errors"])
    if gap > 0.05:
        failures.append("actual_count %d too far from target %d"
                        % (t["actual_errors"], t["target_errors"]))

    manifest = {
        "name": SRC_NAME,
        "baseline": entry["old_baseline"],
        "injection_engine_version": entry.get("injection_engine_version"),
        "cell_rates": entry["cell_rates"],
        "comparison_with_v1": {
            "v1_400_800_1800_cell_level_pct_test": 0.0006,
            "v2_rebuild_cell_level_pct_test": rate_pct,
            "improvement_factor": round(rate_pct / 0.0006, 1),
            "note": "v1 = K29_clean_v2_rawerror_10train_400_800_1800（与 4.7pct 档同一份 v1 基线）",
        },
        "splits": {
            "test": {
                "target_errors": t["target_errors"],
                "actual_errors": t["actual_errors"],
                "subtype_coverage": t["subtype_coverage"],
                "subtype_ratio": ratio,
                "validation_passed": not failures,
                "validation_failures": failures,
                "applied": {
                    "n_cell_updates": applied["n_cell_updates"],
                    "n_duplicates": applied["n_duplicates"],
                    "n_inserts": applied.get("n_inserts", 0),
                    "n_skipped": applied.get("n_skipped", 0),
                    "by_file": applied["by_file"],
                },
                "total_cells": t["total_cells"],
                "cell_level_error_rate": rate_pct,
                "cell_level_error_rate_actual": rate_pct,
                "cell_level_error_rate_fraction": t["cell_level_error_rate"],
                "total_rows": t["total_rows"],
                "natural_noise_count": t["natural_noise_count"],
                "operation_counts": t["operation_counts"],
                "subtype_counts": t["subtype_counts"],
            },
        },
        "files": {
            "driver": str(PROJ / "reproduction/build_k29_rebuild_inject.py"),
            "config": str(PROJ / "reproduction/build_v2_error_configs.py"),
            "test_modification_log": str(LOG),
            "test_modification_log_size_bytes": LOG.stat().st_size,
            "authoritative_source": str(REPORT),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("写出:", OUT)
    print("  cell_level_error_rate = %s%%  (fraction %s)"
          % (rate_pct, t["cell_level_error_rate"]))
    print("  actual_errors = %s  total_cells = %s"
          % (t["actual_errors"], t["total_cells"]))
    print("  validation_failures =", failures)


if __name__ == "__main__":
    main()
