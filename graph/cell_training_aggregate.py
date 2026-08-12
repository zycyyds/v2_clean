from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FORMAL_SEEDS = (666, 667, 668)
METRIC_NAMES = (
    "average_precision",
    "auroc",
    "macro_f1",
    "dirty_precision",
    "dirty_recall",
    "dirty_f1",
)


class CellTrainingAggregateError(ValueError):
    pass


def _read_report(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "report.json"
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CellTrainingAggregateError(f"invalid training report: {resolved}") from exc
    if not isinstance(value, dict) or value.get("status") != "SUCCESS":
        raise CellTrainingAggregateError(f"training report is not SUCCESS: {resolved}")
    return resolved, value


def _summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise CellTrainingAggregateError("aggregate metrics must be finite")
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in {5, 32} or attempt == 5:
                raise
            time.sleep(0.025 * (2 ** attempt))


def aggregate_training_reports(
    report_paths: Iterable[str | Path],
    output_path: str | Path | None = None,
    *,
    require_formal_seeds: bool = True,
) -> dict[str, Any]:
    loaded = [_read_report(path) for path in report_paths]
    if not loaded:
        raise CellTrainingAggregateError("at least one training report is required")
    reports = [value for _, value in loaded]
    model_types = {str(value.get("model_type")) for value in reports}
    identities = {
        json.dumps(value.get("training_identity"), sort_keys=True) for value in reports
    }
    experiment_configs = {
        json.dumps(value.get("experiment_config"), sort_keys=True) for value in reports
    }
    seeds = [int(value.get("seed", -1)) for value in reports]
    if len(model_types) != 1:
        raise CellTrainingAggregateError("all reports must use the same model type")
    if len(identities) != 1:
        raise CellTrainingAggregateError("all reports must use the same training identity")
    if len(experiment_configs) != 1 or next(iter(experiment_configs)) == "null":
        raise CellTrainingAggregateError(
            "all reports must use the same complete experiment configuration"
        )
    if len(set(seeds)) != len(seeds):
        raise CellTrainingAggregateError("training report seeds must be unique")
    if require_formal_seeds and tuple(sorted(seeds)) != FORMAL_SEEDS:
        raise CellTrainingAggregateError(
            f"formal aggregate requires seeds {list(FORMAL_SEEDS)}, got {sorted(seeds)}"
        )
    if any(not value.get("internal_test_evaluated") for value in reports):
        raise CellTrainingAggregateError(
            "all reports must explicitly evaluate the frozen internal test"
        )

    split_names = ("validation", "internal_test")
    splits: dict[str, Any] = {}
    for split in split_names:
        splits[split] = {
            metric: _summary(
                float(value["splits"][split]["metrics"][metric]) for value in reports
            )
            for metric in METRIC_NAMES
        }
        splits[split]["confusion_matrix_total"] = {
            name: sum(
                int(value["splits"][split]["metrics"]["confusion_matrix"][name])
                for value in reports
            )
            for name in ("tn", "fp", "fn", "tp")
        }

    result = {
        "schema_version": 1,
        "status": "SUCCESS",
        "model_type": next(iter(model_types)),
        "seeds": sorted(seeds),
        "run_count": len(reports),
        "training_identity": reports[0]["training_identity"],
        "experiment_config": reports[0]["experiment_config"],
        "selection_metric": "validation_average_precision",
        "threshold_selection": "maximum_validation_macro_f1_after_epoch_selection",
        "threshold": _summary(float(value["validation_threshold"]) for value in reports),
        "best_epoch": _summary(float(value["best_epoch"]) for value in reports),
        "splits": splits,
        "runs": [
            {
                "seed": int(value["seed"]),
                "report": str(path),
                "validation_threshold": float(value["validation_threshold"]),
                "best_epoch": int(value["best_epoch"]),
            }
            for path, value in sorted(loaded, key=lambda item: int(item[1]["seed"]))
        ],
    }
    if output_path is not None:
        resolved_output = Path(output_path).expanduser().resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(resolved_output, result)
    return result
