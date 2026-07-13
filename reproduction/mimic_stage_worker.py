from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


DIAGNOSIS_MODE = "Convert ICD-9 to ICD-10 and group ICD-10 codes"


def _configure_imports(source_root: Path) -> None:
    paths = (
        source_root,
        source_root / "preprocessing" / "day_intervals_preproc",
        source_root / "preprocessing" / "hosp_module_preproc",
        source_root / "model",
    )
    for path in reversed(paths):
        sys.path.insert(0, str(path))


def _move_unchanged(source: Path, target: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(f"expected cohort output is missing: {source}")
    size = source.stat().st_size
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    shutil.move(str(source), str(target))
    if target.stat().st_size != size:
        raise RuntimeError(f"cohort relocation changed file size: {source} -> {target}")


def run_stage(
    stage: str,
    source_root: Path,
    run_root: Path,
    cohort_output: str,
    summary_output: str,
) -> None:
    os.chdir(run_root)
    _configure_imports(source_root)
    for relative in ("data/cohort", "data/features", "data/features/chartevents", "data/summary", "data/csv", "data/dict"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)

    if stage == "cohort":
        import day_intervals_cohort_v3

        actual_name = day_intervals_cohort_v3.extract_data(
            "ICU",
            "Mortality",
            0,
            "No Disease Filter",
            str(run_root),
            "",
            cohort_output=cohort_output,
            summary_output=summary_output,
        )
        if actual_name != cohort_output:
            raise RuntimeError(f"unexpected cohort name: {actual_name}")
        _move_unchanged(run_root / f"{cohort_output}.csv.gz", run_root / "data" / "cohort" / f"{cohort_output}.csv.gz")
        _move_unchanged(run_root / f"{summary_output}.txt", run_root / "data" / "cohort" / f"{summary_output}.txt")
        return

    if stage in {"features", "diagnosis", "summaries"}:
        import feature_selection_icu

        if stage == "features":
            feature_selection_icu.feature_icu(cohort_output, "mimiciv/3.1", True, True, True, True, True)
            extracted = run_root / "data" / "features" / "preproc_diag_icu.csv.gz"
            intermediate = run_root / "data" / "intermediate" / "preproc_diag_icu_extracted.csv.gz"
            intermediate.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(extracted, intermediate)
            if extracted.stat().st_size != intermediate.stat().st_size:
                raise RuntimeError("diagnosis intermediate copy changed file size")
        elif stage == "diagnosis":
            intermediate = run_root / "data" / "intermediate" / "preproc_diag_icu_extracted.csv.gz"
            extracted = run_root / "data" / "features" / "preproc_diag_icu.csv.gz"
            if not intermediate.is_file() or intermediate.stat().st_size == 0:
                raise RuntimeError(f"missing six-column diagnosis intermediate: {intermediate}")
            shutil.copy2(intermediate, extracted)
            feature_selection_icu.preprocess_features_icu(
                cohort_output,
                True,
                DIAGNOSIS_MODE,
                False,
                False,
                False,
                0,
                0,
            )
        else:
            feature_selection_icu.generate_summary_icu(True, True, True, True, True)
        return

    if stage == "stay_generation":
        import data_generation_icu

        data_generation_icu.Generator(
            cohort_output,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            72,
            1,
            2,
        )
        return
    raise ValueError(f"unsupported worker stage: {stage}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cohort-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--stage", choices=("cohort", "features", "diagnosis", "summaries", "stay_generation"), required=True)
    args = parser.parse_args(argv)
    run_stage(args.stage, args.source_root.resolve(), args.run_root.resolve(), args.cohort_output, args.summary_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
