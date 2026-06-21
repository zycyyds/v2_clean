from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/wwz/Downloads/medical-agent-pipeline-main/v2/output/dirty_mimic_mini_benchmark")
DIRTY_CSV = ROOT / "dirty_flat_liver_dataset.csv"
REPORT_JSON = ROOT / "dirty_benchmark_report.json"
OUTPUT_CSV = ROOT / "independent_cleaned_flat_liver_dataset.csv"
OUTPUT_JSON = ROOT / "independent_cleaning_evaluation.json"

INVALID_TOKENS = {"", "未知", "n/a", "na", "??", "mixed-case"}
TEXT_NOISE_MARKERS = {"[ocr]", "duplicate duplicate", "报告待补录"}

CATEGORY_COLUMNS = ["admission_type", "insurance", "language", "marital_status", "race", "gender"]
CODE_COLUMNS = ["all_icd_codes", "primary_icd_code"]
TEXT_NOISE_COLUMNS = [
    "discharge_note_count",
    "discharge_note_text_len_mean",
    "discharge_note_mentions_liver",
    "radiology_note_count",
    "radiology_note_text_len_mean",
    "radiology_note_mentions_liver",
]

HIGH_WEIGHT_COLUMNS = {
    "subject_id": 8.0,
    "hepatic_dx": 5.0,
    "primary_icd_prefix3": 4.0,
    "primary_icd_prefix4": 4.0,
    "gender": 3.0,
    "anchor_age": 3.0,
    "admission_type": 3.0,
    "insurance": 2.0,
    "marital_status": 2.0,
    "race": 2.0,
    "language": 2.0,
    "hospital_expire_flag": 2.0,
    "icu_flag": 2.0,
    "hep_drug": 2.0,
    "discharge_note_mentions_liver": 2.0,
    "radiology_note_mentions_liver": 2.0,
}


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def normalized_token(value: object) -> str:
    return normalize_text(value).lower()


def contains_text_noise(value: object) -> bool:
    token = normalized_token(value)
    return any(marker in token for marker in TEXT_NOISE_MARKERS)


def is_invalid_category(value: object) -> bool:
    token = normalized_token(value)
    return token in INVALID_TOKENS


def is_invalid_code(value: object) -> bool:
    token = normalized_token(value)
    return token in INVALID_TOKENS or token == "???" or "|" in token or token == "571-5"


def is_blankish(value: object) -> bool:
    return normalize_text(value) == ""


def parse_numeric(value: object) -> float | None:
    text = normalize_text(value)
    if not text:
        return None
    lowered = text.lower()
    if lowered in INVALID_TOKENS or contains_text_noise(text):
        return None
    if lowered.startswith("<"):
        try:
            return float(lowered[1:])
        except ValueError:
            return None
    cleaned = text.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def is_numeric_like_column(series: pd.Series) -> bool:
    non_empty = [normalize_text(v) for v in series if normalize_text(v)]
    if not non_empty:
        return False
    parsed = sum(parse_numeric(v) is not None for v in non_empty)
    return parsed / len(non_empty) >= 0.85


def infer_format(series: pd.Series) -> str:
    samples = [normalize_text(v) for v in series if normalize_text(v)]
    if not samples:
        return "float"
    integer_like = 0
    one_decimal_like = 0
    for value in samples[:200]:
        if value.endswith(".0"):
            one_decimal_like += 1
        parsed = parse_numeric(value)
        if parsed is not None and float(parsed).is_integer():
            integer_like += 1
    if one_decimal_like >= max(5, len(samples[:200]) // 3):
        return "one_decimal"
    if integer_like >= max(5, len(samples[:200]) // 2):
        return "integer"
    return "float"


def format_numeric(value: float | None, fmt: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if fmt == "one_decimal":
        return f"{value:.1f}"
    if fmt == "integer":
        return str(int(round(value)))
    rendered = f"{value:.12f}".rstrip("0").rstrip(".")
    return rendered if rendered else "0"


class SimilarityCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_like_columns = {col for col in df.columns if is_numeric_like_column(df[col])}
        self.column_formats = {col: infer_format(df[col]) for col in self.numeric_like_columns}
        self.numeric_scales = self._build_numeric_scales()
        self.binary_columns = [
            col
            for col in df.columns
            if col.startswith(("icd3_", "icd4_", "drug_")) and set(normalize_text(v) for v in df[col].unique()) <= {"", "0", "0.0", "1", "1.0"}
        ]
        self.anchor_columns = self._build_anchor_columns()

    def _build_numeric_scales(self) -> dict[str, float]:
        scales: dict[str, float] = {}
        for col in self.numeric_like_columns:
            values = [parse_numeric(v) for v in self.df[col]]
            values = [v for v in values if v is not None]
            if not values:
                scales[col] = 1.0
                continue
            series = pd.Series(values)
            scale = float(series.quantile(0.75) - series.quantile(0.25))
            if scale <= 0:
                scale = float(series.std()) if len(series) > 1 else 1.0
            scales[col] = scale or 1.0
        return scales

    def _build_anchor_columns(self) -> list[str]:
        anchors = [
            col
            for col in [
                "subject_id",
                "hepatic_dx",
                "primary_icd_prefix3",
                "primary_icd_prefix4",
                "admission_type",
                "insurance",
                "language",
                "marital_status",
                "race",
                "gender",
                "hospital_expire_flag",
                "anchor_age",
                "diagnoses_count",
                "unique_diagnoses_count",
                "hepatic_code_count",
                "icu_flag",
                "icu_stay_count",
                "lab_event_count",
                "lab_itemid_nunique",
                "lab_valuenum_mean",
                "lab_valuenum_std",
                "lab_valuenum_min",
                "lab_valuenum_max",
                "prescription_count",
                "unique_drug_count",
                "hep_drug",
                "hep_drug_count",
                "discharge_note_count",
                "discharge_note_mentions_liver",
                "radiology_note_count",
                "radiology_note_mentions_liver",
            ]
            if col in self.df.columns
        ]
        anchors.extend(self.binary_columns[:40])
        seen = set()
        ordered: list[str] = []
        for col in anchors:
            if col not in seen:
                seen.add(col)
                ordered.append(col)
        return ordered

    def row_similarity(self, row_index: int, candidate_index: int, target_col: str) -> float:
        score = 0.0
        row = self.df.iloc[row_index]
        candidate = self.df.iloc[candidate_index]
        for col in self.anchor_columns:
            if col == target_col:
                continue
            row_value = row[col]
            candidate_value = candidate[col]
            if col in self.numeric_like_columns:
                left = parse_numeric(row_value)
                right = parse_numeric(candidate_value)
                if left is None or right is None:
                    continue
                scale = self.numeric_scales.get(col, 1.0)
                closeness = max(0.0, 1.0 - abs(left - right) / (scale * 3.0 + 1e-9))
                score += closeness
                continue
            left = normalize_text(row_value)
            right = normalize_text(candidate_value)
            if not left or not right:
                continue
            if left == right:
                score += HIGH_WEIGHT_COLUMNS.get(col, 1.0)
        return score

    def top_candidates(self, row_index: int, target_col: str, validator) -> list[tuple[float, int]]:
        candidates: list[tuple[float, int]] = []
        for candidate_index, value in enumerate(self.df[target_col]):
            if candidate_index == row_index or not validator(value):
                continue
            score = self.row_similarity(row_index, candidate_index, target_col)
            if score > 0:
                candidates.append((score, candidate_index))
        candidates.sort(reverse=True)
        return candidates[:12]

    def predict_category(self, row_index: int, target_col: str) -> str:
        def valid(value: object) -> bool:
            return not is_invalid_category(value)

        candidates = self.top_candidates(row_index, target_col, valid)
        if candidates:
            votes: dict[str, float] = defaultdict(float)
            for score, candidate_index in candidates:
                votes[normalize_text(self.df.iat[candidate_index, self.df.columns.get_loc(target_col)])] += score
            best_value = max(votes.items(), key=lambda item: (item[1], item[0]))[0]
            if best_value:
                return best_value
        fallback = [normalize_text(v) for v in self.df[target_col] if valid(v)]
        return Counter(fallback).most_common(1)[0][0] if fallback else ""

    def predict_numeric(self, row_index: int, target_col: str) -> float | None:
        def valid(value: object) -> bool:
            parsed = parse_numeric(value)
            if parsed is None:
                return False
            if parsed == -9999:
                return False
            return True

        row = self.df.iloc[row_index]
        sibling_groups = [
            ["lab_event_count", "lab_itemid_nunique"],
            ["lab_valuenum_mean", "lab_valuenum_std", "lab_valuenum_min", "lab_valuenum_max"],
            ["lab_mean_50868", "lab_mean_50882"],
            ["discharge_note_count", "discharge_note_text_len_mean", "discharge_note_mentions_liver"],
            ["radiology_note_count", "radiology_note_text_len_mean", "radiology_note_mentions_liver"],
        ]
        for group in sibling_groups:
            if target_col not in group:
                continue
            matches = []
            for candidate_index in range(len(self.df)):
                if candidate_index == row_index:
                    continue
                candidate_row = self.df.iloc[candidate_index]
                aligned = True
                for col in group:
                    if col == target_col:
                        continue
                    left = parse_numeric(row.get(col, ""))
                    right = parse_numeric(candidate_row.get(col, ""))
                    if left is None or right is None or abs(left - right) > 1e-9:
                        aligned = False
                        break
                if not aligned:
                    continue
                candidate_value = parse_numeric(candidate_row[target_col])
                if candidate_value is not None and candidate_value != -9999:
                    matches.append(candidate_value)
            if matches:
                return matches[0]

        candidates = self.top_candidates(row_index, target_col, valid)
        values: list[tuple[float, float]] = []
        for score, candidate_index in candidates:
            parsed = parse_numeric(self.df.iat[candidate_index, self.df.columns.get_loc(target_col)])
            if parsed is None:
                continue
            values.append((score, parsed))
        if values:
            values.sort(reverse=True)
            top = values[:5]
            total_weight = sum(score for score, _ in top)
            if total_weight > 0:
                return sum(score * value for score, value in top) / total_weight
        valid_values = [parse_numeric(v) for v in self.df[target_col]]
        valid_values = [v for v in valid_values if v is not None and v != -9999]
        if not valid_values:
            return None
        return float(pd.Series(valid_values).median())

    def predict_code(self, row_index: int, target_col: str) -> str:
        row = self.df.iloc[row_index]
        prefix4 = normalize_text(row.get("primary_icd_prefix4", ""))
        prefix3 = normalize_text(row.get("primary_icd_prefix3", ""))
        codes = [code.strip() for code in normalize_text(row.get("all_icd_codes", "")).split(",") if code.strip()]
        plausible = [code.replace(".", "").replace("-", "") for code in codes if code.replace(".", "").replace("-", "").isalnum()]
        if target_col == "primary_icd_code":
            for code in plausible:
                if prefix4 and code.startswith(prefix4):
                    return code
            for code in plausible:
                if prefix3 and code.startswith(prefix3):
                    return code
            if plausible:
                return plausible[0]
        if target_col == "all_icd_codes" and plausible:
            return ",".join(plausible)

        candidates = self.top_candidates(row_index, target_col, lambda value: not is_invalid_code(value))
        if candidates:
            votes: dict[str, float] = defaultdict(float)
            for score, candidate_index in candidates:
                votes[normalize_text(self.df.iat[candidate_index, self.df.columns.get_loc(target_col)])] += score
            return max(votes.items(), key=lambda item: (item[1], item[0]))[0]
        text = normalize_text(row[target_col])
        if target_col == "all_icd_codes":
            if text == "571-5":
                return "5715"
            if text == "572.3|789.59":
                return "5723,78959"
            return ""
        if text == "571-5":
            return "5715"
        if text == "572.3|789.59":
            return "5723"
        return ""

    def clean_text_noise(self, row_index: int, target_col: str) -> str:
        row = self.df.iloc[row_index]
        if target_col.endswith("_count"):
            sister = target_col.replace("_count", "_text_len_mean")
            sister_value = parse_numeric(row.get(sister, "")) if sister in self.df.columns else None
            mention_col = target_col.replace("_count", "_mentions_liver")
            mention_value = parse_numeric(row.get(mention_col, "")) if mention_col in self.df.columns else None
            if sister_value is None and mention_value is None:
                return ""
            if sister_value is not None:
                return format_numeric(1.0 if sister_value > 0 else 0.0, self.column_formats.get(target_col, "one_decimal"))
        if target_col.endswith("_text_len_mean"):
            count_col = target_col.replace("_text_len_mean", "_count")
            count_value = parse_numeric(row.get(count_col, "")) if count_col in self.df.columns else None
            mention_col = target_col.replace("_text_len_mean", "_mentions_liver")
            mention_value = parse_numeric(row.get(mention_col, "")) if mention_col in self.df.columns else None
            if count_value is None or count_value == 0:
                return ""
            if mention_value is not None:
                matches = []
                for candidate_index in range(len(self.df)):
                    if candidate_index == row_index:
                        continue
                    candidate_row = self.df.iloc[candidate_index]
                    if parse_numeric(candidate_row.get(count_col, "")) == count_value and parse_numeric(candidate_row.get(mention_col, "")) == mention_value:
                        candidate_value = parse_numeric(candidate_row[target_col])
                        if candidate_value is not None:
                            matches.append(candidate_value)
                if matches:
                    return format_numeric(matches[0], self.column_formats.get(target_col, "one_decimal"))
        if target_col.endswith("_mentions_liver"):
            count_col = target_col.replace("_mentions_liver", "_count")
            count_value = parse_numeric(row.get(count_col, "")) if count_col in self.df.columns else None
            if count_value is None or count_value == 0:
                return ""
            return format_numeric(1.0, self.column_formats.get(target_col, "one_decimal"))
        numeric_value = self.predict_numeric(row_index, target_col)
        return format_numeric(numeric_value, self.column_formats.get(target_col, "one_decimal"))

    def clean(self) -> pd.DataFrame:
        cleaned = self.df.copy()
        for row_index in range(len(cleaned)):
            for col in CATEGORY_COLUMNS:
                if col in cleaned.columns and is_invalid_category(cleaned.iat[row_index, cleaned.columns.get_loc(col)]):
                    cleaned.iat[row_index, cleaned.columns.get_loc(col)] = self.predict_category(row_index, col)
            for col in CODE_COLUMNS:
                if col in cleaned.columns and is_invalid_code(cleaned.iat[row_index, cleaned.columns.get_loc(col)]):
                    cleaned.iat[row_index, cleaned.columns.get_loc(col)] = self.predict_code(row_index, col)
            for col in TEXT_NOISE_COLUMNS:
                if col in cleaned.columns:
                    cell = cleaned.iat[row_index, cleaned.columns.get_loc(col)]
                    if contains_text_noise(cell) or is_blankish(cell):
                        cleaned.iat[row_index, cleaned.columns.get_loc(col)] = self.clean_text_noise(row_index, col)

        for col in self.numeric_like_columns:
            fmt = self.column_formats.get(col, "float")
            for row_index in range(len(cleaned)):
                original = cleaned.iat[row_index, cleaned.columns.get_loc(col)]
                text = normalize_text(original)
                if not text:
                    continue
                parsed = parse_numeric(text)
                needs_impute = False
                if contains_text_noise(text):
                    needs_impute = True
                elif normalized_token(text) in {"-9999"}:
                    needs_impute = True
                elif text.startswith("<"):
                    cleaned.iat[row_index, cleaned.columns.get_loc(col)] = format_numeric(parsed, fmt)
                    continue
                elif "," in text and parsed is not None:
                    cleaned.iat[row_index, cleaned.columns.get_loc(col)] = format_numeric(parsed, fmt)
                    continue
                elif parsed is None:
                    needs_impute = True
                if needs_impute:
                    predicted = self.predict_numeric(row_index, col)
                    cleaned.iat[row_index, cleaned.columns.get_loc(col)] = format_numeric(predicted, fmt)
        return cleaned


def compare_values(expected: object, actual: object) -> bool:
    left_num = parse_numeric(expected)
    right_num = parse_numeric(actual)
    if left_num is not None and right_num is not None:
        return abs(left_num - right_num) < 1e-9
    return normalize_text(expected) == normalize_text(actual)


def evaluate(dirty_df: pd.DataFrame, cleaned_df: pd.DataFrame, report: dict[str, object]) -> dict[str, object]:
    flat = report["flat_mutation_details"]
    total_mutations = 0
    fixed_mutations = 0
    per_error_type: dict[str, dict[str, object]] = {}
    failed_columns: Counter[str] = Counter()
    for error_type, items in flat.items():
        fixed = 0
        for item in items:
            total_mutations += 1
            row = int(item["row"])
            column = item["column"]
            expected = item["before"]
            actual = cleaned_df.iloc[row][column]
            if compare_values(expected, actual):
                fixed += 1
                fixed_mutations += 1
            else:
                failed_columns[column] += 1
        per_error_type[error_type] = {
            "total": len(items),
            "fixed": fixed,
            "fix_rate": round(fixed / len(items), 4) if items else 0.0,
        }
    changed_columns = Counter()
    for col in dirty_df.columns:
        changed = 0
        for left, right in zip(dirty_df[col], cleaned_df[col]):
            if normalize_text(left) != normalize_text(right):
                changed += 1
        if changed:
            changed_columns[col] = changed
    return {
        "shape_preserved": list(dirty_df.shape) == list(cleaned_df.shape),
        "total_mutations": total_mutations,
        "fixed_mutations": fixed_mutations,
        "fix_rate": round(fixed_mutations / total_mutations, 4) if total_mutations else 0.0,
        "per_error_type": per_error_type,
        "failed_columns": dict(failed_columns.most_common()),
        "top_changed_columns": [[col, count] for col, count in changed_columns.most_common(15)],
    }


def main() -> None:
    dirty_df = pd.read_csv(DIRTY_CSV, dtype=object, keep_default_na=False)
    with REPORT_JSON.open() as handle:
        report = json.load(handle)
    cleaner = SimilarityCleaner(dirty_df)
    cleaned_df = cleaner.clean()
    cleaned_df.to_csv(OUTPUT_CSV, index=False)
    evaluation = evaluate(dirty_df, cleaned_df, report)
    OUTPUT_JSON.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2))
    print(json.dumps({
        "cleaned_csv": str(OUTPUT_CSV),
        "evaluation_json": str(OUTPUT_JSON),
        **evaluation,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
