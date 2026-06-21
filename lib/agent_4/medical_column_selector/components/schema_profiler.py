"""表结构分析器：给每一列生成基础画像。"""
import os
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import re


VALID_MODALITIES = [
    "Identifiers/Keys", "Time", "Demographics", "Encounter",
    "Diagnosis", "Medication", "Labs", "Vitals",
    "Procedures", "Outcomes", "ClinicalNotes"
]


class SchemaProfiler:
    """分析输入表格，并输出每列的统计信息和 modality。"""

    def __init__(self, max_rows: int = 1000, sampling_method: str = "random", random_seed: int = 42):
        """初始化采样配置。"""
        self.max_rows = max_rows
        self.sampling_method = sampling_method
        self.random_seed = random_seed

    def analyze_csv(self, csv_path: str) -> Dict[str, Any]:
        """兼容旧接口。"""
        return self.analyze_file(csv_path)

    def analyze_file(self, input_path: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:
        """读取表格，并为每一列生成画像。"""
        df_full = self._read_tabular_file(input_path, sheet_name=sheet_name)
        df = self._sample_dataframe(df_full)

        schema_profile = {
            "dataset_info": {
                "total_rows": len(df_full),
                "sampled_rows": len(df),
                "total_columns": len(df.columns),
                "profile_generated_at": datetime.now().isoformat(),
            },
            "columns": {},
        }

        for col in df.columns:
            schema_profile["columns"][col] = self._analyze_column(df[col], df_full[col])

        return schema_profile

    def _sample_dataframe(self, df_full: pd.DataFrame) -> pd.DataFrame:
        """大表只抽一部分行做分析，避免无意义地全表扫描。"""
        if len(df_full) <= self.max_rows:
            return df_full.copy()
        if self.sampling_method == "random":
            return df_full.sample(n=self.max_rows, random_state=self.random_seed)
        if self.sampling_method == "tail":
            return df_full.tail(self.max_rows)
        return df_full.head(self.max_rows)

    def _read_tabular_file(self, input_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """统一读取 CSV / Excel。"""
        ext = os.path.splitext(input_path)[1].lower()

        if ext == ".csv":
            return pd.read_csv(input_path)
        if ext in {".xlsx", ".xls"}:
            return pd.read_excel(input_path, sheet_name=sheet_name or 0)

        raise ValueError(f"Unsupported input file type: {ext}. Expected .csv, .xlsx, or .xls")

    def _analyze_column(self, col_sample: pd.Series, col_full: pd.Series) -> Dict[str, Any]:
        """为单列生成画像。"""
        dtype_guess = str(col_sample.dtype)
        missing_rate = col_full.isna().sum() / len(col_full)
        unique_ratio = col_full.nunique() / len(col_full) if len(col_full) > 0 else 0

        is_phi = self._is_potential_phi(col_sample.name)
        examples = self._get_safe_examples(col_sample.dropna(), is_phi)
        numeric_stats = {}
        if pd.api.types.is_numeric_dtype(col_sample):
            numeric_col = pd.to_numeric(col_sample, errors='coerce')
            if pd.api.types.is_bool_dtype(col_sample):
                numeric_col = numeric_col.astype(float)
            numeric_stats = {
                "mean": float(numeric_col.mean()) if not numeric_col.empty else None,
                "std": float(numeric_col.std()) if not numeric_col.empty else None,
                "min": float(numeric_col.min()) if not numeric_col.empty else None,
                "max": float(numeric_col.max()) if not numeric_col.empty else None,
                "q25": float(numeric_col.quantile(0.25)) if not numeric_col.empty else None,
                "q50": float(numeric_col.quantile(0.50)) if not numeric_col.empty else None,
                "q75": float(numeric_col.quantile(0.75)) if not numeric_col.empty else None,
            }

        inferred_modality = self._infer_modality(col_sample.name.lower())

        return {
            "name": col_sample.name,
            "dtype_guess": dtype_guess,
            "missing_rate": missing_rate,
            "unique_ratio": unique_ratio,
            "examples": examples,
            "numeric_stats": numeric_stats,
            "is_potential_phi": is_phi,
            "inferred_modality": inferred_modality,
            "is_constant": len(col_full.unique()) == 1 if len(col_full) > 0 else True,
        }

    def _is_potential_phi(self, col_name: str) -> bool:
        """根据列名判断是否可能包含直接隐私信息。"""
        col_lower = col_name.lower()

        # 常见连接键和文件元数据默认不按 PHI 处理。
        if self._is_operational_metadata_name(col_lower):
            return False
        if self._is_task_supporting_identifier_name(col_lower):
            return False

        phi_patterns = [
            r'(^|[^a-z])name([^a-z]|$)', r'first.*name', r'last.*name',
            r'full.*name', r'fname', r'lname',
            r'address', r'phone', r'email', r'contact',
            r'ssn', r'mrn', r'medical.*record', r'insurance.*id',
            r'dob', r'date.*birth', r'birthday', r'birth.*date',
            r'account.*number', r'license', r'passport', r'zip.*code',
            r'location', r'city', r'state', r'country', r'county'
        ]

        for pattern in phi_patterns:
            if re.search(pattern, col_lower):
                return True
        return False

    def _is_operational_metadata_name(self, col_lower: str) -> bool:
        """文件级元数据一般不视为 PHI。"""
        return col_lower in {
            "file",
            "filename",
            "file_name",
            "filepath",
            "file_path",
            "path",
            "source_file",
            "source_filename",
        }

    def _is_task_supporting_identifier_name(self, col_lower: str) -> bool:
        """常见数据集主键默认保留，方便 join / 去重 / 分组。"""
        sensitive_identifier_tokens = [
            "mrn", "medical_record", "insurance", "policy", "account",
            "license", "passport", "ssn"
        ]
        if any(token in col_lower for token in sensitive_identifier_tokens):
            return False

        if col_lower in {"id", "pid", "eid", "patient_id"}:
            return True

        return bool(re.search(
            r'(^|_)(patient|subject|study|record|encounter|visit|case|episode|stay|sample|specimen|accession|order)_?id$',
            col_lower
        ))

    def _get_safe_examples(self, series: pd.Series, is_phi: bool) -> List[str]:
        """取少量示例值；如果是 PHI，则先打码。"""
        if len(series) == 0:
            return []

        examples = []
        for val in series.head(5):
            str_val = str(val)
            if is_phi:
                if self._looks_like_date(str_val):
                    examples.append("<DATE_REDACTED>")
                elif str_val.replace("-", "").replace(".", "").replace(",", "").isdigit() or self._looks_like_id(str_val):
                    examples.append("<ID_REDACTED>")
                else:
                    examples.append("<TEXT_REDACTED>")
            else:
                examples.append(str_val)
        return examples

    def _looks_like_date(self, value: str) -> bool:
        """判断字符串是否像日期。"""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}/\d{2}/\d{2}'   # YYYY/MM/DD
        ]
        for pattern in date_patterns:
            if re.match(pattern, value.strip()):
                return True
        return False

    def _looks_like_id(self, value: str) -> bool:
        """判断字符串是否像编号。"""
        digit_ratio = sum(c.isdigit() for c in value) / len(value) if len(value) > 0 else 0
        return digit_ratio > 0.7 and len(value) >= 3

    def _infer_modality(self, col_name: str) -> Optional[str]:
        """根据列名推断医学数据类型。"""
        if self._is_operational_metadata_name(col_name):
            return None

        structured_modality = self._infer_structured_table_modality(col_name)
        if structured_modality:
            return structured_modality

        modality_keywords = {
            "Identifiers/Keys": [
                "id", "identifier", "patient_id", "mrn", "study_id", "subject_id",
                "record_id", "encounter_id", "visit_id", "pid", "eid",
                "患者id", "病历号", "住院号", "就诊号", "病例号"
            ],
            "Time": [
                "date", "time", "datetime", "timestamp", "admission", "discharge",
                "visit", "encounter", "onset", "death", "event", "follow_up",
                "birth", "dob", "age", "duration", "interval", "day", "month",
                "year", "hour", "minute", "second",
                "日期", "时间", "病程", "发作", "出生年月", "诊断日期", "持续时间"
            ],
            "Demographics": [
                "sex", "gender", "race", "ethnicity", "language", "religion",
                "marital", "occupation", "education", "income", "pregnant",
                "deceased", "veteran", "smoking", "alcohol", "height", "weight",
                "bmi", "blood_type",
                "一般资料", "性别", "年龄", "民族", "职业", "工作性质", "婚育", "婚姻"
            ],
            "Encounter": [
                "encounter", "visit", "admission", "discharge", "transfer",
                "provider", "attending", "room", "bed", "ward", "department",
                "location", "clinic", "hospital", "emergency", "icu", "ccu",
                "unit", "service", "disposition", "admit_type", "discharge_disposition",
                "门诊", "住院", "入院", "出院", "科室", "复诊"
            ],
            "Diagnosis": [
                "diagnosis", "dx", "icd", "problem", "condition", "comorbidity",
                "primary_dx", "secondary_dx", "icd9", "icd10", "icd11", "snomed",
                "chief_complaint", "presenting", "symptom", "sign", "manifestation",
                "症状学", "症状", "诊断", "综合征", "定位诊断", "病史", "眩晕", "头晕",
                "耳鸣", "头痛", "复视", "恶心", "呕吐", "平衡障碍", "前庭视觉"
            ],
            "Medication": [
                "medication", "drug", "med", "rx", "prescription", "order",
                "dosage", "dose", "frequency", "route", "ingredient", "atc",
                "ndc", "generic", "brand", "administration", "infusion",
                "dispense", "pharmacy", "medication_order", "medication_administration",
                "药物", "用药", "给药", "处方", "药名"
            ],
            "Labs": [
                "lab", "test", "value", "result", "panel", "loinc", "units",
                "abnormal", "flag", "reference", "normal", "range", "micro",
                "culture", "sensitivity", "pathogen", "organism", "antibiotic",
                "culture", "sensitivity", "gram", "morphology",
                "辅助检查", "实验室检查", "检查", "检验", "试验", "结果", "数值",
                "眼震", "前庭", "冷热试验", "角速度", "偏差", "标准差", "翻滚角",
                "hit试验", "dix-hallpike", "roll-test", "labvalue", "test_抽取"
            ],
            "Vitals": [
                "vital", "bp", "blood_pressure", "heart_rate", "pulse",
                "temperature", "temp", "oxygen", "saturation", "resp", "rate",
                "breath", "rr", "hr", "sys", "dia", "map", "etco2", "pao2",
                "ph", "po2", "pc", "inhale", "fi", "weight", "height", "bmi",
                "生命体征", "血压", "心率", "脉搏", "体温", "呼吸"
            ],
            "Procedures": [
                "procedure", "proc", "cpt", "procedure_code", "surgery",
                "operation", "intervention", "treatment", "therapy", "icd_pcs",
                "service", "order", "status", "provider", "physician", "surgeon",
                "anesthesia", "approach", "device", "implant", "graft",
                "床旁查体", "查体", "治疗", "手术", "干预", "操作"
            ],
            "Outcomes": [
                "outcome", "mortality", "death", "survival", "readmission",
                "complication", "recovery", "failure", "success", "response",
                "remission", "relapse", "progression", "recurrence", "survived",
                "alive", "dead", "expired", "los", "length_of_stay",
                "结局", "转归", "死亡", "复发", "再入院"
            ],
            "ClinicalNotes": [
                "note", "text", "clinical_note", "discharge_summary",
                "radiology", "imaging", "ecg", "ekg", "echo", "procedure_note",
                "consult", "progress", "assessment", "plan", "history",
                "physical", "review", "systems", "family_history",
                "social_history", "allergy", "past_medical_history",
                "备注", "说明", "文本", "ocr_text", "preprocessed_text"
            ]
        }

        for modality, keywords in modality_keywords.items():
            for keyword in keywords:
                if keyword in col_name:
                    return modality

        return None

    def _infer_structured_table_modality(self, col_name: str) -> Optional[str]:
        """给中文结构化表前缀做专门映射。"""
        if col_name.startswith("table_一般资料"):
            if any(token in col_name for token in ["出生年月", "年龄", "日期"]):
                return "Time"
            return "Demographics"

        if col_name.startswith("table_症状学"):
            return "Diagnosis"

        if col_name.startswith("table_床旁查体"):
            return "Procedures"

        if col_name.startswith("table_初步诊断") or col_name.startswith("table_第一次复诊"):
            return "Diagnosis"

        if col_name.startswith("table_辅助检查"):
            if any(token in col_name for token in ["查体", "hit试验", "dix-hallpike", "roll-test", "位置试验"]):
                return "Procedures"
            if any(token in col_name for token in ["诊断日期", "持续时间"]):
                return "Time"
            return "Labs"

        return None
