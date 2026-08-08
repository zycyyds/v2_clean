from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import numpy as np

_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _hash_bucket(value: str, buckets: int = 48) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % buckets


def node_feature(node: dict[str, Any], *, degree: int, frequency: int) -> np.ndarray:
    """Create a stable 64-dim feature without an ID lookup table."""
    result = np.zeros(64, dtype=np.float32)
    node_type = str(node["node_type"])
    result[0] = 1.0 if node_type == "Row" else 0.0
    result[1] = 1.0 if node_type == "Value" else 0.0
    result[2] = math.log1p(degree)
    result[3] = math.log1p(frequency)
    if node_type == "Value":
        domain = str(node.get("domain", ""))
        domain_index = {
            "entity.subject_id": 0,
            "entity.hadm_id": 1,
            "entity.stay_id": 2,
            "transaction": 3,
            "dictionary": 4,
        }
        result[4 + domain_index.get(domain.split(".", 1)[0], 5)] = 1.0
        result[12 + _hash_bucket(domain, 32)] = 1.0
    else:
        result[4 + _hash_bucket(str(node.get("table", "")), 16)] = 1.0
        result[20 + _hash_bucket(str(node.get("table", "")), 32)] = 1.0
    return result


def canonical_scalar(value: str, canonicalizer: str, *, icd_version: str = "") -> str | None:
    value = value.strip()
    if not value or value.lower() in {"nan", "none", "null", "nat"}:
        return None
    if canonicalizer == "icd_version_code":
        version = icd_version.strip() or "unknown"
        return f"{version}:{value}"
    if canonicalizer == "string_id":
        return value
    raise ValueError(f"unsupported canonicalizer: {canonicalizer}")
