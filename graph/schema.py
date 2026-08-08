from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json

NODE_TYPES = ("Row", "Value")


@dataclass(frozen=True)
class SharedField:
    """A column whose values are allowed to be shared across rows/tables."""

    domain: str
    columns: tuple[str, ...]
    tables: tuple[str, ...] = ()
    canonicalizer: str = "string_id"

    def allows(self, table_id: str, column: str) -> bool:
        return column in self.columns and (not self.tables or table_id in self.tables)


@dataclass(frozen=True)
class GraphSchema:
    """Configuration for the Row + typed Value graph.

    ``shared_fields`` is deliberately an allow-list. A column named ``itemid``
    must not become a global node unless its table/domain is declared here.
    """

    shared_fields: tuple[SharedField, ...]
    node_types: tuple[str, ...] = NODE_TYPES
    version: int = 1

    def field_for(self, table_id: str, column: str) -> SharedField | None:
        for field_spec in self.shared_fields:
            if field_spec.allows(table_id, column):
                return field_spec
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.version,
            "node_types": list(self.node_types),
            "shared_fields": [
                {
                    "domain": item.domain,
                    "columns": list(item.columns),
                    "tables": list(item.tables),
                    "canonicalizer": item.canonicalizer,
                }
                for item in self.shared_fields
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GraphSchema":
        fields = tuple(
            SharedField(
                domain=str(item["domain"]),
                columns=tuple(str(value) for value in item.get("columns", [])),
                tables=tuple(str(value) for value in item.get("tables", [])),
                canonicalizer=str(item.get("canonicalizer", "string_id")),
            )
            for item in payload.get("shared_fields", [])
        )
        if not fields:
            raise ValueError("schema must declare at least one shared field")
        node_types = tuple(str(item) for item in payload.get("node_types", NODE_TYPES))
        if node_types != NODE_TYPES:
            raise ValueError(f"schema node_types must be exactly {NODE_TYPES}")
        return cls(shared_fields=fields, node_types=node_types, version=int(payload.get("schema_version", 1)))

    @classmethod
    def load(cls, path: str | Path) -> "GraphSchema":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _tables(*names: str) -> tuple[str, ...]:
    return tuple(names)


DEFAULT_SCHEMA = GraphSchema(
    shared_fields=(
        SharedField("entity.subject_id", ("subject_id",)),
        SharedField("entity.hadm_id", ("hadm_id",)),
        SharedField("entity.stay_id", ("stay_id",)),
        SharedField(
            "transaction.poe_id",
            ("poe_id",),
            _tables("hosp/emar", "hosp/pharmacy", "hosp/poe", "hosp/poe_detail", "hosp/prescriptions"),
        ),
        SharedField(
            "transaction.pharmacy_id",
            ("pharmacy_id",),
            _tables("hosp/pharmacy", "hosp/emar", "hosp/emar_detail"),
        ),
        SharedField(
            "transaction.emar_id",
            ("emar_id",),
            _tables("hosp/emar", "hosp/emar_detail"),
        ),
        SharedField(
            "transaction.icu_order_id",
            ("orderid", "icu_order_id"),
            _tables("icu/ingredientevents", "icu/inputevents", "icu/procedureevents"),
        ),
        SharedField(
            "transaction.icu_linkorder_id",
            ("linkorderid", "icu_linkorder_id"),
            _tables("icu/ingredientevents", "icu/inputevents", "icu/procedureevents"),
        ),
        SharedField(
            "dictionary.icu_item",
            ("itemid",),
            _tables("icu/chartevents", "icu/datetimeevents", "icu/d_items", "icu/ingredientevents", "icu/inputevents", "icu/outputevents", "icu/procedureevents"),
        ),
        SharedField("dictionary.lab_item", ("itemid",), _tables("hosp/labevents", "hosp/d_labitems")),
        SharedField(
            "dictionary.diagnosis_icd",
            ("icd_code",),
            _tables("hosp/diagnoses_icd", "hosp/d_icd_diagnoses"),
            "icd_version_code",
        ),
        SharedField(
            "dictionary.procedure_icd",
            ("icd_code",),
            _tables("hosp/procedures_icd", "hosp/d_icd_procedures"),
            "icd_version_code",
        ),
        SharedField("dictionary.hcpcs", ("hcpcs_cd", "code", "hcpcs"), _tables("hosp/hcpcsevents", "hosp/d_hcpcs")),
    )
)
