"""Restricted filesystem and adapter tools for Data Cleaning Agent."""

from typing import TYPE_CHECKING, Any

from .context import EngineerToolContext

if TYPE_CHECKING:
    from .reference_variants import ReferenceVariantTools
    from .tools import EngineerTools

__all__ = [
    "EngineerToolContext",
    "EngineerTools",
    "ReferenceVariantTools",
    "register_engineer_atomic_tools",
    "register_reference_variant_tools",
]


def __getattr__(name: str) -> Any:
    if name in {"ReferenceVariantTools", "register_reference_variant_tools"}:
        from .reference_variants import ReferenceVariantTools, register_reference_variant_tools

        return {
            "ReferenceVariantTools": ReferenceVariantTools,
            "register_reference_variant_tools": register_reference_variant_tools,
        }[name]
    if name in {"EngineerTools", "register_engineer_atomic_tools"}:
        from .tools import EngineerTools, register_engineer_atomic_tools

        return {
            "EngineerTools": EngineerTools,
            "register_engineer_atomic_tools": register_engineer_atomic_tools,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
