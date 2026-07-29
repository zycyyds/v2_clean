from __future__ import annotations

from pathlib import Path


def build_macos_sandbox_profile(
    *,
    executable: Path,
    read_roots: list[Path],
    write_roots: list[Path],
    allow_network: bool,
    traversal_roots: list[Path] | None = None,
) -> str:
    """Build a deny-by-default profile inherited by the Agent process tree."""

    def literal(path: Path) -> str:
        return str(path.expanduser().resolve()).replace("\\", "\\\\").replace('"', '\\"')

    readable = _unique_paths([*read_roots, *write_roots])
    writable = _unique_paths(write_roots)
    traversable = _unique_paths(traversal_roots or [])
    lines = [
        "(version 1)",
        "(deny default)",
        '(import "system.sb")',
        "(allow process*)",
        "(allow network-outbound)" if allow_network else "(deny network*)",
        '(allow file-read-metadata (literal "/"))',
        f'(allow process-exec (literal "{literal(executable)}"))',
    ]
    lines.extend(
        f'(allow file-read* (literal "{literal(root)}") (subpath "{literal(root)}"))'
        for root in readable
    )
    lines.extend(
        f'(allow file-read* (literal "{literal(root)}"))'
        for root in traversable
    )
    lines.extend(
        f'(allow file-read-metadata file-test-existence (path-ancestors "{literal(root)}"))'
        for root in readable
    )
    lines.extend(
        f'(allow file-write* (literal "{literal(root)}") (subpath "{literal(root)}"))'
        for root in writable
    )
    return "\n".join(lines) + "\n"


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: dict[str, Path] = {}
    for path in paths:
        resolved = path.expanduser().resolve()
        unique[str(resolved)] = resolved
    return [unique[key] for key in sorted(unique)]
