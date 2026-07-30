from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path


TEST_GOLD_OVERLAP_ERROR = "unsafe Test Gold path overlap"


def sandbox_visible_recursive_roots(
    *application_roots: Path,
    include_shell_state: bool = False,
) -> list[Path]:
    """Return the recursive roots shared by worker and replay profiles."""

    roots = [
        *application_roots,
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path("/System"),
        Path("/usr"),
        Path("/bin"),
        Path("/sbin"),
        Path("/private/etc"),
    ]
    if include_shell_state:
        roots.append(Path("/private/var/select"))
    return _unique_paths(roots)


def require_test_gold_isolated(
    test_gold: Path,
    recursive_roots: Iterable[Path],
    *,
    literal_paths: Iterable[Path] = (),
) -> None:
    """Reject recursive overlap, while treating literal paths as exact-only."""

    gold = test_gold.expanduser().resolve(strict=False)
    for recursive_root in recursive_roots:
        public = recursive_root.expanduser().resolve(strict=False)
        if gold == public or gold in public.parents or public in gold.parents:
            raise ValueError(TEST_GOLD_OVERLAP_ERROR)
    for literal_path in literal_paths:
        if gold == literal_path.expanduser().resolve(strict=False):
            raise ValueError(TEST_GOLD_OVERLAP_ERROR)


def build_macos_sandbox_profile(
    *,
    executable: Path,
    read_roots: list[Path],
    write_roots: list[Path],
    allow_network: bool,
    traversal_roots: list[Path] | None = None,
    literal_read_paths: list[Path] | None = None,
) -> str:
    """Build a deny-by-default profile inherited by the Agent process tree."""

    def literal(path: Path) -> str:
        return str(path.expanduser().resolve()).replace("\\", "\\\\").replace('"', '\\"')

    readable = _unique_paths([*read_roots, *write_roots])
    writable = _unique_paths(write_roots)
    traversable = _unique_paths(traversal_roots or [])
    literal_readable = _unique_paths(literal_read_paths or [])
    lines = [
        "(version 1)",
        "(deny default)",
        '(import "system.sb")',
        "(allow process*)",
        "(allow signal (target same-sandbox))",
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
        for root in [*traversable, *literal_readable]
    )
    lines.extend(
        f'(allow file-read-metadata file-test-existence (path-ancestors "{literal(root)}"))'
        for root in [*readable, *literal_readable]
    )
    lines.extend(
        f'(allow file-write* (literal "{literal(root)}") (subpath "{literal(root)}"))'
        for root in writable
    )
    return "\n".join(lines) + "\n"


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique: dict[str, Path] = {}
    for path in paths:
        resolved = path.expanduser().resolve(strict=False)
        unique[str(resolved)] = resolved
    return [unique[key] for key in sorted(unique)]
