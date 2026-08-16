from __future__ import annotations

from agent.pi_repair_audit_cli import parse_args


def test_cli_exposes_build_train_test_and_full_contracts() -> None:
    build = parse_args(
        [
            "build-view",
            "--dirty-raw",
            "dirty",
            "--clean-raw",
            "clean",
            "--graph-dir",
            "graph",
            "--supervision-dir",
            "supervision",
            "--paired-log",
            "pairs.csv",
            "--output-dir",
            "public",
        ]
    )
    assert build.command == "build-view"
    assert build.expected_dirty == 20_000
    assert build.expected_clean == 10_000

    train = parse_args(
        [
            "train",
            "--experiment-dir",
            "train-exp",
            "--public-train-root",
            "public",
            "--train-gold-log",
            "gold.csv",
            "--train-row-gold-log",
            "rows.csv",
        ]
    )
    assert train.command == "train"
    assert train.max_repair_turns == 2

    test = parse_args(
        [
            "test",
            "--train-experiment",
            "train-exp",
            "--test-experiment",
            "test-exp",
            "--test-raw",
            "raw",
            "--test-gold-log",
            "gold.csv",
        ]
    )
    assert test.command == "test"

    full = parse_args(
        [
            "full",
            "--experiment-dir",
            "train-exp",
            "--public-train-root",
            "public",
            "--train-gold-log",
            "train-gold.csv",
            "--train-row-gold-log",
            "train-row-gold.csv",
            "--test-experiment",
            "test-exp",
            "--test-raw",
            "test-raw",
            "--test-gold-log",
            "test-gold.csv",
        ]
    )
    assert full.command == "full"
