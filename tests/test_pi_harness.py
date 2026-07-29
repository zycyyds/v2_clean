from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

from agent.pi_harness import (
    HarnessProgress,
    Submission,
    _RedactingTee,
    directory_sha256,
    load_submission,
    record_round_score,
    render_replay_argv,
    restore_agent_workdir,
    snapshot_agent_workdir,
    validate_plain_directory_tree,
)


def _write_submission(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_worker_output_redacts_api_keys_from_all_logs() -> None:
    terminal = StringIO()
    worker_log = StringIO()
    secret = "sk-private-test-key"
    stream = _RedactingTee((terminal, worker_log), (secret,))

    stream.write(f"request failed for {secret}\n")
    stream.flush()

    for output in (terminal.getvalue(), worker_log.getvalue()):
        assert secret not in output
        assert "<redacted-api-key>" in output


def test_submission_accepts_dynamic_paths_and_renders_placeholders(tmp_path: Path) -> None:
    workdir = tmp_path / "agent_workdir"
    workdir.mkdir()
    _write_submission(
        workdir / "submission.json",
        {
            "schema_version": 1,
            "result_root": "outputs/package",
            "replay": {
                "argv": [
                    "python",
                    "scripts/build.py",
                    "--raw",
                    "{raw_root}",
                    "--train-reference",
                    "{train_reference}",
                    "--out",
                    "{output_dir}",
                ],
            },
        },
    )

    submission = load_submission(workdir)

    assert submission.result_root == workdir / "outputs/package"
    assert render_replay_argv(
        submission,
        raw_root=Path("/public/validation/raw"),
        train_reference=Path("/public/train/reference"),
        output_dir=Path("/tmp/replay-output"),
        workdir=Path("/tmp/replay-workdir"),
    ) == [
        "python",
        "scripts/build.py",
        "--raw",
        "/public/validation/raw",
        "--train-reference",
        "/public/train/reference",
        "--out",
        str(Path("/tmp/replay-output").resolve()),
    ]


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"schema_version": 1, "result_root": "../escape", "replay": {"argv": ["x"]}}, "result_root"),
        (
            {
                "schema_version": 1,
                "result_root": ".",
                "replay": {
                    "argv": ["python", "x.py", "--raw", "{raw_root}", "--out", "{output_dir}"],
                },
            },
            "dedicated subdirectory",
        ),
        ({"schema_version": 1, "result_root": "result", "replay": {"argv": "python x.py"}}, "argv"),
        (
            {
                "schema_version": 1,
                "result_root": "result",
                "replay": {"argv": ["python", "x.py", "--out", "{unknown}"]},
            },
            "unsupported placeholder",
        ),
        (
            {
                "schema_version": 1,
                "result_root": "result",
                "replay": {"argv": ["python", "x.py", "--out", "fixed"]},
            },
            "output_dir",
        ),
        (
            {
                "schema_version": 1,
                "result_root": "result",
                "replay": {
                    "argv": [
                        "python",
                        "/private/host/script.py",
                        "--raw",
                        "{raw_root}",
                        "--out",
                        "{output_dir}",
                    ],
                },
            },
            "static absolute paths",
        ),
        (
            {
                "schema_version": 1,
                "result_root": "result",
                "replay": {
                    "argv": [
                        "sh",
                        "-c",
                        "python x.py --raw {raw_root} --out {output_dir}",
                    ],
                },
            },
            "shell command strings",
        ),
    ],
)
def test_submission_rejects_unsafe_or_non_replayable_contract(
    tmp_path: Path,
    payload: dict,
    message: str,
) -> None:
    workdir = tmp_path / "agent_workdir"
    workdir.mkdir()
    _write_submission(workdir / "submission.json", payload)

    with pytest.raises(ValueError, match=message):
        load_submission(workdir)


def test_round_state_promotes_strictly_and_stops_after_three_non_improvements() -> None:
    state = HarnessProgress()
    state, first = record_round_score(state, 0.4, max_rounds=20, patience=3, target_score=1.0)
    assert first.promoted
    assert state.best_score == 0.4
    assert state.best_round == 1
    assert state.consecutive_non_improvements == 0

    state, second = record_round_score(state, 0.4, max_rounds=20, patience=3, target_score=1.0)
    state, third = record_round_score(state, 0.399999, max_rounds=20, patience=3, target_score=1.0)
    state, fourth = record_round_score(state, 0.4000005, max_rounds=20, patience=3, target_score=1.0)

    assert not second.promoted and not third.promoted and not fourth.promoted
    assert state.stop_reason == "patience"
    assert state.round_index == 4


def test_round_state_stops_on_target_or_max_rounds() -> None:
    state, decision = record_round_score(
        HarnessProgress(),
        1.0,
        max_rounds=20,
        patience=3,
        target_score=1.0,
    )
    assert decision.promoted
    assert state.stop_reason == "target_score"

    state = HarnessProgress(round_index=19, best_score=0.5, best_round=1)
    state, decision = record_round_score(
        state,
        0.6,
        max_rounds=20,
        patience=3,
        target_score=1.0,
    )
    assert decision.promoted
    assert state.stop_reason == "max_rounds"


def test_exact_one_micro_point_gain_promotes_and_invalid_attempt_does_not() -> None:
    state = HarnessProgress(round_index=1, best_score=0.4, best_round=1)
    state, decision = record_round_score(
        state,
        0.400001,
        max_rounds=20,
        patience=3,
        target_score=1.0,
    )
    assert decision.promoted
    assert state.best_score == 0.400001

    state = HarnessProgress()
    state, decision = record_round_score(
        state,
        0.0,
        max_rounds=20,
        patience=3,
        target_score=1.0,
        eligible_for_promotion=False,
    )
    assert not decision.promoted
    assert state.best_score == -1.0


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_round_state_rejects_non_finite_or_out_of_range_scores(score: float) -> None:
    with pytest.raises(ValueError, match="score must be finite"):
        record_round_score(
            HarnessProgress(),
            score,
            max_rounds=20,
            patience=3,
            target_score=1.0,
        )


def test_snapshot_restore_preserves_agent_runs_but_reverts_outputs(tmp_path: Path) -> None:
    workdir = tmp_path / "agent_workdir"
    snapshot = tmp_path / "host/best"
    (workdir / "scripts").mkdir(parents=True)
    (workdir / "result_package").mkdir()
    (workdir / ".agent_runs/run-1").mkdir(parents=True)
    (workdir / "scripts/build.py").write_text("best", encoding="utf-8")
    (workdir / "result_package/data.csv").write_text("best", encoding="utf-8")
    (workdir / ".agent_runs/run-1/log").write_text("context", encoding="utf-8")

    snapshot_agent_workdir(workdir, snapshot)
    best_digest = directory_sha256(snapshot)
    (workdir / "scripts/build.py").write_text("bad", encoding="utf-8")
    (workdir / "result_package/data.csv").write_text("bad", encoding="utf-8")
    (workdir / "temporary.txt").write_text("remove", encoding="utf-8")
    (workdir / ".agent_runs/run-1/log").write_text("context-continued", encoding="utf-8")

    restore_agent_workdir(workdir, snapshot)

    assert (workdir / "scripts/build.py").read_text() == "best"
    assert (workdir / "result_package/data.csv").read_text() == "best"
    assert not (workdir / "temporary.txt").exists()
    assert (workdir / ".agent_runs/run-1/log").read_text() == "context-continued"
    assert directory_sha256(workdir) == best_digest


def test_replay_tree_rejects_symbolic_links_and_special_files(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "file.txt").write_text("ok", encoding="utf-8")
    validate_plain_directory_tree(tree)

    (tree / "link").symlink_to(tree / "file.txt")
    with pytest.raises(ValueError, match="symbolic link"):
        validate_plain_directory_tree(tree)
