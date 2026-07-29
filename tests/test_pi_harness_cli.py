from __future__ import annotations

from agent.pi_harness_cli import parse_args


def test_harness_cli_parses_public_and_private_inputs() -> None:
    args = parse_args(
        [
            "--experiment-dir", "/tmp/experiment",
            "--train-raw", "/data/train/raw",
            "--train-reference", "/data/train/reference",
            "--validation-raw", "/data/validation/raw",
            "--validation-gold", "/private/validation/gold",
            "--evaluation-manifest", "/manifests/mimic.json",
            "--prompt-file", "/prompts/task.txt",
            "--skills-dir", "/skills/one",
            "--skills-dir", "/skills/two",
        ],
    )

    assert args.max_rounds == 20
    assert args.patience == 3
    assert args.target_score == 1.0
    assert args.max_iters == 10_000
    assert args.skills_dir == ["/skills/one", "/skills/two"]
