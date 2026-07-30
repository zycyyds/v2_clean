from agent.pi_test_harness_cli import parse_args
from agent.pi_test_preflight_cli import parse_args as parse_preflight_args


def test_preflight_cli_parses_strict_public_inputs() -> None:
    args = parse_preflight_args(
        [
            "--validation-experiment", "/runs/validation",
            "--preflight-experiment", "/runs/preflight",
            "--preflight-raw", "/data/validation/raw",
        ],
    )

    assert args.replay_timeout == 1800.0
    assert not hasattr(args, "max_iters")


def test_test_cli_requires_attestation_and_has_no_agent_controls() -> None:
    args = parse_args(
        [
            "--validation-experiment", "/runs/validation",
            "--preflight-attestation", "/runs/preflight/host/preflight_attestation.json",
            "--test-experiment", "/runs/test",
            "--test-raw", "/data/test/raw",
            "--test-gold", "/data/test/reference_private",
            "--evaluation-manifest", "/manifests/evaluation.json",
        ],
    )

    assert args.replay_timeout == 1800.0
    assert args.scoring_timeout == 3600.0
    assert not hasattr(args, "max_declaration_repairs")
    assert not hasattr(args, "max_iters")
