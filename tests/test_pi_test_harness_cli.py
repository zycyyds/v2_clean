from agent.pi_test_harness_cli import parse_args


def test_test_harness_cli_defaults_to_sibling_experiment() -> None:
    args = parse_args(
        [
            "--validation-experiment", "/runs/validation",
            "--test-raw", "/data/test/raw",
            "--test-gold", "/private/test/reference_private",
            "--evaluation-manifest", "/manifests/mimic.json",
        ]
    )

    assert args.test_experiment is None
    assert args.max_declaration_repairs == 3
