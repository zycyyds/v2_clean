from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from agent.pi_fcorr_synthesis import synthesize_fcorr_with_pi
from agent.pi_train_repair import default_repair_skill_dirs

from .cell_repair import (
    CellRepairError,
    apply_repair_plan,
    build_field_pairs,
    build_repair_targets,
    evaluate_candidates,
    evaluate_repairs,
    freeze_rule_registry,
    recover_raw_from_graph,
    run_frozen_rules,
    synthesize_fcorr,
)


def _print_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build Train-only field evidence, synthesize multi-candidate F_corr functions, "
            "and execute frozen candidate audits."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    recover = subparsers.add_parser(
        "recover-raw",
        help="Recover a complete logical dirty raw copy from graph Cell observations.",
    )
    recover.add_argument("--graph-dir", required=True, type=Path)
    recover.add_argument("--output-dir", required=True, type=Path)

    pairs = subparsers.add_parser(
        "build-pairs",
        help="Build complete folds 0/1/2 dirty-clean pairs grouped by table.column.",
    )
    pairs.add_argument("--graph-dir", required=True, type=Path)
    pairs.add_argument("--supervision-dir", required=True, type=Path)
    pairs.add_argument(
        "--raw-dir",
        type=Path,
        help="Optional raw directory for an additional raw/graph value cross-check.",
    )
    pairs.add_argument("--paired-log", required=True, type=Path)
    pairs.add_argument("--output-dir", required=True, type=Path)

    synthesize = subparsers.add_parser(
        "synthesize-fcorr",
        help=(
            "Use MiniMax M3 to synthesize and Train-validate one multi-candidate "
            "F_corr per field."
        ),
    )
    synthesize.add_argument("--evidence-dir", required=True, type=Path)
    synthesize.add_argument("--output-dir", required=True, type=Path)
    synthesize.add_argument("--agent-key", default="react_planner")
    synthesize.add_argument("--max-attempts", type=int, default=12)
    synthesize.add_argument(
        "--field",
        action="append",
        default=[],
        help="Limit synthesis to one table.column; repeat for multiple fields.",
    )

    synthesize_pi = subparsers.add_parser(
        "synthesize-fcorr-pi",
        help=(
            "Use one persistent Train-only Pi Agent session and four correction skills "
            "to synthesize frozen multi-candidate F_corr rules."
        ),
    )
    synthesize_pi.add_argument("--project-root", default=str(Path(__file__).parents[1]))
    synthesize_pi.add_argument("--evidence-dir", required=True, type=Path)
    synthesize_pi.add_argument("--output-dir", required=True, type=Path)
    synthesize_pi.add_argument("--agent-key", default="react_planner")
    synthesize_pi.add_argument("--max-rounds", type=int, default=8)
    synthesize_pi.add_argument("--max-iters", type=int, default=10_000)
    synthesize_pi.add_argument(
        "--field",
        action="append",
        default=[],
        help="Limit synthesis to one table.column; repeat for multiple fields.",
    )
    synthesize_pi.add_argument(
        "--skill-dir",
        action="append",
        default=[],
        help="Override the four default correction skill directories.",
    )

    freeze = subparsers.add_parser(
        "freeze-registry",
        help="Rebuild the frozen registry from accepted field manifests.",
    )
    freeze.add_argument("--synthesis-dir", required=True, type=Path)

    targets = subparsers.add_parser(
        "build-targets",
        help="Select Strict R-GCN detector-positive Cells for one split.",
    )
    targets.add_argument("--predictions", required=True, type=Path)
    targets.add_argument("--graph-dir", required=True, type=Path)
    targets.add_argument(
        "--raw-dir",
        type=Path,
        help="Optional raw directory for an additional raw/graph value cross-check.",
    )
    targets.add_argument("--split", required=True)
    targets.add_argument("--output-dir", required=True, type=Path)

    run = subparsers.add_parser(
        "run-rules",
        help="Run frozen field F_corr functions to emit up to five candidates per target.",
    )
    run.add_argument("--targets", required=True, type=Path)
    run.add_argument("--rule-dir", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)

    apply = subparsers.add_parser(
        "apply",
        help=(
            "Apply a later selector's declared replacements; candidate generation "
            "does not emit them."
        ),
    )
    apply.add_argument("--raw-dir", required=True, type=Path)
    apply.add_argument("--repair-plan", required=True, type=Path)
    apply.add_argument("--output-dir", required=True, type=Path)

    evaluate = subparsers.add_parser(
        "evaluate",
        help=(
            "Evaluate a later selector's final repair plan; use evaluate-candidates "
            "in this stage."
        ),
    )
    evaluate.add_argument("--repair-values", required=True, type=Path)
    evaluate.add_argument("--repair-plan", required=True, type=Path)
    evaluate.add_argument("--injection-log", required=True, type=Path)
    evaluate.add_argument("--predictions", required=True, type=Path)
    evaluate.add_argument("--graph-dir", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)

    evaluate_candidates_parser = subparsers.add_parser(
        "evaluate-candidates",
        help="Run a host-private frozen candidate Recall@1/3/5 audit for one split.",
    )
    evaluate_candidates_parser.add_argument("--candidate-values", required=True, type=Path)
    evaluate_candidates_parser.add_argument("--injection-log", required=True, type=Path)
    evaluate_candidates_parser.add_argument("--predictions", required=True, type=Path)
    evaluate_candidates_parser.add_argument("--graph-dir", required=True, type=Path)
    evaluate_candidates_parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "recover-raw":
            report = recover_raw_from_graph(
                graph_dir=args.graph_dir,
                output_dir=args.output_dir,
            )
        elif args.command == "build-pairs":
            report = build_field_pairs(
                graph_dir=args.graph_dir,
                supervision_dir=args.supervision_dir,
                raw_dir=args.raw_dir,
                paired_log=args.paired_log,
                output_dir=args.output_dir,
            )
        elif args.command == "synthesize-fcorr":
            report = synthesize_fcorr(
                evidence_dir=args.evidence_dir,
                output_dir=args.output_dir,
                agent_key=args.agent_key,
                max_attempts=args.max_attempts,
                fields=args.field,
            )
        elif args.command == "synthesize-fcorr-pi":
            project = Path(args.project_root).expanduser().resolve()
            skills = (
                tuple(Path(value) for value in args.skill_dir)
                if args.skill_dir
                else default_repair_skill_dirs(project)
            )
            result = asyncio.run(synthesize_fcorr_with_pi(
                project_root=project,
                evidence_dir=args.evidence_dir,
                output_dir=args.output_dir,
                agent_key=args.agent_key,
                max_rounds=args.max_rounds,
                max_iters=args.max_iters,
                fields=tuple(args.field),
                skill_dirs=skills,
            ))
            report = {
                "status": result.status,
                "workflow": "pi_agent_train_only_fcorr_synthesis",
                "rounds": result.rounds,
                "output_dir": str(result.output_dir),
                "field_count": result.field_count,
                "rule_count": result.rule_count,
                "registry": result.registry,
            }
        elif args.command == "freeze-registry":
            report = freeze_rule_registry(synthesis_dir=args.synthesis_dir)
        elif args.command == "build-targets":
            report = build_repair_targets(
                predictions=args.predictions,
                graph_dir=args.graph_dir,
                raw_dir=args.raw_dir,
                split=args.split,
                output_dir=args.output_dir,
            )
        elif args.command == "run-rules":
            report = run_frozen_rules(
                targets=args.targets,
                rule_dir=args.rule_dir,
                output_dir=args.output_dir,
            )
        elif args.command == "apply":
            report = apply_repair_plan(
                raw_dir=args.raw_dir,
                repair_plan=args.repair_plan,
                output_dir=args.output_dir,
            )
        elif args.command == "evaluate":
            report = evaluate_repairs(
                repair_values=args.repair_values,
                repair_plan=args.repair_plan,
                injection_log=args.injection_log,
                predictions=args.predictions,
                graph_dir=args.graph_dir,
                output_dir=args.output_dir,
            )
        else:
            report = evaluate_candidates(
                candidate_values=args.candidate_values,
                injection_log=args.injection_log,
                predictions=args.predictions,
                graph_dir=args.graph_dir,
                output_dir=args.output_dir,
            )
    except CellRepairError as exc:
        raise SystemExit(f"FAILED: {exc}") from exc
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
