from __future__ import annotations

import argparse
import json
from pathlib import Path

from .candidate_ranking import (
    PrepareRankingConfig,
    RankTrainingConfig,
    export_train_validation_gold,
    prepare_ranking_data,
    train_complex_ranker,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and train a frozen Strict R-GCN + ComplEx candidate ranker."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scope = subparsers.add_parser(
        "scope-gold",
        help="Export a host-private folds 0-3 paired log; fold 4 is never exported.",
    )
    scope.add_argument("--graph-dir", required=True, type=Path)
    scope.add_argument("--supervision-dir", required=True, type=Path)
    scope.add_argument("--paired-log", required=True, type=Path)
    scope.add_argument("--predictions", required=True, type=Path)
    scope.add_argument("--detector-report", required=True, type=Path)
    scope.add_argument("--output-dir", required=True, type=Path)

    prepare = subparsers.add_parser(
        "prepare",
        help=(
            "Freeze Strict R-GCN contexts and Qwen candidate values into private "
            "Train/Validation ranking artifacts."
        ),
    )
    prepare.add_argument("--graph-dir", required=True, type=Path)
    prepare.add_argument("--supervision-dir", required=True, type=Path)
    prepare.add_argument("--embedding-dir", required=True, type=Path)
    prepare.add_argument("--detector-checkpoint", required=True, type=Path)
    prepare.add_argument("--predictions", required=True, type=Path)
    prepare.add_argument("--train-candidates", required=True, type=Path)
    prepare.add_argument("--train-candidate-manifest", required=True, type=Path)
    prepare.add_argument("--validation-candidates", required=True, type=Path)
    prepare.add_argument("--validation-candidate-manifest", required=True, type=Path)
    prepare.add_argument("--paired-log", required=True, type=Path)
    prepare.add_argument("--paired-log-manifest", required=True, type=Path)
    prepare.add_argument("--qwen-model", required=True)
    prepare.add_argument("--qwen-revision")
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    prepare.add_argument("--qwen-batch-size", type=int, default=32)
    prepare.add_argument("--rgcn-batch-size", type=int, default=64)

    train = subparsers.add_parser(
        "train",
        help="Train only the ComplEx projection/scoring head on prepared representations.",
    )
    train.add_argument("--prepared-dir", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--seed", required=True, type=int)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--rank-dim", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--patience", type=int, default=10)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "scope-gold":
        report = export_train_validation_gold(
            graph_dir=args.graph_dir,
            supervision_dir=args.supervision_dir,
            paired_log=args.paired_log,
            predictions=args.predictions,
            detector_report=args.detector_report,
            output_dir=args.output_dir,
        )
    elif args.command == "prepare":
        report = prepare_ranking_data(PrepareRankingConfig(
            graph_dir=str(args.graph_dir),
            supervision_dir=str(args.supervision_dir),
            embedding_dir=str(args.embedding_dir),
            detector_checkpoint=str(args.detector_checkpoint),
            predictions=str(args.predictions),
            train_candidates=str(args.train_candidates),
            train_candidate_manifest=str(args.train_candidate_manifest),
            validation_candidates=str(args.validation_candidates),
            validation_candidate_manifest=str(args.validation_candidate_manifest),
            paired_log=str(args.paired_log),
            paired_log_manifest=str(args.paired_log_manifest),
            qwen_model=args.qwen_model,
            qwen_revision=args.qwen_revision,
            output_dir=str(args.output_dir),
            device=args.device,
            qwen_batch_size=args.qwen_batch_size,
            rgcn_batch_size=args.rgcn_batch_size,
        ))
    else:
        report = train_complex_ranker(RankTrainingConfig(
            prepared_dir=str(args.prepared_dir),
            output_dir=str(args.output_dir),
            seed=args.seed,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            rank_dim=args.rank_dim,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            patience=args.patience,
        ))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
