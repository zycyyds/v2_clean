from __future__ import annotations

import argparse
from pathlib import Path

from .cell_training import TrainingConfig, run_training


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train typed-value Cell error classifiers with frozen Qwen embeddings."
    )
    parser.add_argument("--graph-dir", required=True, type=Path)
    parser.add_argument("--supervision-dir", required=True, type=Path)
    parser.add_argument("--embedding-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--model-type",
        required=True,
        choices=("triple_mlp", "strict_mlp", "fullrow_rgcn", "strict_rgcn"),
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults to 256 for either MLP and 64 for either R-GCN.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--rgcn-layers", type=int, default=2)
    parser.add_argument("--rgcn-bases", type=int, default=16)
    parser.add_argument("--fanouts", default="16,8")
    parser.add_argument(
        "--evaluate-internal-test",
        action="store_true",
        help="Evaluate fold 4 after the model configuration has been frozen.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    fanouts = tuple(int(value.strip()) for value in args.fanouts.split(",") if value.strip())
    config = TrainingConfig(
        graph_dir=str(args.graph_dir),
        supervision_dir=str(args.supervision_dir),
        embedding_dir=str(args.embedding_dir),
        output_dir=str(args.output_dir),
        model_type=args.model_type,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        batch_size=(
            args.batch_size
            if args.batch_size is not None
            else (256 if args.model_type.endswith("_mlp") else 64)
        ),
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        rgcn_layers=args.rgcn_layers,
        rgcn_bases=args.rgcn_bases,
        fanouts=fanouts,
        evaluate_internal_test=args.evaluate_internal_test,
        resume=args.resume,
    )
    report = run_training(config)
    if report["internal_test_evaluated"]:
        test = report["splits"]["internal_test"]["metrics"]
        suffix = (
            f"test_average_precision={test['average_precision']} "
            f"test_macro_f1={test['macro_f1']}"
        )
    else:
        suffix = "internal_test=WITHHELD"
    print(
        f"SUCCESS model={report['model_type']} seed={report['seed']} "
        f"best_epoch={report['best_epoch']} {suffix}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
