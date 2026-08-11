from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embedding import Qwen3EmbeddingBackend, build_embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate resumable Qwen3 embeddings for a typed-value graph.")
    parser.add_argument("--graph-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--revision")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--storage-dtype", choices=("float16",), default="float16")
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    backend = Qwen3EmbeddingBackend(
        args.model,
        device=args.device,
        max_length=args.max_length,
        revision=args.revision,
    )
    manifest = build_embeddings(
        args.graph_dir,
        args.output_dir,
        backend,
        batch_size=args.batch_size,
        max_length=args.max_length,
        storage_dtype=args.storage_dtype,
        max_nodes=args.max_nodes,
        resume=args.resume,
    )
    report = json.loads((args.output_dir / "benchmark_report.json").read_text(encoding="utf-8"))
    print(
        f"{manifest['status']} nodes={manifest['completed']['nodes']} "
        f"relations={manifest['completed']['relations']} "
        f"nodes_per_second={report['nodes_per_second']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
