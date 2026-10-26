"""Replay entry point of the N18 cohort cleaning pipeline.

    python3 run_pipeline.py --raw-root RAW --train-reference TRAIN_REF \
                            --output-dir OUT [--workdir WORK] [--submission SUB]

The replay only reads {raw_root}, {train_reference} and writes {output_dir}
(and the optional {workdir}); it never touches the training raw data, the test
gold or any evaluation metadata.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import n18_clean  # noqa: E402

SUBMISSION_TEMPLATE = {
    "schema_version": 1,
    "result_root": "output",
    "replay": {
        "argv": [
            "--raw-root", "{raw_root}",
            "--train-reference", "{train_reference}",
            "--output-dir", "{output_dir}",
            "--workdir", "{workdir}",
        ],
        "command": "python3 pipeline/run_pipeline.py",
        "dependencies": ["{raw_root}", "{train_reference}", "{output_dir}", "{workdir}"],
    },
}


def main(argv=None):
    ap = argparse.ArgumentParser(description="N18 MIMIC-IV cohort pipeline (replay)")
    ap.add_argument("--raw-root", required=True, help="raw slice root (hosp/ icu/ ed/ note/)")
    ap.add_argument("--train-reference", required=True, help="train reference package")
    ap.add_argument("--output-dir", required=True, help="where the 18 csv files are written")
    ap.add_argument("--workdir", default=None, help="optional scratch directory")
    ap.add_argument("--submission", default=None, help="optional path of a submission.json")
    args = ap.parse_args(argv)

    report = n18_clean.run(args.raw_root, args.train_reference, args.output_dir, args.workdir)

    if args.submission:
        submission = json.loads(json.dumps(SUBMISSION_TEMPLATE))
        submission["result_root"] = os.path.relpath(os.path.abspath(args.output_dir),
                                                    os.path.dirname(os.path.abspath(args.submission)))
        with open(args.submission, "w") as fh:
            json.dump(submission, fh, indent=1)
        print("wrote", args.submission)

    print(json.dumps(report["steps"], indent=1))


if __name__ == "__main__":
    main()
