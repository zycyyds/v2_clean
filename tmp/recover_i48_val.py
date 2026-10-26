"""I48 val harness 崩溃后的**只收尾**恢复脚本（不改 harness 源码、不重跑 LLM）。

事故经过
--------
跑到第 7 轮时 MiniMax 两个 API key 全部 429 限流
（`Pi worker AllApiKeysRateLimitedError: all 2 configured API keys failed`），
带容错的驱动重试 4 次全失败 → harness 抛异常 → 兜底收尾接手，
但**兜底收尾自己也没成功**：

    收尾（worker_crash_finalized）：独立重放 best_snapshot ...
      replay status=EXECUTION_FAILED score=0.000000
    !! 重放不可接受（0.000000 < 0.769646），不收尾

根因（不是 agent 的清洗逻辑问题）
--------------------------------
agent 写的 `submission.json` 把命令行拆成了两个键：

    "replay": {
      "argv": ["--raw-root", "{raw_root}", "--train-reference", "{train_reference}",
               "--output-dir", "{output_dir}", "--workdir", "{workdir}"],
      "entrypoint": "pipeline/run_pipeline.py"      <- harness 不读这个键
    }

但 harness 只读 `replay.argv`，并原样拿去 exec：

    command = ["/usr/bin/sandbox-exec", "-f", profile, *argv]

于是 sandbox-exec 试图执行一个名为 `--raw-root` 的程序 → 立刻 EXECUTION_FAILED。
契约要求 `replay.argv` 是**完整命令**（argv[0] 是解释器或脚本）。

本脚本做什么
------------
1. 把 `replay.argv` 修成完整命令（**只改这一个字段**：
   结果包 13 个 csv 一字节不动，清洗规则一行不改 —— 脚本会 md5 校验）
2. 复刻 harness 自己的收尾流程（`_run_replay` → `_replay_is_acceptable`
   → `snapshot_agent_workdir` → `_atomic_json(run_report.json)`）

用法: python tmp/recover_i48_val.py [实验目录]
"""
import asyncio
import hashlib
import json
import shutil
import sys
from pathlib import Path

PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
sys.path.insert(0, str(PROJ))

from agent.pi_harness import (  # noqa: E402
    PiHarnessResult,
    PiValidationHarness,
    PiValidationHarnessConfig,
    build_worker_model_environment,
    load_evaluation_manifest,
    snapshot_agent_workdir,
)

WT = PROJ / ".claude/worktrees/eval-movies-cleaning"
RUN = WT / "k29_agent_run"
DS = PROJ / "datasets/注入错误数据集/i48_less_1"
FULL = PROJ / "datasets/注入错误数据集/I48_atrial_fibrillation_v2_rawerror_test1.6pct_411M"
MANIFEST = PROJ / "evaluation_manifests/i48_atrial_fibrillation_ECG_v1.json"

EXP = Path(sys.argv[1]) if len(sys.argv) > 1 else RUN / "experiments/i48_1p6_val_seed666_v1"
HOST = EXP / "host"

FIXED_ARGV = [
    "python3", "pipeline/run_pipeline.py",
    "--raw-root", "{raw_root}",
    "--train-reference", "{train_reference}",
    "--output-dir", "{output_dir}",
    "--workdir", "{workdir}",
]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def result_fingerprint(root: Path) -> dict:
    """结果包 13 个 csv 的 md5 —— 证明修 argv 没有动结果。"""
    out = {}
    for p in sorted(root.rglob("*.csv")):
        out[str(p.relative_to(root))] = md5(p)
    return out


def patch_submission(path: Path) -> tuple[bool, str]:
    """只改 replay.argv；返回 (是否改了, 原因)。原文件备份成 .orig。"""
    if not path.is_file():
        return False, "不存在"
    payload = json.loads(path.read_text(encoding="utf-8"))
    replay = payload.get("replay") or {}
    argv = replay.get("argv") or []
    if argv and argv[0] in {"python", "python3"}:
        return False, "已经是完整命令"
    if not path.with_suffix(".json.orig").exists():
        shutil.copy2(path, path.with_suffix(".json.orig"))
    replay["argv"] = list(FIXED_ARGV)
    payload["replay"] = replay
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return True, f"argv[0] 从 {argv[0]!r} 改成 'python3'"


def best_score_of(host_dir: Path) -> tuple[float, int]:
    best = 0.0
    rounds = sorted((host_dir / "rounds").glob("round_*.json"))
    for p in rounds:
        d = json.loads(p.read_text(encoding="utf-8"))
        best = max(best, float((d.get("metrics") or {}).get("composite_score") or 0.0))
    return best, len(rounds)


def load_round_executions(host_dir: Path) -> list[dict]:
    out = []
    for p in sorted((host_dir / "rounds").glob("round_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        ex = d.get("execution")
        if isinstance(ex, dict):
            out.append(ex)
    return out


async def main() -> int:
    if not (HOST / "best_snapshot").is_dir():
        print("!! 没有 best_snapshot，无可收尾", file=sys.stderr)
        return 2

    # ---- 0. 修 argv（只改 submission.json 的 replay.argv）----
    before = result_fingerprint(HOST / "best_snapshot" / "result")
    for target in [HOST / "best_snapshot" / "submission.json",
                   EXP / "agent_workdir" / "submission.json"]:
        changed, why = patch_submission(target)
        print(f"  submission.json {target}: {'已修' if changed else '无需改'} ({why})")
    after = result_fingerprint(HOST / "best_snapshot" / "result")
    if before != after:
        print("!! 结果包在修 argv 过程中被改动了，中止", file=sys.stderr)
        return 4
    print(f"  结果包 {len(after)} 个 csv md5 校验：未改动 ✓")

    config = PiValidationHarnessConfig(
        project_root=PROJ,
        experiment_dir=EXP,
        train_raw=DS / "train/raw",
        train_reference=DS / "train/reference",
        validation_raw=DS / "validation/raw",
        validation_gold=DS / "validation/reference",
        evaluation_manifest=MANIFEST,
        dataset_manifest=FULL / "v2_rawerror_i48_final_report.json",
        max_rounds=20,
        patience=3,
        max_iters=10000,
    )
    harness = PiValidationHarness(config)
    harness._key_columns = load_evaluation_manifest(
        config.evaluation_manifest, config.validation_gold)
    harness._model_environment = build_worker_model_environment("react_planner", "MiniMax-M3")
    harness._round_executions = load_round_executions(HOST)

    best_score, rounds = best_score_of(HOST)
    print(f"\n实验目录 : {EXP}")
    print(f"已完成轮次: {rounds}   best_score = {best_score:.6f}")
    print("独立重放 best_snapshot ...", flush=True)

    replay = await harness._run_replay(harness.best_snapshot)
    print(f"  replay status = {replay.status}  score = {replay.score:.6f}  "
          f"rc={replay.returncode}  {replay.duration_seconds:.1f}s")
    if not harness._replay_is_acceptable(replay, best_score):
        print(f"!! 重放不可接受（{replay.score:.6f} < {best_score:.6f}），不收尾", file=sys.stderr)
        print((replay.stderr or "")[-3000:], file=sys.stderr)
        return 3

    final_snapshot = HOST / "reproducible_snapshot"
    snapshot_agent_workdir(harness.best_snapshot, final_snapshot)
    result = PiHarnessResult(
        status="SUCCESS_REPRODUCIBLE",
        stop_reason="rate_limited_finalized_after_submission_argv_fix",
        rounds=rounds,
        repair_rounds=0,
        best_score=max(best_score, 0.0),
        reproducible_score=replay.score,
        best_snapshot=harness.best_snapshot,
        reproducible_snapshot=final_snapshot,
    )
    payload = harness._run_report_payload(result)
    harness._atomic_json(HOST / "run_report.json", payload)
    print("\n收尾完成：reproducible_snapshot + run_report.json")
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
