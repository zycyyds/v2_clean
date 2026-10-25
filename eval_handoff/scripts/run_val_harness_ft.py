"""带容错的 val harness 驱动（不改 harness 源码）。

为什么需要它：连续两次跑批都死在 worker 的模型调用上 ——
  v1: 'key_commands' is a required property（第 5 轮，4 轮成果保住了）
  v2: Pi worker ReadTimeout（第 1 轮，一轮都没交付，全丢）
harness 的 run() 对 worker.run_turn() 抛出的异常零容错，一次抖动整跑完蛋。

这个驱动在 harness 外面加两层保护：
  1. 给 run_turn 加重试：模型调用失败（ReadTimeout / 429 / 畸形响应）自动重试
  2. 兜底收尾：harness.run() 抛异常时，不让整跑白费 —— 走它自己的收尾流程
     （独立重放 best_snapshot → 可接受就冻结 reproducible_snapshot + 写 run_report.json）
     这样「崩在第 1 轮」也能留下第 1 轮之前的成果

用法: python run_val_harness_ft.py <profile> [实验目录后缀]
  profile: less1 | more1   （决定数据集视图、prompt、实验名前缀、dataset manifest）

环境变量可覆盖默认路径（换机器/换数据集时用）：
  PROJ       项目根，默认 /Users/mac/PycharmProjects/v2_clean
  RUN        agent run 目录，默认 $PROJ/k29_agent_run
  PROMPT_DIR prompt 目录，默认 $RUN/prompts
"""
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

PROJ = Path(os.environ.get("PROJ", "/Users/mac/PycharmProjects/v2_clean"))
sys.path.insert(0, str(PROJ))

from agent.pi_harness import (  # noqa: E402
    PiHarnessResult,
    PiValidationHarness,
    PiValidationHarnessConfig,
    build_worker_model_environment,
    load_evaluation_manifest,
    snapshot_agent_workdir,
)
from agent.pi_worker_client import JsonlWorkerClient  # noqa: E402

RUN = Path(os.environ.get("RUN", str(PROJ / "k29_agent_run")))
PROMPT_DIR = Path(os.environ.get("PROMPT_DIR", str(RUN / "prompts")))

# profile → (视图目录, prompt 文件, 实验名前缀, dataset manifest 所在的完整数据集)
PROFILES = {
    "less1": {
        "ds": PROJ / "datasets/val_views/icu5000_less1",
        "prompt": PROMPT_DIR / "icu5000_1.6pct_val_prompt_v1.md",
        "exp_prefix": "icu5000_1p6_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集"
                     / "icu_5000_v2_rawerror_test1.6pct_614M"
                     / "v2_rawerror_icu5000_final_report.json"),
    },
    "more1": {
        "ds": PROJ / "datasets/val_views/icu5000_more1",
        "prompt": PROMPT_DIR / "icu5000_4.7pct_val_prompt_v1.md",
        "exp_prefix": "icu5000_4p7_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集"
                     / "icu_5000_v2_rawerror_test4.7pct_614M"
                     / "v2_rawerror_icu5000_final_report.json"),
    },
}

MAX_TURN_RETRIES = 4
RETRY_SLEEP = 15

_orig_run_turn = JsonlWorkerClient.run_turn


async def _retrying_run_turn(self, prompt: str):
    """模型调用失败自动重试。worker 进程死了就没救了，直接抛。"""
    last: Exception | None = None
    for attempt in range(MAX_TURN_RETRIES):
        try:
            return await _orig_run_turn(self, prompt)
        except Exception as exc:
            last = exc
            dead = getattr(self, "process", None) is not None and self.process.returncode is not None
            print(f"[retry] run_turn 失败 ({type(exc).__name__}: {exc}) "
                  f"{'worker 已死，放弃' if dead else f'重试 {attempt + 1}/{MAX_TURN_RETRIES}'}",
                  flush=True)
            if dead:
                raise
            await asyncio.sleep(RETRY_SLEEP)
    raise last  # type: ignore[misc]


JsonlWorkerClient.run_turn = _retrying_run_turn


def make_config(exp_name: str, prof: dict) -> PiValidationHarnessConfig:
    ds = prof["ds"]
    return PiValidationHarnessConfig(
        project_root=PROJ,
        experiment_dir=RUN / "experiments" / exp_name,
        train_raw=ds / "train/raw",
        train_reference=ds / "train/reference",
        validation_raw=ds / "validation/raw",
        validation_gold=ds / "validation/reference",
        evaluation_manifest=PROJ / "evaluation_manifests/icu_5000_v1.json",
        dataset_manifest=prof["manifest"],
        max_rounds=20,
        patience=3,
        max_iters=10000,
    )


async def finalize(harness: PiValidationHarness, best_score: float, why: str) -> int:
    """harness 自己最后那段收尾：独立重放 → 通过就冻结 + 写 report。"""
    if not harness.best_snapshot.is_dir():
        print(f"!! {why}：没有 best_snapshot，无可收尾", file=sys.stderr)
        return 2
    print(f"\n收尾（{why}）：独立重放 best_snapshot ...", flush=True)
    replay = await harness._run_replay(harness.best_snapshot)
    print(f"  replay status={replay.status} score={replay.score:.6f}", flush=True)
    if not harness._replay_is_acceptable(replay, best_score):
        print(f"!! 重放不可接受（{replay.score:.6f} < {best_score:.6f}），不收尾", file=sys.stderr)
        return 3
    final_snapshot = harness.host_dir / "reproducible_snapshot"
    snapshot_agent_workdir(harness.best_snapshot, final_snapshot)
    result = PiHarnessResult(
        status="SUCCESS_REPRODUCIBLE",
        stop_reason=why,
        rounds=len(list((harness.host_dir / "rounds").glob("round_*.json"))),
        repair_rounds=0,
        best_score=max(best_score, 0.0),
        reproducible_score=replay.score,
        best_snapshot=harness.best_snapshot,
        reproducible_snapshot=final_snapshot,
    )
    harness._atomic_json(harness.host_dir / "run_report.json",
                         harness._run_report_payload(result))
    print("收尾完成：reproducible_snapshot + run_report.json")
    print(json.dumps(harness._run_report_payload(result), ensure_ascii=False, indent=1))
    return 0


def best_score_of(harness: PiValidationHarness) -> float:
    best = 0.0
    for p in sorted((harness.host_dir / "rounds").glob("round_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        best = max(best, float((d.get("metrics") or {}).get("composite_score") or 0.0))
    return best


async def main() -> int:
    profile = sys.argv[1] if len(sys.argv) > 1 else "less1"
    if profile not in PROFILES:
        print(f"未知 profile: {profile}（可选: {', '.join(PROFILES)}）", file=sys.stderr)
        return 2
    prof = PROFILES[profile]
    suffix = sys.argv[2] if len(sys.argv) > 2 else "v1"
    exp_name = f"{prof['exp_prefix']}_{suffix}"
    config = make_config(exp_name, prof)
    harness = PiValidationHarness(config)
    prompt = prof["prompt"].read_text(encoding="utf-8")
    print(f"[profile={profile}] experiment={config.experiment_dir}", flush=True)
    print(f"[profile={profile}] prompt={prof['prompt']}", flush=True)
    print(f"[profile={profile}] dataset={prof['ds']}", flush=True)

    try:
        result = await harness.run(prompt)
        print(json.dumps({
            "status": result.status, "stop_reason": result.stop_reason,
            "rounds": result.rounds, "repair_rounds": result.repair_rounds,
            "best_score": result.best_score,
            "reproducible_score": result.reproducible_score,
        }, ensure_ascii=False, indent=1))
        return 0 if result.status == "SUCCESS_REPRODUCIBLE" else 1
    except BaseException:
        traceback.print_exc()
        print("\n!! harness 抛异常 —— 转入兜底收尾，保留已有轮次", file=sys.stderr)
        harness._key_columns = load_evaluation_manifest(
            config.evaluation_manifest, config.validation_gold)
        harness._model_environment = build_worker_model_environment("react_planner", "MiniMax-M3")
        return await finalize(harness, best_score_of(harness), "worker_crash_finalized")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
