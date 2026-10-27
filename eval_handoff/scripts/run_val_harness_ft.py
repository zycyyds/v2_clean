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
  profile: less1 | more1 | n18 | n18_more | i48
           （决定数据集视图、prompt、实验名前缀、dataset manifest、evaluation manifest）

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
    restore_agent_workdir,
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
    # N18 是另一个任务（不是 icu5000）：18 个输出文件、切分键 subject_id、
    # 有 ed/ + note/ 多模态、manifest 也不同。
    "n18": {
        "ds": PROJ / "datasets/注入错误数据集/n18_less_1",
        "prompt": PROMPT_DIR / "n18_1.6pct_val_prompt_v1.md",
        "exp_prefix": "n18_1p6_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集"
                     / "N18_ckd_ED_v2_rawerror_test1.6pct_490M"
                     / "v2_rawerror_n18_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/n18_ckd_ED_notes_v1.json",
    },
    # N18 4.7pct（more）：与 icu5000 4.7pct 同样的密度剖面
    #   train 2.33% / validation 0.89% / test 4.72%（照抄 K29 的分 split 率）
    "n18_more": {
        "ds": PROJ / "datasets/注入错误数据集/n18_more_1",
        "prompt": PROMPT_DIR / "n18_4.7pct_val_prompt_v1.md",
        "exp_prefix": "n18_4p7_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集"
                     / "N18_ckd_ED_v2_rawerror_test4.7pct_490M"
                     / "v2_rawerror_n18_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/n18_ckd_ED_notes_v1.json",
    },
    # I48（房颤）：13 个输出文件（7 顶层 + 4 icu + 2 ecg）、切分键 subject_id、
    # 多模态是 ecg/ 而不是 ed/+note/，manifest 也不同。
    # 密度剖面跟 N18 1.6pct 一样：train 1.30% / validation 2.24% / test 1.58%。
    "i48": {
        "ds": PROJ / "datasets/注入错误数据集/i48_less_1",
        "prompt": PROMPT_DIR / "i48_1.6pct_val_prompt_v1.md",
        "exp_prefix": "i48_1p6_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集"
                     / "I48_atrial_fibrillation_v2_rawerror_test1.6pct_411M"
                     / "v2_rawerror_i48_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/i48_atrial_fibrillation_ECG_v1.json",
    },
    # I48 4.7pct（more）：密度剖面是「反转」的 —— train 2.33% / validation 0.89% /
    # test 4.72%。validation 的错误密度比 train 低三倍、比 test 低五倍。
    "i48_more": {
        "ds": PROJ / "datasets/注入错误数据集/i48_more_1",
        "prompt": PROMPT_DIR / "i48_4.7pct_val_prompt_v1.md",
        "exp_prefix": "i48_4p7_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集"
                     / "I48_atrial_fibrillation_v2_rawerror_test4.7pct_411M"
                     / "v2_rawerror_i48_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/i48_atrial_fibrillation_ECG_v1.json",
    },
    # K29（胃炎/十二指肠炎）低密度档：train 400 / validation 800 / test 1,800 条错误
    # （fixed-v1 引擎造的，0 自然噪声）。切分键 hadm_id、12 个输出文件、
    # 多模态是 note/。视图走 datasets/val_views/（密封，父目录里没有别的东西）。
    # 任务书用 v2 —— v1 里写的密度是 4.7pct 的数字，跟这一档对不上；
    # v1 的 sha256 已被台账记为旧跑分（k29_1800_val_seed666_v1）的证据，不能改。
    "k29_1800": {
        "ds": PROJ / "datasets/val_views/k29_1800",
        "prompt": PROMPT_DIR / "k29_1800_val_prompt_v2.md",
        "exp_prefix": "k29_1800_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集"
                     / "K29_clean_v2_rawerror_10train_400_800_1800"
                     / "v2_rawerror_k29_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/k29_gastritis_duodenitis_v3_1.json",
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
        evaluation_manifest=prof.get(
            "eval_manifest",
            PROJ / "evaluation_manifests/icu_5000_v1.json"),
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

    # 续跑：把上一轮实验的 best_snapshot 灌进新实验的 workdir，
    # 让 agent 从自己上一轮的状态继续（analysis/ pipeline/ train_analysis.json 都在），
    # 而不是从头重推一遍。上一次跑被 API 限流掐断时用得上。
    #
    # 实现：harness 的 _prepare 要求「实验目录全新且为空」，且最后才 mkdir workdir。
    # 所以不能在 run() 之前预置 workdir —— 改为包装 _prepare：
    # 先让它按原样建好空的 workdir 和 host 目录，再把种子灌进去。
    seed_from = os.environ.get("SEED_FROM", "").strip()
    if seed_from:
        seed = Path(seed_from)
        if not (seed / "result").is_dir():
            print(f"!! SEED_FROM 不是合法的快照（缺 result/）：{seed}", file=sys.stderr)
            return 5
        _orig_prepare = harness._prepare

        # 注意：harness._prepare 是**实例属性**，不是绑定方法，
        # 所以 self._prepare(prompt) 只会传 prompt 一个参数 —— 这里用闭包。
        def _prepare_then_seed(prompt: str = "") -> None:
            _orig_prepare(prompt)
            restore_agent_workdir(harness.agent_workdir, seed)
            print(f"[seed] 已从 {seed} 恢复 agent workdir（续跑）", flush=True)

        harness._prepare = _prepare_then_seed
        print(f"[seed] 续跑模式，种子 = {seed}", flush=True)

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
