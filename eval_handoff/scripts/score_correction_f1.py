"""校正类数据集（单表纠错）—— 换算到 **v29 的 Det F1 / Rep F1** 口径。

为什么能换算：v29 的错误级指标是按「注入目标」算的，而校正类这一族的
注入目标就是 **dirty cell**（`_source/probe_report.json` 证实：
dirty_rows == clean_rows ⇒ 无行增删；dirty_cells_total == 全部差异 ⇒ 无 natural noise）。

v29 的公式（用 K29 1800 agent 的 JSON 反推核对过，逐位对得上）：
  N         = 注入目标数
  TP_det    = 改了的目标数（b_clean + b_wrong + dup_removed + leak_removed）
  TP_rep    = 改成干净值的目标数（b_clean + dup_removed + leak_removed）
  b_wrong   = 改了但改成错值的目标数
  fp_det    = 误改的干净 cell 数
  det.P = TP_det / (TP_det + fp_det)        det.R = TP_det / N
  rep.P = TP_rep / (TP_rep + b_wrong + fp_det)   rep.R = TP_rep / N
  F1  = 2PR / (P + R)

切分：train = agent 看得见 reference 的那几行；test = correction 切分（其余全部）。
"""
import csv
import json
from pathlib import Path

PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
SPLIT = PROJ / "datasets/correction"
SRC = PROJ / "datasets/_source"
RUNS = PROJ / "runs"

RUNS_MAP = {
    "tax": (PROJ / "agent_work/tax_correction_v1/result_package/tax.csv", "__row_id"),
    "beers": (PROJ / "agent_work/beers_correction_v1/result_package/beers.csv", "index"),
    "rayyan": (PROJ / "rayyan_clean_v1/result_package/cleaned.csv", "id"),
    "flights": (PROJ / "agent_work/flights_correction/output/flights.csv", "tuple_id"),
}


def load(p):
    with Path(p).open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def keyset(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        next(rd)
        return {r[0] for r in rd if r and r[0] != ""}


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0


def score(name):
    res_path, key = RUNS_MAP[name]
    dcols, drows = load(SRC / name / "dirty.csv")
    ccols, crows = load(SRC / name / "clean.csv")
    rcols, rrows = load(res_path)
    assert len(rrows) == len(drows), f"{name}: agent 输出行数 {len(rrows)} != dirty {len(drows)}"

    base = SPLIT / f"{name}_curated10_v1"
    splits = {"train": keyset(base / "train/keys.csv"),
              "test(correction)": keyset(base / "correction/keys.csv")}

    if key == "__row_id":                      # tax：合成的位置键
        for i, d in enumerate(drows):
            d["__row_id"] = str(i)

    if key in rcols:
        rmap = {}
        for r in rrows:
            rmap.setdefault(r.get(key, ""), []).append(r)
        idx = []
        for d in drows:
            b = rmap.get(d.get(key, ""))
            idx.append(b.pop(0) if b else None)
    else:
        idx = list(rrows)[:len(drows)] + [None] * max(0, len(drows) - len(rrows))

    out = {}
    for split, keys in splits.items():
        n = tp_det = tp_rep = b_wrong = fp = 0
        for i, (d, c) in enumerate(zip(drows, crows)):
            if d.get(key, "") not in keys:
                continue
            r = idx[i]
            if r is None:
                continue
            for dc, cc in zip(dcols, ccols):
                dv, cv = d.get(dc, ""), c.get(cc, "")
                rv = r.get(dc, "") or ""
                if dv != cv:
                    n += 1
                    if rv != dv:
                        tp_det += 1
                    if rv == cv:
                        tp_rep += 1
                    elif rv != dv:
                        b_wrong += 1
                elif rv != dv:
                    fp += 1
        det_p = tp_det / (tp_det + fp) if (tp_det + fp) else 0.0
        det_r = tp_det / n if n else 0.0
        rep_p = tp_rep / (tp_rep + b_wrong + fp) if (tp_rep + b_wrong + fp) else 0.0
        rep_r = tp_rep / n if n else 0.0
        out[split] = dict(
            injected_targets=n, tp_det=tp_det, tp_rep=tp_rep, b_wrong=b_wrong, fp_det=fp,
            fn_det=n - tp_det, fn_rep=n - tp_rep,
            det_P=det_p, det_R=det_r, det_F1=f1(det_p, det_r),
            rep_P=rep_p, rep_R=rep_r, rep_F1=f1(rep_p, rep_r),
        )
    return out


RUNS.mkdir(parents=True, exist_ok=True)
all_res = {}
for name in RUNS_MAP:
    res = score(name)
    all_res[name] = res
    (RUNS / f"correction_{name}_f1_by_split.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"=== {name}")
    for split in ("train", "test(correction)"):
        s = res[split]
        print(f"  [{split}] 注入目标 {s['injected_targets']:,}"
              f"（改过 {s['tp_det']:,} / 改对 {s['tp_rep']:,} / 改错 {s['b_wrong']:,}）"
              f"  fp_det {s['fp_det']:,}")
        print(f"     Det : P {s['det_P']:.6f}  R {s['det_R']:.6f}  **F1 {s['det_F1']:.6f}**")
        print(f"     Rep : P {s['rep_P']:.6f}  R {s['rep_R']:.6f}  **F1 {s['rep_F1']:.6f}**")
    print()

print("→ runs/correction_<name>_f1_by_split.json")
