"""校正类（单表纠错）结果评分 —— 统一的对外口径。

用法:
    python3 score_correction_result.py --dataset flights --result <结果csv> --out <json>

口径（与台账 §1.4 完全一致）：
  行级 exact  = 结果行 == 干净行（整行）        —— 官方 correction_evaluation.py 的口径
  cell 级     = 逐 cell：修对 / 改错 / 未检出 / 误伤
  v29 Det F1 / Rep F1（公式用 K29 1800 的真实 JSON 逐位复现验证过）

同时给出 **全表** 与 **test(=correction) 切分** 两套数字：
  - 全表：train + correction 全部行
  - test 切分：只有 correction 那部分（agent 训练时看不见 reference 的部分）

真值只用 `datasets/_source` 的原始 dirty/clean 对（hospital 的源文件是
`raha_hospital_{dirty,clean}_1000.csv`，且**列名不同、位置对应**）。
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
SRC = PROJ / "datasets/_source"
SPLIT_ROOT = PROJ / "datasets/correction"
TIME = re.compile(r"(\d{1,2}):(\d{2})")

# dataset -> (源 dirty, 源 clean, 切分目录, key 列, clean 列是否位置对应)
DATASETS = {
    "flights": ("flights/dirty.csv", "flights/clean.csv",
                 "flights_curated10_val20_v1", "tuple_id", False),
    "hospital": ("raha_hospital_dirty_1000.csv", "raha_hospital_clean_1000.csv",
                 "raha_hospital_curated10_val20_dense_v1", "index", True),
    "tax": ("tax/dirty.csv", "tax/clean.csv", "tax_curated10_v1", "__row_id", False),
    "beers": ("beers/dirty.csv", "beers/clean.csv", "beers_curated10_v1", "index", False),
    "rayyan": ("rayyan/dirty.csv", "rayyan/clean.csv", "rayyan_curated10_v1", "id", False),
}


def load(p):
    with open(p, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def keyset(path):
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        next(rd)
        return {r[0] for r in rd if r and r[0] != ""}


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0


def score(dataset, result_path, split_override=None):
    dirty_rel, clean_rel, split, key, positional = DATASETS[dataset]
    if split_override:
        split = split_override
    dcols, drows = load(SRC / dirty_rel)
    ccols, crows = load(SRC / clean_rel)
    rcols, rrows = load(result_path)
    if len(rrows) != len(drows):
        raise SystemExit(f"结果行数 {len(rrows)} != 脏表 {len(drows)}")
    if not positional and rcols != dcols:
        raise SystemExit(f"结果列名与脏表不一致：\n  脏表 {dcols}\n  结果 {rcols}")

    base = SPLIT_ROOT / split
    splits = {"train": keyset(base / "train/keys.csv"),
              "test(correction)": keyset(base / "correction/keys.csv")}

    # tax 没有自然主键：切分键 __row_id = 原表行序
    if key == "__row_id":
        for i, d in enumerate(drows):
            d["__row_id"] = str(i)

    # 结果按 key 对齐（tax 的结果包里没有 __row_id 列 ⇒ 按行序）
    if key in rcols:
        rmap = defaultdict(list)
        for r in rrows:
            rmap.setdefault(r.get(key, ""), []).append(r)
        idx = []
        for d in drows:
            b = rmap.get(d.get(key, ""))
            idx.append(b.pop(0) if b else None)
    else:
        idx = list(rrows) + [None] * max(0, len(drows) - len(rrows))

    out = {}
    for name, keys in splits.items():
        n = tp_det = tp_rep = b_wrong = fp = 0
        row_ex = row_ms = row_ic = dirty_rows = 0
        clean_tot = coll = 0
        for i, (d, c) in enumerate(zip(drows, crows)):
            if d.get(key, "") not in keys:
                continue
            r = idx[i]
            if r is None:
                continue
            row_dir = [d.get(dc, "") or "" for dc in dcols]
            row_cln = [c.get(cc, "") or "" for cc in ccols]
            row_res = [r.get(dc, "") or "" for dc in dcols]
            dirty_cells = []
            for j, dc in enumerate(dcols):
                dv = row_dir[j]
                cv = row_cln[j] if j < len(row_cln) else ""
                rv = row_res[j]
                if dv != cv:
                    dirty_cells.append((dv, cv, rv))
                    n += 1
                    if rv != dv:
                        tp_det += 1
                    if rv == cv:
                        tp_rep += 1
                    elif rv != dv:
                        b_wrong += 1
                else:
                    clean_tot += 1
                    if rv != dv:
                        coll += 1
            if dirty_cells:
                dirty_rows += 1
                if row_res == row_cln:
                    row_ex += 1
                elif row_res == row_dir:
                    row_ms += 1
                else:
                    row_ic += 1
        det_p = tp_det / (tp_det + coll) if (tp_det + coll) else 0.0
        det_r = tp_det / n if n else 0.0
        rep_p = tp_rep / (tp_rep + b_wrong + coll) if (tp_rep + b_wrong + coll) else 0.0
        rep_r = tp_rep / n if n else 0.0
        out[name] = dict(
            rows=len(keys), dirty_rows=dirty_rows, dirty_cells=n,
            row_exact=row_ex, row_missed=row_ms, row_incorrect=row_ic,
            row_repair_rate=row_ex / dirty_rows if dirty_rows else 1.0,
            cell_repaired=tp_rep, cell_missed=n - tp_det, cell_incorrect=b_wrong,
            cell_repair_rate=tp_rep / n if n else 1.0,
            cell_detect_rate=tp_det / n if n else 1.0,
            clean_cells=clean_tot, collateral=coll,
            collateral_rate=coll / clean_tot if clean_tot else 0.0,
            det_P=det_p, det_R=det_r, det_F1=f1(det_p, det_r),
            rep_P=rep_p, rep_R=rep_r, rep_F1=f1(rep_p, rep_r),
        )
    # 全表 = train + test 两段合并
    a, b = out["train"], out["test(correction)"]
    rows_tot = a["dirty_rows"] + b["dirty_rows"]
    out["whole_table"] = dict(
        rows=a["rows"] + b["rows"], dirty_rows=rows_tot,
        dirty_cells=a["dirty_cells"] + b["dirty_cells"],
        row_exact=a["row_exact"] + b["row_exact"],
        row_missed=a["row_missed"] + b["row_missed"],
        row_incorrect=a["row_incorrect"] + b["row_incorrect"],
        row_repair_rate=((a["row_exact"] + b["row_exact"]) / rows_tot) if rows_tot else 1.0,
        clean_cells=a["clean_cells"] + b["clean_cells"],
        collateral=a["collateral"] + b["collateral"],
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--result", required=True, help="agent 的结果 csv")
    ap.add_argument("--out", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--split", default="",
                    help="覆盖切分目录（默认用 DATASETS 里登记的那个）")
    args = ap.parse_args()

    res = score(args.dataset, args.result, args.split or None)
    payload = {
        "dataset": args.dataset,
        "label": args.label,
        "result_package": str(Path(args.result).resolve()),
        "split_dir": args.split or DATASETS[args.dataset][2],
        "metrics": res,
        "note": ("自算口径（datasets/_source 的 dirty/clean 对，不读 host_private）；"
                 "行级=整行等于干净行；Det/Rep F1 用 v29 公式"),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    for split in ("train", "test(correction)", "whole_table"):
        s = res[split]
        print(f"[{split}] 脏行 {s['dirty_rows']:,} / 脏 cell {s['dirty_cells']:,}")
        if "row_repair_rate" in s:
            print(f"    行级 exact {s['row_repair_rate']*100:6.2f}%   "
                  f"cell 级 exact {s['cell_repair_rate']*100:6.2f}%   "
                  f"cell 级检测 {s['cell_detect_rate']*100:6.2f}%")
            print(f"    Det F1 {s['det_F1']:.6f}   Rep F1 {s['rep_F1']:.6f}   "
                  f"误伤 {s['collateral']:,}/{s['clean_cells']:,} "
                  f"({s['collateral_rate']*100:.4f}%)")
        else:
            print(f"    行级 exact {s['row_repair_rate']*100:6.2f}%   "
                  f"误伤 {s['collateral']:,}")
    if args.out:
        print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
