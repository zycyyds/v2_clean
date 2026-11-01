"""校正类数据集 —— 按 **train / correction(=test) 切分** 的官方口径评分。

官方口径（workflow/correction_evaluation.py）：
  逐行比较三张表（dirty / result / clean）
    exact    = 结果行 == 干净行（整行）
    missed   = 结果行 == 脏行（整行）
    incorrect= 其余（改了一半 / 改错了）
  修复率 = exact / 脏行数

额外给出 **cell 级**（逐个被注入的 cell）：
  检测 = 结果值 != 脏值（agent 动过）
  修复 = 结果值 == 干净值
  未检出 = 结果值 == 脏值
  改错 = 其它

以及误伤：本来干净、结果改掉的 cell。

真值只用 datasets/_source 的 dirty/clean 对（不读 host_private）。
"""
import csv
import json
import re
from pathlib import Path

PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
SPLIT = PROJ / "datasets/correction"
SRC = PROJ / "datasets/_source"
RUNS = PROJ / "runs"
TIME = re.compile(r"(\d{1,2}):(\d{2})")

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


def shift(v, k):
    """把脏字符串里的时间挪 k 分钟，其余原样保留。"""
    m = TIME.search(v or "")
    if not m:
        return None
    total = (int(m.group(1)) * 60 + int(m.group(2)) + k) % 1440
    hs = str(total // 60) if len(m.group(1)) == 1 else f"{total // 60:02d}"
    return v[:m.start()] + f"{hs}:{total % 60:02d}" + v[m.end():]


def score(name, offsets=None):
    res_path, key = RUNS_MAP[name]
    dcols, drows = load(SRC / name / "dirty.csv")
    ccols, crows = load(SRC / name / "clean.csv")
    rcols, rrows = load(res_path)

    base = SPLIT / f"{name}_curated10_v1"
    splits = {"train": keyset(base / "train/keys.csv"),
              "test(correction)": keyset(base / "correction/keys.csv")}

    # tax 没有自然主键：切分键 __row_id = 原表行序
    if key == "__row_id":
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
        n_rows = dirty_rows = row_ex = row_ms = row_ic = 0
        tot = ce = cm = ci = 0
        clean_tot = coll = 0
        for i, (d, c) in enumerate(zip(drows, crows)):
            if d.get(key, "") not in keys:
                continue
            r = idx[i]
            if r is None:
                continue
            n_rows += 1
            dirty = []
            for dc, cc in zip(dcols, ccols):
                dv, cv = d.get(dc, ""), c.get(cc, "")
                if dv != cv:
                    dirty.append((dc, dv, cv))
                    tot += 1
                else:
                    clean_tot += 1
                    if (r.get(dc, "") or "") != dv:
                        coll += 1
            if not dirty:
                continue
            dirty_rows += 1
            for dc, dv, cv in dirty:
                rv = r.get(dc, "") or ""
                if offsets and dc in offsets and rv != cv:
                    cand = shift(dv, offsets[dc])
                    if cand is not None and cand == cv:
                        rv = cand
                if rv == cv:
                    ce += 1
                elif rv == dv:
                    cm += 1
                else:
                    ci += 1
            # 官方行级：整行相等才算 exact / missed
            row_res = [r.get(dc, "") or "" for dc in dcols]
            row_dir = [d.get(dc, "") or "" for dc in dcols]
            row_cln = [c.get(cc, "") or "" for cc in ccols]
            if row_res == row_cln:
                row_ex += 1
            elif row_res == row_dir:
                row_ms += 1
            else:
                row_ic += 1
        out[split] = dict(
            rows=n_rows, dirty_rows=dirty_rows, dirty_cells=tot,
            row_exact=row_ex, row_missed=row_ms, row_incorrect=row_ic,
            row_repair_rate=row_ex / dirty_rows if dirty_rows else 1.0,
            row_detect_rate=(row_ex + row_ic) / dirty_rows if dirty_rows else 1.0,
            cell_repaired=ce, cell_missed=cm, cell_incorrect=ci,
            cell_repair_rate=ce / tot if tot else 1.0,
            cell_detect_rate=(ce + ci) / tot if tot else 1.0,
            clean_cells=clean_tot, collateral=coll,
            collateral_rate=coll / clean_tot if clean_tot else 0.0,
        )
    return out


RUNS.mkdir(parents=True, exist_ok=True)
all_res = {}
for name in RUNS_MAP:
    res = score(name)
    all_res[name] = res
    (RUNS / f"correction_{name}_score_by_split.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"=== {name}")
    for split in ("train", "test(correction)"):
        s = res[split]
        print(f"  [{split}] 脏行 {s['dirty_rows']:,} / 脏 cell {s['dirty_cells']:,}"
              f"（总行 {s['rows']:,}）")
        print(f"     行级(官方)：修复 {s['row_repair_rate']*100:6.2f}%"
              f"   检测 {s['row_detect_rate']*100:6.2f}%"
              f"   未检出 {s['row_missed']:,}  改错 {s['row_incorrect']:,}")
        print(f"     cell 级  ：修复 {s['cell_repair_rate']*100:6.2f}%"
              f"   检测 {s['cell_detect_rate']*100:6.2f}%"
              f"   未检出 {s['cell_missed']:,}  改错 {s['cell_incorrect']:,}")
        print(f"     误伤     ：{s['collateral']:,} / {s['clean_cells']:,} 干净 cell"
              f" ({s['collateral_rate']*100:.4f}%)")
    print()

# ---- flights 改进投影（test 切分）----
proj = {}
for tag, offs in (("train众数 act_dep-1", {"act_dep_time": -1}),
                  ("train众数 act_dep-1+act_arr+3", {"act_dep_time": -1, "act_arr_time": 3}),
                  ("oracle act_dep-1+act_arr+2", {"act_dep_time": -1, "act_arr_time": 2})):
    r = score("flights", offsets=offs)["test(correction)"]
    proj[tag] = r
    print(f"flights[test] + {tag:28s} 行级修复 {r['row_repair_rate']*100:6.2f}%"
          f"   cell级修复 {r['cell_repair_rate']*100:6.2f}%")
(RUNS / "correction_flights_score_by_split_projection.json").write_text(
    json.dumps(proj, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n→ runs/correction_<name>_score_by_split.json")
