#!/usr/bin/env python3
"""最终整理：只留 5 个「我们的 1.6/4.7」+ 5 个「台账记录的对应基线」，按疾病分组。

要移除的（都是本次打包我自己生成的产物）：
  N18/、K29/1800档_cc基线_ClaudeCode/、ICU5000/1.6pct_cc基线_ClaudeCode/、
  ICU5000/4.7pct_cc基线_ClaudeCode/、ICU5000/codex基线_evalrest/、I48/codex基线_evalrest/、
  Z_附加基线/
K29 的两个 codex 尝试拆成 1.6pct / 4.7pct 两档的基线目录。
"""
import json
import shutil
from pathlib import Path

OUT = Path("/Users/mac/Downloads/ceshi1")
RUNS = Path("/Users/mac/PycharmProjects/v2_clean/runs")

REMOVE = [
    "N18",
    "K29/1800档_cc基线_ClaudeCode",
    "ICU5000/1.6pct_cc基线_ClaudeCode",
    "ICU5000/4.7pct_cc基线_ClaudeCode",
    "ICU5000/codex基线_evalrest",
    "I48/codex基线_evalrest",
    "Z_附加基线",
]


def split_k29_codex():
    """K29 的两个 codex 尝试 → 拆成 1.6pct / 4.7pct 两个基线目录。"""
    src = OUT / "K29/codex基线_evalrest_无同档"
    if not src.is_dir():
        return
    mapping = [
        ("1.6pct_codex基线", "codex_尝试A_scorer.log", "k29_codex_score",
         "K29 1.6pct 的 codex 基线（⚠️ codex 跑的是**原始 eval-rest 数据集**，"
         "不是 mimicraw、也不是 1.6pct 专属档；K29 的 mimicraw 两档**没有**同档基线）"),
        ("4.7pct_codex基线", "codex_尝试B_scorer.log", "k29_codex_v6",
         "K29 4.7pct 的 codex 基线（⚠️ 同上：原始 eval-rest 数据集；这是 K29 codex 的第 2 次尝试）"),
    ]
    for dirname, fname, runs_dir, note in mapping:
        d = OUT / "K29" / dirname
        d.mkdir(parents=True, exist_ok=True)
        s = src / fname
        if s.is_file():
            shutil.copy2(s, d / "codex_scorer.log")
        payload = {"dataset": "K29（eval-rest 数据集，非 mimicraw、非档位专属）",
                   "who": "codex（OpenAI Codex CLI，2026-09-19）",
                   "note": note,
                   "files": {}}
        # 从 scorer.log 里抽 composite
        try:
            txt = (d / "codex_scorer.log").read_text(encoding="utf-8", errors="replace")
            objs = []
            dec = json.JSONDecoder()
            idx = 0
            while idx < len(txt):
                try:
                    obj, end = dec.raw_decode(txt, idx)
                except ValueError:
                    idx += 1
                    continue
                objs.append(obj)
                idx = end
            if objs:
                o = objs[-1]
                payload["files"]["codex_scorer.log"] = {
                    "composite": o.get("metrics", {}).get("composite_score"),
                    "schema_f1": o.get("metrics", {}).get("schema_f1"),
                    "key_structure_f1": o.get("metrics", {}).get("key_structure_f1"),
                    "row_aligned_cell_f1": o.get("metrics", {}).get("row_aligned_cell_f1"),
                    "exact_row_f1": o.get("metrics", {}).get("exact_row_f1"),
                    "note": "scorer.log（完整 JSON，含逐文件明细）",
                }
        except Exception as e:
            payload["files"]["codex_scorer.log"] = {"error": str(e)}
        (d / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    # 旧的合并目录移除
    for f in src.iterdir():
        target = OUT / "K29" / f.name
        if target.exists():
            continue
        shutil.move(str(f), str(target))
    shutil.rmtree(src)


def main():
    split_k29_codex()
    for rel in REMOVE:
        p = OUT / rel
        if p.is_dir():
            shutil.rmtree(p)
            print("  移除:", rel)
    print("\n最终结构：")
    for dz in sorted(p for p in OUT.iterdir() if p.is_dir()):
        for sub in sorted(p for p in dz.iterdir() if p.is_dir()):
            print("   %s/%s" % (dz.name, sub.name))


if __name__ == "__main__":
    main()
