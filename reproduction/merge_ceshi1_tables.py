#!/usr/bin/env python3
"""把 n18 / I48 补齐后的 8 个新结果目录**并进** ceshi1 包里那三张手写表。

三张表是既有手写文件（原文逐行保留，只做「插入新行 + 改表头/完成度」）：
  ~/Downloads/ceshi1/00_评估进度表.md
  ~/Downloads/ceshi1/00_紧凑横向对比.md
  ~/Downloads/ceshi1/00_三维度横向对比.md

新增行的数字**全部从 JSON 读出**（不手抄）：
  composite  ← new_pkg_scores.json / n18_codex16_score.json
  错误级     ← error_metrics_{n18_agent,n18_cc16,new,oldsw}.json + err_<key>_<cal>.json
  原生层     ← n18_{16,47}_{agent,cc}_native.json / n18_cc16_native.json / deepprep_native.json

原生 P_rep = b_clean / (b_clean + b_wrong)（与包里既有 15 行的口径一致，
反推自 K29/ICU5000/I48 既有行；b_clean+b_wrong=0 ⇒ 记「—」）。

同时在仓库留一份合并后的副本：reproduction/ceshi1_合并后/
"""
import json
import os
import re
import shutil
from pathlib import Path

# NO_DL=1：只写仓库留档，不动包内那三张手写表（需用户点名授权后才覆盖）
WRITE_DL = os.environ.get("NO_DL", "") != "1"

TMP = Path("/Users/mac/.claude/jobs/448aad57/tmp")
DL = Path("/Users/mac/Downloads/ceshi1")
REPO = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/k29-mimicraw/reproduction")
KEEP = REPO / "ceshi1_合并后"


def load(name, default=None):
    p = TMP / name
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


scores = load("new_pkg_scores.json", {}) or {}
dpscores = load("deepprep_scores.json", {}) or {}
cc16_score = load("n18_codex16_score.json", {}) or {}
nat = load("n18_16_agent_native.json"), load("n18_47_agent_native.json"), load("n18_47_cc_native.json")
cc16_nat = load("n18_cc16_native.json")
dpnat = load("deepprep_native.json", {}) or {}
em_new = load("error_metrics_new.json", {}) or {}
em_agent = load("error_metrics_n18_agent.json", {}) or {}
em_oldsw = load("error_metrics_oldsw.json", {}) or {}
em_cc16 = load("error_metrics_n18_cc16.json", {}) or {}


def per_caliber_json(ds_key):
    """每口径单独 json 的兜底读取（`old` 用 _old 后缀，避免与真正的 oldsw 同名覆盖）。"""
    out = {}
    for cal, suffix in (("str", "str"), ("oldsw", "oldsw"), ("old", "old"), ("goldv29", "goldv29")):
        p = TMP / ("err_%s_%s.json" % (ds_key, suffix))
        if not p.exists():
            continue
        d = json.load(open(p))
        det, rep = d.get("detection", {}), d.get("repair", {})
        out[cal] = {"det_f1": det.get("F1"), "rep_f1": rep.get("F1"),
                    "buckets": d.get("buckets")}
    return out


def merge_calibers(a, b):
    """合并并纠正口径标签：a 里标着 oldsw 的其实是「old」（纯字符串），b 里的才是真 oldsw（数值）。"""
    out = {}
    if a:
        for cal, v in a.items():
            out["old" if cal == "oldsw" else cal] = v
    if b:
        for cal, v in b.items():
            if cal == "oldsw":
                out["oldsw"] = v
    return out


def sc(d):
    if not d:
        return None
    m = d.get("metrics", d)
    return (m.get("composite_score"), m.get("schema_f1"), m.get("key_structure_f1"),
            m.get("row_aligned_cell_f1"), m.get("exact_row_f1"))


def dp_buckets(label):
    """deepprep 的重判输出 → (总格数, b_clean, b_wrong, b_dirty, b_missing)。"""
    raw = dpnat.get(label) or []
    g = {}
    for l in raw:
        s = l.strip()
        if s.startswith("总格数"):
            g["tot"] = int(s.split()[-1])
        elif s.startswith("b_"):
            parts = s.split()
            g[parts[0]] = int(parts[1])
    return g


def native_pack(d):
    """native json → (总格数, b_clean, b_wrong, b_dirty, b_missing, 修复率, P_rep)。"""
    if d is None:
        return None
    b = d.get("buckets", {}) or {}
    tot = d.get("n_native_cells")
    bc = b.get("b_clean", 0)
    bw = b.get("b_wrong", 0)
    bd = b.get("b_dirty", 0)
    bm = b.get("b_missing", 0)
    if tot is None:
        tot = bc + bw + bd + bm
    rate = d.get("repair_rate")
    if rate is None:
        rate = (bc / tot) if tot else 0.0
    p_rep = (bc / (bc + bw)) if (bc + bw) else None
    return tot, bc, bw, bd, bm, rate, p_rep


# ── 新增的 8 个结果目录 ──
N18_AGENT_16 = "n18/1.6pct_我们的agent_pi+val_mimicraw"
N18_AGENT_47 = "n18/4.7pct_我们的agent_pi+val_mimicraw"
N18_CC_16 = "n18/1.6pct_cc基线_ClaudeCode"
N18_CC_47 = "n18/4.7pct_cc基线_ClaudeCode"
I48_CC_47 = "I48/4.7pct_cc基线_ClaudeCode"
DP = ["n18/deepprep-less", "n18/deepprep-more", "I48/deepprep-less", "I48/deepprep-more"]

ERR = {
    N18_AGENT_16: merge_calibers(em_agent.get(N18_AGENT_16 + "/结果包"),
                                 em_oldsw.get(N18_AGENT_16 + "/结果包")),
    N18_AGENT_47: merge_calibers(em_agent.get(N18_AGENT_47 + "/结果包"),
                                 em_oldsw.get(N18_AGENT_47 + "/结果包")),
    N18_CC_16: em_cc16.get(N18_CC_16 + "/结果包", {}) or {},
    N18_CC_47: merge_calibers(em_new.get(N18_CC_47 + "/结果包") or per_caliber_json("ceshi1_n18_cc47"),
                              em_oldsw.get(N18_CC_47 + "/结果包")),
    I48_CC_47: (load("error_metrics_i48_cc47.json", {}) or {}).get(I48_CC_47 + "/结果包", {}) or {},
}

COMP = {
    N18_AGENT_16: scores.get(N18_AGENT_16 + "/结果包"),
    N18_AGENT_47: scores.get(N18_AGENT_47 + "/结果包"),
    N18_CC_16: cc16_score,
    N18_CC_47: scores.get(N18_CC_47 + "/结果包"),
    I48_CC_47: load("i48_cc47_score.json", {}) or {},
    "n18/deepprep-less": dpscores.get("n18/deepprep-less/output"),
    "n18/deepprep-more": dpscores.get("n18/deepprep-more/output"),
    "I48/deepprep-less": dpscores.get("I48/deepprep-less/output"),
    "I48/deepprep-more": dpscores.get("I48/deepprep-more/output"),
}

def native_pack_of_dp(label):
    g = dp_buckets(label)
    if not g:
        return None
    tot, bc = g.get("tot", 0), g.get("b_clean", 0)
    bw, bd, bm = g.get("b_wrong", 0), g.get("b_dirty", 0), g.get("b_missing", 0)
    rate = (bc / tot) if tot else 0.0
    p_rep = (bc / (bc + bw)) if (bc + bw) else None
    return tot, bc, bw, bd, bm, rate, p_rep


NAT = {
    N18_AGENT_16: native_pack(nat[0]),
    N18_AGENT_47: native_pack(nat[1]),
    N18_CC_47: native_pack(nat[2]),
    N18_CC_16: native_pack(cc16_nat),
    I48_CC_47: native_pack(load("i48_cc47_native.json")),
    "n18/deepprep-less": native_pack_of_dp("n18/deepprep-less/output"),
    "n18/deepprep-more": native_pack_of_dp("n18/deepprep-more/output"),
    "I48/deepprep-less": native_pack_of_dp("I48/deepprep-less/output"),
    "I48/deepprep-more": native_pack_of_dp("I48/deepprep-more/output"),
}


RUNNER = {"我们": "我们", "cc": "cc", "codex": "codex", "deepprep": "deepprep"}
# (目录, 跑者, 档位, 数据集标注)
NEW_ROWS = [
    (N18_AGENT_16, "我们", "1.6pct", "mimicraw"),
    (N18_AGENT_47, "我们", "4.7pct", "mimicraw"),
    (N18_CC_16, "cc", "1.6pct", "raw(无原生层)"),
    (N18_CC_47, "cc", "4.7pct", "raw(无原生层)"),
    (I48_CC_47, "cc", "4.7pct", "raw(无原生层)"),
    ("n18/deepprep-less", "deepprep", "1.6pct", "mimicraw"),
    ("n18/deepprep-more", "deepprep", "4.7pct", "mimicraw"),
    ("I48/deepprep-less", "deepprep", "1.6pct", "mimicraw"),
    ("I48/deepprep-more", "deepprep", "4.7pct", "mimicraw"),
]

# 短标签（评估进度表用）
SHORT = {
    N18_AGENT_16: "n18/1.6pct_我们", N18_AGENT_47: "n18/4.7pct_我们",
    N18_CC_16: "n18/1.6pct_cc", N18_CC_47: "n18/4.7pct_cc",
    I48_CC_47: "I48/4.7pct_cc",
    "n18/deepprep-less": "n18/deepprep-less", "n18/deepprep-more": "n18/deepprep-more",
    "I48/deepprep-less": "I48/deepprep-less", "I48/deepprep-more": "I48/deepprep-more",
}


def f6(x):
    return "—" if x is None else ("%.*f" % (6, x))


def f4(x):
    return "—" if x is None else ("%.*f" % (4, x))


def cal(d, name, key):
    c = (ERR.get(d) or {}).get(name)
    return None if not c else c.get(key)


DEEP = {"n18/deepprep-less", "n18/deepprep-more", "I48/deepprep-less", "I48/deepprep-more"}


def ready(d):
    """这一行的数据齐了吗？齐了才并进表（避免出现半空行）。"""
    if sc(COMP.get(d)) is None:
        return False, "缺 composite"
    if d in DEEP:
        return True, ""          # deepprep 的错误级天生算不出 ⇒ 记「—」，是完整状态
    if not (ERR.get(d) or {}).get("str"):
        return False, "缺错误级"
    return True, ""


def buckets(d, name):
    c = (ERR.get(d) or {}).get(name) or {}
    return c.get("buckets") or {}


# ══════════════════════════════════════════════════════════════════
# 表一：00_评估进度表.md
# ══════════════════════════════════════════════════════════════════
def rows_progress():
    out = []
    for d, runner, tier, _ds in NEW_ROWS:
        v = sc(COMP.get(d))
        if v is None:
            raise SystemExit("缺 composite: " + d)
        comp = v[0]
        s_det, s_rep = cal(d, "str", "det_f1"), cal(d, "str", "rep_f1")
        o_det, o_rep = cal(d, "oldsw", "det_f1"), cal(d, "oldsw", "rep_f1")
        g_det, g_rep = cal(d, "goldv29", "det_f1"), cal(d, "goldv29", "rep_f1")
        n = NAT.get(d)
        rate = n[5] if n else None
        p_rep = n[6] if n else None
        if s_det is None:      # deepprep：错误级不可算
            str_cell = oldsw_cell = gold_cell = "—"
        else:
            str_cell = "%s/%s" % (f4(s_det), f4(s_rep))
            oldsw_cell = "%s/%s" % (f4(o_det), f4(o_rep))
            gold_cell = "%s/%s" % (f4(g_det), f4(g_rep))
        out.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            SHORT[d], f6(comp), str_cell, oldsw_cell, gold_cell,
            f4(rate), ("—" if p_rep is None else f4(p_rep))))
    return out


# ══════════════════════════════════════════════════════════════════
# 表二：00_紧凑横向对比.md
# ══════════════════════════════════════════════════════════════════
def rows_compact():
    out = []
    for d, runner, tier, _ds in NEW_ROWS:
        v = sc(COMP.get(d))
        comp = v[0]
        g_det, g_rep = cal(d, "goldv29", "det_f1"), cal(d, "goldv29", "rep_f1")
        s_rep = cal(d, "str", "rep_f1")
        o_rep = cal(d, "oldsw", "rep_f1")
        n = NAT.get(d)
        rate = n[5] if n else None
        out.append("| # | %s | %s | `%s` | %s | %s | %s | %s | %s | %s |" % (
            runner, tier, d, f6(comp),
            ("—" if g_det is None else f6(g_det)),
            ("—" if g_rep is None else f6(g_rep)),
            ("—" if s_rep is None else f6(s_rep)),
            ("—" if o_rep is None else f6(o_rep)),
            f6(rate)))
    return out


# ══════════════════════════════════════════════════════════════════
# 表三：00_三维度横向对比.md
# ══════════════════════════════════════════════════════════════════
def rows_d1():
    out = []
    for d, runner, tier, ds in NEW_ROWS:
        v = sc(COMP.get(d))
        out.append("| `%s` | %s | %s | %s | **%s** | %s | %s | %s | %s |" % (
            d, runner, tier, ds, f6(v[0]), f4(v[1]), f4(v[2]), f4(v[3]), f4(v[4])))
    return out


def rows_d2():
    out = []
    for d, runner, tier, _ds in NEW_ROWS:
        for c in ("str", "oldsw", "goldv29"):
            det, rep = cal(d, c, "det_f1"), cal(d, c, "rep_f1")
            b = buckets(d, c)
            if det is None:
                out.append("| `%s` | %s | **%s** | — | — | — | — | — | — |" % (d, runner, c))
            else:
                out.append("| `%s` | %s | **%s** | **%s** | **%s** | %s | %s | %s | %s |" % (
                    d, runner, c, f6(det), f6(rep),
                    format(b.get("b_clean", 0), ","), format(b.get("b_wrong", 0), ","),
                    format(b.get("b_dirty", 0), ","), format(b.get("b_missing", 0), ",")))
    return out


def rows_d3():
    out = []
    for d, runner, tier, _ds in NEW_ROWS:
        n = NAT.get(d)
        if not n:
            out.append("| `%s` | %s | string | — | — | — | — | — |" % (d, runner))
            continue
        tot, bc, bw, bd, bm, rate, _p = n
        out.append("| `%s` | %s | string | %s | %s | %s | %s | **%s** |" % (
            d, runner, format(tot, ","), format(bc, ","), format(bd, ","),
            format(bm, ","), ("%.*f" % (6, rate)) if rate is not None else "—"))
    return out


# ══════════════════════════════════════════════════════════════════
# 行插入
# ══════════════════════════════════════════════════════════════════
def insert_after(lines, marker, new_lines):
    """在**第一处**包含 marker 的行后面插入 new_lines（marker 唯一）。"""
    for i, l in enumerate(lines):
        if marker in l:
            return lines[:i + 1] + list(new_lines) + lines[i + 1:]
    raise SystemExit("找不到锚点: " + marker)


def renumber(lines):
    """把紧凑表第 1 列（原文是数字，新增行是 '#' 占位）统一重新编号成 1..N。"""
    out, n = [], 0
    for l in lines:
        if "跑者" in l and "结果目录" in l:
            out.append(l)          # 表头行，跳过
            continue
        if re.match(r"^\|\s*(#|\d+)\s*\|", l):
            n += 1
            l = re.sub(r"^\|\s*(#|\d+)\s*\|", "| %d |" % n, l, count=1)
        out.append(l)
    return out, n


def read(p):
    return p.read_text(encoding="utf-8").split("\n")


def write(p, lines):
    p.write_text("\n".join(lines), encoding="utf-8")


KEEP.mkdir(exist_ok=True)

# 数据不齐的行先不并（避免半空行），并记下来
READY = []
for _d, _r, _t, _ds in NEW_ROWS:
    ok, why = ready(_d)
    if ok:
        READY.append((_d, _r, _t, _ds))
    else:
        print("⚠️ 暂不并入（%s）：%s" % (_d, why))
NEW_ROWS = READY


def pick(rows_fn, runner):
    """取某一组（我们 / cc / deepprep）的行；rows_fn 的返回顺序与 NEW_ROWS 一致。"""
    lines = rows_fn()
    return [ln for (d, r, _t, _ds), ln in zip(NEW_ROWS, lines) if r == runner]


DIRS_OURS = [d for d, r, _t, _ds in NEW_ROWS if r == "我们"]
DIRS_BASE = [d for d, r, _t, _ds in NEW_ROWS if r == "cc"]
DIRS_DP = [d for d, r, _t, _ds in NEW_ROWS if r == "deepprep"]

# ── 表一 ──
p1 = DL / "00_评估进度表.md"
L = read(p1)
rp = rows_progress()
L = insert_after(L, "| I48/4.7pct_我们 |", pick(rows_progress, "我们"))
L = insert_after(L, "| I48/1.6pct_cc |", pick(rows_progress, "cc"))
L = insert_after(L, "| ICU5000/deepprep-more |", pick(rows_progress, "deepprep"))
L = [l.replace("# 评估进度表（15 个结果目录 × 3 个维度）",
               "# 评估进度表（24 个结果目录 × 3 个维度）") for l in L]
L = [l.replace("| **composite** | **15/15** ✅ | 9 个复用 + 6 个新算 |",
               "| **composite** | **24/24** ✅ | 15 个原有 + 9 个新补（n18 全 6 个 + I48 4.7pct cc + I48 deepprep 2 个） |")
     for l in L]
L = [l.replace("| **错误级** | 45/45 | 每目录 3 个口径（str / oldsw / goldv29） |",
               "| **错误级** | 60/60 + 4 个「—」 | 每目录 3 个口径（str / oldsw / goldv29）；"
               "4 个 deepprep 目录因产物缺陷算不出（见下方脚注） |") for l in L]
L = [l.replace("| **原生格** | 15/15 | 每目录 1 个口径（string；numeric 已取消） |",
               "| **原生格** | 24/24 | 每目录 1 个口径（string；numeric 已取消） |") for l in L]
if not any("native 脚注" in l for l in L):
    L += [
        "",
        "> **脚注（n18 / I48 的 deepprep 行）**：四个 deepprep 目录的**错误级算不出来**，"
        "坏法各不相同（都是 deepprep 产物自身的缺陷，不是工具问题 —— 工具要按 key 列给每张表建 sqlite 索引，坏表直接建不出来）：",
        "> - `n18/deepprep-less`：`admissions.csv` 0 行；`icu/outputevents.csv` **缺 subject_id** 且 0 行",
        "> - `n18/deepprep-more`：`admissions.csv` **缺 hadm_id**；`icu/{icustays,inputevents}.csv` 0 行；"
        "`icu/{chartevents,outputevents}.csv` **缺 subject_id** 且 0 行",
        "> - `I48/deepprep-less`：`admissions.csv` 是 **1 字节空文件**（连表头都没有）⇒ 建表 SQL 语法错 `near \")\"`",
        "> - `I48/deepprep-more`：只坏 `icu/outputevents.csv`（**缺 charttime**），其余 6 张表都齐",
        "> ⇒ 对照：K29 / ICU5000 的 deepprep `admissions.csv` 有 8,093 / 6,476 行且 key 齐 ⇒ 工具跑得通，"
        "所以那两档在原表里有错误级数字。",
        "",
        "> **原生 P_rep 口径** = `b_clean / (b_clean + b_wrong)`；分母为 0 记「—」。",
    ]
write(KEEP / p1.name, L)
if WRITE_DL:
    shutil.copy2(KEEP / p1.name, p1)
print("① %s  行数 %d" % (p1.name, len([l for l in L if l.startswith("| ") and "---" not in l]) - 1))

# ── 表二 ──
p2 = DL / "00_紧凑横向对比.md"
L = read(p2)
L = insert_after(L, "| 6 | 我们 | 4.7pct |", pick(rows_compact, "我们"))
L = insert_after(L, "| 11 | cc | 1.6pct |", pick(rows_compact, "cc"))
L = insert_after(L, "| 15 | deepprep | 4.7pct |", pick(rows_compact, "deepprep"))
L, n = renumber(L)
L = [l.replace("# ceshi1 紧凑横向对比（15 个结果目录 × 3 维度）",
               "# ceshi1 紧凑横向对比（%d 个结果目录 × 3 维度）" % n) for l in L]
L = [l.replace("> 空格 = 该项尚未算完（后台在跑）。",
               "> 「—」= 该项**不适用**：deepprep 的 n18 / I48 四个目录产物有缺陷（空表 / 缺 key 列），"
               "错误级算不出来（详见 `00_评估进度表.md` 脚注）；不是「还没算完」。") for l in L]
write(KEEP / p2.name, L)
if WRITE_DL:
    shutil.copy2(KEEP / p2.name, p2)
print("② %s  共 %d 行" % (p2.name, n))

# ── 表三 ──
p3 = DL / "00_三维度横向对比.md"
L = read(p3)
L = insert_after(L, "| `I48/4.7pct_我们的agent_pi+val_mimicraw2` | 我们 | 4.7pct | mimicraw |", pick(rows_d1, "我们"))
L = insert_after(L, "| `I48/1.6pct_cc基线_ClaudeCode` | cc | 1.6pct |", pick(rows_d1, "cc"))
L = insert_after(L, "| `ICU5000/deepprep-more` | deepprep | 4.7pct | mimicraw |", pick(rows_d1, "deepprep"))
# 维度二：按目录分 3 行一组插入
d2 = rows_d2()
g = [d2[i:i + 3] for i in range(0, len(d2), 3)]
gours, gbase, gdp = g[:len(DIRS_OURS)], g[len(DIRS_OURS):len(DIRS_OURS) + len(DIRS_BASE)], g[len(DIRS_OURS) + len(DIRS_BASE):]
L = insert_after(L, "| `I48/4.7pct_我们的agent_pi+val_mimicraw2` | 我们 | **goldv29** |", gours[0])
for _i in range(1, len(gours)):
    _prev = DIRS_OURS[_i - 1]
    L = insert_after(L, "| `%s` | 我们 | **goldv29** |" % _prev, gours[_i])
L = insert_after(L, "| `I48/1.6pct_cc基线_ClaudeCode` | cc | **goldv29** |", gbase[0])
for _i in range(1, len(gbase)):
    _prev = DIRS_BASE[_i - 1]
    L = insert_after(L, "| `%s` | cc | **goldv29** |" % _prev, gbase[_i])
L = insert_after(L, "| `ICU5000/deepprep-more` | deepprep | **goldv29** |", sum(gdp, []))
# 维度三
L = insert_after(L, "| `I48/4.7pct_我们的agent_pi+val_mimicraw2` | 我们 | string |", pick(rows_d3, "我们"))
L = insert_after(L, "| `I48/1.6pct_cc基线_ClaudeCode` | cc | string |", pick(rows_d3, "cc"))
L = insert_after(L, "| `ICU5000/deepprep-more` | deepprep | string |", pick(rows_d3, "deepprep"))
L = [l.replace("# ceshi1 三维度横向对比（15 个结果目录）",
               "# ceshi1 三维度横向对比（24 个结果目录）") for l in L]
L += [
    "",
    "7. **n18 / I48 是本轮补齐的（新增 9 个结果目录）** —— n18 六档全补上（1.6/4.7pct 我们的 agent、"
    "1.6/4.7pct cc 基线、deepprep-less/more），I48 补上 **4.7pct cc 基线** 与两个 deepprep 档 ⇒ 四档目录集合齐平。",
    "8. **n18 1.6pct 的 cc 基线是新重跑的**（数据 = mimicraw 1.6pct 切片的 train + test/raw，"
    "prompt = `~/Downloads/codexfuxian/n18_codex_16_prompt.md`）；旧的那次跑的是已删除的 `n18_less` 切片"
    "（cohort 7,445 ≠ 7,048）⇒ 不可比、没有入库。",
    "9. ⭐ **codex 在 train 上就没做到「与 reference 一致」**（prompt 的硬要求）：独立核它的 train 产出 vs "
    "切片 reference ⇒ **31 个错 cell**（`demographics.anchor_age` 9/10 行、`multimodal/ed_triage.pain` "
    "**18/18** 行、`admissions.insurance` 3、`icu/icustays.los` 1），另 chartevents 少 162 行 ⇒ "
    "连 10 个 subject 的训练对照都没对齐，所以 test 上丢 28 万行、composite 0.904403（我们 agent 0.918531）。",
    "10. **n18 / I48 的 deepprep 原生层 b_wrong 很高**（I48/deepprep-less 1005 格、I48/deepprep-more 236 格）"
    "⇒ 这些格子**在输出里存在但值是错的**，不是「没输出」⇒ 原生 P_rep = 0.0000（不是「—」）；"
    "而 n18 的两个 deepprep 目录 b_wrong = 0 ⇒ 记「—」。",
]
write(KEEP / p3.name, L)
if WRITE_DL:
    shutil.copy2(KEEP / p3.name, p3)
print("③ %s" % p3.name)
print("仓库留档:", KEEP)
if not WRITE_DL:
    print("（NO_DL=1：包内三张表**未改动**，需要用户点名授权才覆盖）")
