#!/usr/bin/env python3
"""生成 n18 1.6pct 的 **codex 基线 prompt**（结构照 k29_codex_16_prompt.md，数字取自日志统计）。

产出：/Users/mac/Downloads/codexfuxian/n18_codex_16_prompt.md
并把条目登记进 /Users/mac/Downloads/codexfuxian/_prompt_manifest.json
"""
import json
from pathlib import Path

STATS = Path("/Users/mac/.claude/jobs/448aad57/tmp/n18_codex_stats.json")
OUT = Path("/Users/mac/Downloads/codexfuxian/n18_codex_16_prompt.md")
MANIFEST = Path("/Users/mac/Downloads/codexfuxian/_prompt_manifest.json")
SLICE = "/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18_codex_16"

st = json.loads(STATS.read_text(encoding="utf-8"))
tr, te = st["train"], st["test"]

TRAIN_IDS = ["11258695", "11437257", "13813219", "13901287", "14675727",
             "14779548", "16934764", "16994397", "17459613", "18976991"]


def types_line(d):
    """按 error type（subtype 优先，缺失则 class）排序列出类型与计数。"""
    c = d.get("by_sub") or d.get("by_cls") or {}
    items = sorted(c.items(), key=lambda kv: -kv[1])
    return "、".join("%s×%s" % (k, format(v, ",")) for k, v in items), len(items)


def ops_line(d):
    c = d.get("by_op") or {}
    return "、".join("%s %s" % (format(v, ","), k) for k, v in sorted(c.items(), key=lambda kv: -kv[1]))


tr_types, tr_ntypes = types_line(tr)
te_types, te_ntypes = types_line(te)

# 每个 subject 的平均错误数
tr_avg = tr["entries"] / 10.0
te_avg = te["entries"] / 2970.0

# train / test 的 top 类型（用于「分布差很多」那句）
tr_top = sorted((tr.get("by_sub") or tr.get("by_cls") or {}).items(), key=lambda kv: -kv[1])[0]
te_top = sorted((te.get("by_sub") or te.get("by_cls") or {}).items(), key=lambda kv: -kv[1])[0]
if tr_top[0] == te_top[0]:
    TOP_NOTE = ("**注意 test 的错误类型分布和 train 差很多**（两档的 top 类型都是 `%s`，"
                "但占比从 train 的 %.1f%% 变成 test 的 %.1f%%）—— 判定规则要从 train 推导，"
                "但覆盖面要按 test 的分布来准备"
                % (tr_top[0], 100.0 * tr_top[1] / tr["entries"], 100.0 * te_top[1] / te["entries"]))
else:
    TOP_NOTE = ("**注意 test 的错误类型分布和 train 差很多**（train 的 top 类型是 %s，"
                "test 的 top 类型是 %s）—— 判定规则要从 train 推导，但覆盖面要按 test 的分布来准备"
                % (tr_top[0], te_top[0]))

PROMPT = f"""你是 Data Cleaning Agent。请完成一个基于少量标准示例的 MIMIC-IV 数据清洗任务，最终输出一个名为 N18 的疾病 cohort 结果包（N18 = ICD-10 慢性肾病的章节，对应 N18.0 ~ N18.9），围绕该 cohort 做按患者（subject_id）切片与事件级清洗。

    训练输入与目标输入的路径由用户在启动 codex 之前 export 的环境变量提供（codex 不会自动注入路径）：

    {{TRAIN_RAW}} —— 含脏数据的 MIMIC-IV 风格 hosp/、icu/、ed/ 和 note/ 原始表（已按 subject_id 切片到 10 个 subject 的子集）
    {{TRAIN_REFERENCE}} —— 同一批 10 个 subject_id 的干净标准结果包（agent 应产出）。codex 必须对照这个目录理解每条规则的预期输出
    {{TEST_RAW}} —— codex 唯一的目标处理数据集。codex 读完 train 对照、推导完规则后，直接在 test/raw 上跑 pipeline 并产出最终 18 个 csv 结果包和 submission.json
    注意：codex 没有 val/raw、没有 harness、没有多轮反馈机制。请不要假设存在 Validation raw；也不要试图读取 pi_test_scorer.py / evaluation_manifest.json 之类的评分元数据。

    train/raw 是按 subject_id 切片的（不是 hadm_id 也不是 stay_id），keys.csv 的列是 subject_id；同一个 subject 可能对应多个 hadm_id，所以 cohort.csv 的行数会多于 keys.csv 的行数。

    本数据集的 10 个 train subject_id（覆盖 train 全集，无 held-out 训练对照）：
    {", ".join(TRAIN_IDS)}

    本轮的核心不是只生成同名文件，而是从 10 个 train 示例推导可重放的转换规则。开始时请直接读取 {{TRAIN_RAW}} / {{TRAIN_REFERENCE}} / {{TEST_RAW}} 三个路径。可以枚举其共同的 train/ 父目录，但不得读取任何其他数据集或实验目录。

    N18 任务定义
    cohort 选择：cohort.csv 列出本次入选的住院，含 `cohort_type` / `cohort_label` / `icd_root` 三列，**入选依据需要你从 train 对照里推导出来**（哪些 hadm 入选、哪些没入选，以及为什么）。hospital_expire_flag 作为 in-hospital mortality 标签。
    ⚠️ 本任务的输出表是**cohort 筛选之后**的表，不是全量表：先修好 raw 的错误，再按 N18 入选规则筛出 cohort，最后才生成这些输出表。

    必须输出的 18 个 csv 文件：

    7 个顶层：admissions.csv, cohort.csv, demographics.csv, diagnoses.csv, procedures.csv, medications.csv, labs.csv
    4 个 ICU：icu/icustays.csv, icu/chartevents.csv, icu/inputevents.csv, icu/outputevents.csv
    5 个 ED 聚合：multimodal/ed_diagnosis.csv, multimodal/ed_medrecon.csv, multimodal/ed_pyxis.csv, multimodal/ed_triage.csv, multimodal/ed_vitalsign.csv
    1 个 ED 主表：multimodal/edstays.csv
    1 个文本：multimodal/notes.csv

    数据特点（v2 rawerror 关键区别）
    train/raw 和 test/raw 是注入了错误的 MIMIC-IV 原始数据：`hosp/` 和 `icu/` 子目录里包含多种类型的 cell-level 错误（值越界、enum 违例、时间序颠倒、跨表不一致等）；`ed/` 和 `note/` 多模态原始文件是干净原样数据（因为没有这些子目录的注入规则），但它们的内容会参与 cohort 判定、并被聚合成上面那 6 个 ED/文本输出文件 —— 不要跳过它们。
    raw 下共 34 张表，但**错误高度集中在少数几张**（注入只发生在 7 张 raw 表上）。请自己逐表比对确认哪些表是干净的，不要对每个表都套一遍修正逻辑。

    具体有哪些错误子类型、有多少错误、如何修正——这些都需要你通过对照 {{TRAIN_RAW}} 与 {{TRAIN_REFERENCE}} 自己推导。常见错误模式包括但不限于：值越界 / 枚举违例 / 必填字段缺失 / 时间序颠倒 / 哨兵值（sentinel）/ 量纲错误 / 整行重复 / 跨表孤儿 ID / 跨表 ID 错位 / 标签泄漏 / 任务窗口泄漏 / 代码别名污染 等等。
    另外还有一类**不是注入器造成的**错误：**MIMIC 自己就录错的值**（生理不可能区间、哨兵占位码）——它们只出现在 `icu/chartevents.csv` 的 `valuenum` 列上（类型名带 `mimic_native` 前缀）；train 对照里有注入的同形态示例可供你学这条规则。

    错误注入量级（供你估算工作量，不是硬约束）：

    train: 约 {tr['entries']:,} 条错误，分布在 {tr['stays']} 个 subject 上，涉及 {tr['files']} 张 raw 表（相关 cell 共 {tr['cells']:,} 个；这 {tr['files']} 张表合计 {tr['total_cells']:,} 个非 key cell ⇒ cell-level {100.0 * tr['cells'] / tr['total_cells']:.2f}%）
        按操作类型：{ops_line(tr)}
        全部 {tr_ntypes} 个错误类型：{tr_types}
    test: 约 {te['entries']:,} 条错误，分布在 {te['stays']:,} 个 subject 上，涉及 {te['files']} 张 raw 表（相关 cell 共 {te['cells']:,} 个；这 {te['files']} 张表合计 {te['total_cells']:,} 个非 key cell ⇒ cell-level {100.0 * te['cells'] / te['total_cells']:.2f}%）
        按操作类型：{ops_line(te)}
        全部 {te_ntypes} 个错误类型：{te_types}
    注意：train 只有 {tr['stays']} 个 subject、{tr['entries']:,} 条错误，test 有 {te['entries']:,} 条、覆盖 {te['stays']:,} 个 subject
    **train 里每一类错误的样本都很少** —— 必须靠领域常识补齐判定规则，不要因为某类错误在 train 里只出现一两次就判定它不重要，也不要因为它在 train 里出现得多就以为 test 里也占多数
    {TOP_NOTE}
    test 的错误量级大（{te['entries']:,} 条），平均每个 subject {te_avg:.0f} 条；**绝大多数 subject 并非完全干净**，不要假设大部分表是原样的

    推导与修正流程
    推荐流程（具体细节由你按 train 对照决定）：

    加载 raw → 解析所有 hosp/icu/ed/note/*.csv（ed/ 与 note/ 不需要修，原样处理）
    对照 train/raw 与 train/reference → 系统性比对每个文件的每行每列：
    哪些 cell 值与 reference 不一致？
    不一致是否构成有意义的"错误"（不是 schema 差异，不是过滤差异，不是聚合差异）？
    这些错误能否归类为已知模式（值越界 / 枚举违例 / 时间序颠倒 / 哨兵值 / 跨表不一致 ...）？
    建立基线表（如 valid gender/age 范围、icustays 三元组等）
    对每个发现的错误模式写检测 + 修复逻辑（按你的实现风格；建议先小后大、先 train 自检通过再处理 test）
    推断 N18 cohort 的入选规则，并在 train 上验证（入选集合必须与 train/reference 的 cohort.csv 一致）
    生成与 reference/ 相同的 18 个 csv

    重要约束：

    不要硬编码任何 subject_id / hadm_id / stay_id / note_id / 行号或标准答案
    不要用重复行、截断行、填充空值的方式拟合 test 数据的规模
    不能读取 test Gold / host_private / pi_test_scorer.py / evaluation_manifest.json
    如果发现无法判定的错误（同一 cell 命中多个子类），优先按 reference 同一 (stay_id, itemid, charttime) 行反推正确值
    最终 replay 只能依赖 {{raw_root}}、{{train_reference}}、{{output_dir}} 和可选 {{workdir}}；不得使用 {{train_raw}}
    必须完成的工作：

    比较 train/raw 与 train/reference 的目录、文件、schema、主键、实体关系、时间字段、行级变化和聚合结果。
    在 Agent 工作目录写出 train_analysis.json。对每个 18 个目标 csv 文件记录：对应 raw 输入表、主键、时间锚点、过滤条件、字段转换、聚合规则，以及你从 train 对照中推导出来的错误模式清单（每条至少配一项 train 对照证据）。
    基于这些规则实现可执行 pipeline。它必须从 raw/hosp、raw/icu、raw/ed、raw/note 开始，不能将已有结果包当作 test 输出输入。
    实际在 train/raw 上运行该 pipeline（包括错误修正步骤与 cohort 筛选），并将输出与 train/reference 比较，记录可解释的 train 自检结果（train 自检必须通过：与 reference 一致）。
    在 test/raw 上运行同一 pipeline，生成完整 18 个 csv 结果包和 submission.json。
    输出与复现要求：

    保持 train/reference 的相对目录结构、18 个 csv 文件名、schema、数据类型、主键语义和聚合语义
    summary.json 不参与行对齐评分，但若 train/reference 里有它则建议保留同位置同 schema 的输出（不强制）
    不得硬编码训练或 test 的 subject_id、hadm_id、stay_id、note_id、行号或标准答案
    不得用重复行、截断行或填充空值的方式仅为了匹配 test 数据规模
    保留分析脚本、最终 pipeline、规则/映射、运行入口和 train_analysis.json
    submission.json 必须包含 schema_version=1、相对 result_root，以及 JSON 参数数组形式的 replay.argv
    最终 replay 只能依赖 {{raw_root}}、{{train_reference}}、{{output_dir}} 和可选 {{workdir}}；不得使用 {{train_raw}}
    因为 codex 没有 harness 评分，请在交付前自己至少做一次完整的 train 自检
    限制：

    不使用网络
    不读取、搜索或推断 test Gold、其他实验目录、历史 pipeline、其他工作树或项目外数据（也不读取 pi_test_scorer.py / evaluation_manifest.json 之类的评分元数据）
    已注册的 Skills 可按需读取，但它们不包含隐藏数据或隐藏答案
    证据不足时采用保守、可解释、由 train 对照支持的规则；先保证训练对照、train 自检通过，再追求 test 正确性

    {{TRAIN_RAW}}        = {SLICE}/train/raw
    {{TRAIN_REFERENCE}}  = {SLICE}/train/reference
    {{TEST_RAW}}         = {SLICE}/test/raw

    注意：上面三个路径所在的 n18_codex_16/ 目录已经剔除了 modification_log.csv、mimic_native_error_log.csv 和 test/reference。
    这几个文件不在数据集里，不要去找，也不要去父目录里找同名文件。
"""

OUT.write_text(PROMPT, encoding="utf-8")
print("写出 %s（%d 字符）" % (OUT, len(PROMPT)))

man = json.loads(MANIFEST.read_text(encoding="utf-8"))
man["n18_16"] = {"file": str(OUT), "chars": len(PROMPT), "train_ids": 10}
MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("更新 _prompt_manifest.json：", list(man))
