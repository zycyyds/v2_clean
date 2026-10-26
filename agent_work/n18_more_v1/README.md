# N18 (CKD) MIMIC-IV cohort cleaning — v2 rawerror, `n18_more`

Data-cleaning agent for the `n18_more` slice of the **N18** (ICD-10 chronic kidney
disease) cohort task.  The slice ships an **error-injected** MIMIC-IV style tree
(`hosp/`, `icu/`, `ed/`, `note/`); the job is to repair the injected errors, select
the N18 cohort and emit the 18 file result package.

```
prompts/最新注入错误prompt/n18_4.7pct_codex_prompt.md   <- the task statement
datasets/注入错误数据集/n18_more/
    train/raw        train/reference        <- the 10 subject training pair
    test/raw         test/keys.csv          <- the 2970 subject test slice
```

## Layout

```
agent_work/n18_more_v1/
├── README.md                    this file
├── train_analysis.json          the required analysis document: 19 error
│                                patterns (each with train evidence) + the rule
│                                set of the 18 output files
├── submission.json              schema_version 1, relative result_root, replay.argv
├── train_self_check.py          runs the pipeline on train/raw and diffs it
│                                against train/reference
├── analysis/
│   ├── train_analysis.json      (copy of the document above)
│   ├── train_selfcheck.json     the self check report
│   ├── sweep_train.json         the raw-vs-reference cell diff of every output
│   ├── test_package_validation.json
│   └── test_sanity.json
├── pipeline/
│   ├── n18_clean.py             the pipeline (repairs + cohort + 18 csv files)
│   ├── run_pipeline.py          replay entry point
│   ├── validate_package.py      schema/row validation of a produced package
│   └── sanity_check.py          checks that no corruption signature survived
├── probes/                      the analysis scripts used to derive the rules
├── output/                      the produced result package (18 csv files)
└── work/                        scratch directory of the last run
```

## Replay

The replay only reads `{raw_root}`, `{train_reference}` and writes `{output_dir}`
plus the optional `{workdir}`; it never reads the training raw data, the test gold
or any evaluation metadata.

```bash
python3 pipeline/run_pipeline.py \
    --raw-root        <raw_root> \
    --train-reference <train_reference> \
    --output-dir      <output_dir> \
    --workdir         <workdir> \
    --submission      <submission.json>
```

The same command produced the package in `output/` (test slice, 80 s) and the
train self check (`python3 train_self_check.py`).

## Rules (short version, details in `train_analysis.json`)

* **cohort** — a `hadm_id` joins the cohort when it carries an ICD-10 `N18%` or an
  ICD-9 `585%` diagnosis *and* appears in `ed/edstays.csv` (the required ED
  modality).  18/18 train hadm_id, 7048 test hadm_id.
* **alias** — `diagnoses_icd.icd_code` may carry an `ALIAS_` prefix (train only).
* **mortality label** — `hospital_expire_flag` was inverted, it is restored as
  `1 - raw` (identical to "1 iff deathtime is not null" on every train row).
* **gender** — restored from `ed/edstays.gender`; **anchor_age** is a sentinel
  everywhere and is not recoverable (constant fallback, documented).
* **insurance** — nulled everywhere, not recoverable (constant fallback).
* **ICU outtime** — restored as `intime + los * 86400 s` (the raw holds
  `intime - 1h`).
* **chart events** — the label leakage rows (`itemid 999001`) are dropped, the
  `99`-prefixed itemids are stripped, the rows that were pushed out of the ICU
  task window (`charttime = outtime + 1 day`) get their charttime back from the
  neighbouring row of the same stay, `valuenum` is rebuilt from the textual
  `value` column (sentinels, 100x multiples and arbitrary numbers) and from the
  dominant value of the (itemid, value) group, `valueuom` from the dominant unit
  of the itemid, `subject_id`/`hadm_id` from the ICU stay registry, and the
  injected whole row copies are de-duplicated.
* **input events** — `hadm_id` from the stay registry, `amount` from
  `originalamount`, `endtime = starttime + 1 min` (the raw holds `starttime - 1 min`).
* **output events** — the epoch `charttime` is rebuilt on the hour grid of
  `storetime`; the output events of one (stay, caregiver) are strictly increasing,
  which disambiguates the grid point.
* **formatting** — the reference writes every numeric column as a float
  (`97.0`), the ED timestamps as `%Y-%m-%d %H:%M:%S` and the ED identifiers
  without their zero padding.

## Train self check

`python3 train_self_check.py` re-runs the whole pipeline on `train/raw`
(including the repair steps) and compares it with `train/reference`:

```
file                                 rows         positional             merged  rowF1
admissions.csv                        18/18         0.9872 (231/234)   0.9848 (195/198)  1.0000
cohort.csv                            18/18         1.0000 (108/108)   1.0000 (72/72)  1.0000
demographics.csv                      10/10         0.8333 (50/60)     0.8000 (40/50)   1.0000
diagnoses.csv                       289/289         1.0000 (1445/1445) 1.0000 (578/578)  1.0000
procedures.csv                        32/32         1.0000 (160/160)   1.0000 (64/64)    1.0000
medications.csv                      786/786         1.0000 (6288/6288) 0.9544 (3604/3776) 1.0000
labs.csv                            4073/4073         1.0000 (24438/24438) 0.9995 (8158/8162) 1.0000
icu/icustays.csv                       3/3          0.9444 (17/18)     0.9333 (14/15)   1.0000
icu/chartevents.csv                3740/4013         0.5441            0.9893           0.9568
icu/inputevents.csv                   28/28         0.9541 (187/196)   0.9196 (103/112)  1.0000
icu/outputevents.csv                 56/56         0.9137 (307/336)   0.9570 (89/93)   0.5536
multimodal/notes.csv                  17/17         1.0000 (136/136)   1.0000 (119/119)  1.0000
multimodal/ed_diagnosis.csv           30/30         1.0000 (180/180)   1.0000 (120/120)  1.0000
multimodal/ed_medrecon.csv          242/242         1.0000 (2178/2178) 0.9186 (1286/1400) 1.0000
multimodal/ed_pyxis.csv               87/87         1.0000 (609/609)   1.0000 (261/261)  1.0000
multimodal/ed_triage.csv              18/18         1.0000 (198/198)   1.0000 (180/180)  1.0000
multimodal/ed_vitalsign.csv           89/89         1.0000 (979/979)   1.0000 (801/801)  1.0000
multimodal/edstays.csv                18/18         1.0000 (162/162)   1.0000 (144/144)  1.0000

OVERALL positional cell accuracy: 0.8118
OVERALL merged     cell accuracy: 0.9831
```

`positional` compares the produced rows with the reference rows in file order,
`merged` joins them on the natural key of the table.  The positional score of
`icu/chartevents.csv` is low because the reference contains 296 chart events whose
original charttime the corrupter destroyed (the key aligned score, 0.9893, is the
meaningful one).

## Known gaps

* `demographics.anchor_age` — the sentinel is not recoverable from any other raw
  table (the notes redact the date of birth), a constant is written.
* `admissions.insurance` — the column is null in the whole raw slice, a constant
  is written.
* `icu/chartevents` — 296 of the 4013 train rows (7.4%) were moved out of the ICU
  task window *and* had their storetime rewritten, so their original charttime
  cannot be restored; the rows are kept (their itemid/value are still correct).
* `icu/outputevents.charttime` — the lag between the grid point and the storetime
  is recovered statistically, not exactly (48% of the train rows are exact).
* `icu/inputevents` — the duration of the long infusions is not encoded in the
  raw.
