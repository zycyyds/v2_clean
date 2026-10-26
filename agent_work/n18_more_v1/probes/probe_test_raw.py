import pandas as pd

TEST = "/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18_more/test"
pd.set_option("display.width", 400)

p = pd.read_csv(f"{TEST}/raw/hosp/patients.csv", dtype=str)
print("=== test patients.csv ===")
print(p.head(4).to_string())
print("gender values:", p.gender.value_counts(dropna=False).to_dict())
print("anchor_age values:", p.anchor_age.value_counts(dropna=False).head().to_dict())
print("anchor_year_group values:", p.anchor_year_group.value_counts(dropna=False).to_dict())

adm = pd.read_csv(f"{TEST}/raw/hosp/admissions.csv", dtype=str)
print("\n=== test admissions.csv ===")
print(adm.head(3).to_string())
print("insurance:", adm.insurance.value_counts(dropna=False).to_dict())
print("hospital_expire_flag:", adm.hospital_expire_flag.value_counts(dropna=False).to_dict())
print("deathtime non-null:", adm.deathtime.notna().sum(), "of", len(adm))

ed = pd.read_csv(f"{TEST}/raw/ed/edstays.csv", dtype=str, nrows=5)
print("\n=== test edstays head ===")
print(ed.to_string())
print("intime sample formats:", ed.intime.tolist()[:3])

print("\n=== test ed/medrecon head ===")
m = pd.read_csv(f"{TEST}/raw/ed/medrecon.csv", dtype=str, nrows=4)
print(m.to_string())

print("\n=== test icu/icustays head ===")
i = pd.read_csv(f"{TEST}/raw/icu/icustays.csv", dtype=str, nrows=4)
print(i.to_string())

print("\n=== test icu/chartevents head ===")
c = pd.read_csv(f"{TEST}/raw/icu/chartevents.csv", dtype=str, nrows=4)
print(c.to_string())

print("\n=== test icu/outputevents head ===")
o = pd.read_csv(f"{TEST}/raw/icu/outputevents.csv", dtype=str, nrows=4)
print(o.to_string())

print("\n=== test hosp/labevents head ===")
l = pd.read_csv(f"{TEST}/raw/hosp/labevents.csv", dtype=str, nrows=3)
print(l.to_string())

print("\n=== test note/discharge head ===")
n = pd.read_csv(f"{TEST}/raw/note/discharge.csv", dtype=str, nrows=2)
print(n[["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "storetime"]].to_string())
