"""Kiem chung toan ven dataset (chi dung thu vien chuan Python).

Kiem tra:
  1. SHA-256 cua moi file khop manifest (neu co manifest).
  2. File nguon: dung so dong, luoi 15 phut deu, khong trung, khong mat moc.
  3. Moi file split: dung khoang thoi gian, luoi 15 phut lien tuc, khong trung.
  4. Tung o cua split bang dung chuoi gia tri tuong ung trong file nguon.
  5. Ba split khong chong lan va noi tiep nhau khong ho.
  6. So NA tung cot cua split bang so NA cua file nguon trong cung khoang.
Chay tu thu muc dataset:  python scripts/verify.py           (kiem chung)
                          python scripts/verify.py --write   (ghi lai manifest)
"""
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw", "cemp_RRI_2007_2025_source.csv")
MANIFEST = os.path.join(ROOT, "manifest_sha256.json")
FMT = "%Y-%m-%d %H:%M:%S"
STEP = timedelta(minutes=15)
EXPECTED = {
    "train": ("2019-01-01 00:00:00", "2024-05-31 23:45:00", 189888),
    "val": ("2024-06-01 00:00:00", "2025-02-28 23:45:00", 26208),
    "test": ("2025-03-01 00:00:00", "2025-11-30 23:45:00", 26400),
}
RAW_ROWS = 631835
MAP = {"do_mg_l": "dissolvedoxygen", "water_temp_c": "watertemperature", "ph": "ph",
       "spc_us_cm": "spc", "turbidity_fnu": "turbidity", "chl_fluor_ug_l": "fluorescence"}
TRACKED = ["raw/cemp_RRI_2007_2025_source.csv", "splits/train.csv", "splits/val.csv", "splits/test.csv",
           "metadata/cemp_station_metadata.csv", "metadata/cemp_edi_1177_9_eml.xml",
           "scripts/make_splits.py", "scripts/verify.py", "scripts/describe.py", "README.md",
           "stats/feature_stats.csv", "stats/gap_stats.csv", "stats/long_gaps.csv", "stats/do_dynamics.csv",
           "stats/window_counts.csv", "stats/monthly_dynamics.csv", "stats/split_by_regime.csv",
           "DE_XUAT_TIEN_XU_LY.md", "analysis/cleaning_study.py", "analysis/window_scaler_study.py",
           "analysis/gap_distribution_study.py", "analysis/imputation_benchmark.py", "analysis/threshold_study.py",
           "analysis/gap_histogram_figure.py"]

errors = []


def check(cond, msg):
    if not cond:
        errors.append(msg)
    return cond


def sha256(path):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    write = "--write" in sys.argv
    # 1. manifest
    if write:
        man = {p: {"sha256": sha256(p), "bytes": os.path.getsize(os.path.join(ROOT, p))} for p in TRACKED}
        json.dump(man, open(MANIFEST, "w"), indent=1)
        print("manifest written")
    if os.path.exists(MANIFEST):
        man = json.load(open(MANIFEST))
        for p, m in man.items():
            check(sha256(p) == m["sha256"], f"sha256 mismatch: {p}")
        print(f"[1] sha256: {len(man)} files checked")
    # 2. raw
    raw = {}
    prev = None
    with open(RAW, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            t = datetime.strptime(f"{row['date']} {row['time']}", FMT)
            check(t not in raw, f"raw duplicate {t}")
            if prev is not None:
                check(t - prev == STEP, f"raw step error {prev} -> {t}")
            raw[t] = row
            prev = t
    check(len(raw) == RAW_ROWS, f"raw rows {len(raw)} != {RAW_ROWS}")
    print(f"[2] raw: {len(raw)} rows, {min(raw)} .. {max(raw)}")
    # 3-6. splits
    spans = []
    for name, (a, b, n_exp) in EXPECTED.items():
        rows, prev = [], None
        with open(os.path.join(ROOT, "splits", f"{name}.csv"), newline="") as f:
            rd = csv.DictReader(f)
            check(rd.fieldnames == ["datetime_pst"] + list(MAP), f"{name}: header {rd.fieldnames}")
            for row in rd:
                t = datetime.strptime(row["datetime_pst"], FMT)
                if prev is not None:
                    check(t - prev == STEP, f"{name}: step error {prev} -> {t}")
                prev = t
                rows.append((t, row))
        check(len(rows) == n_exp, f"{name}: rows {len(rows)} != {n_exp}")
        check(rows[0][0] == datetime.strptime(a, FMT), f"{name}: start {rows[0][0]}")
        check(rows[-1][0] == datetime.strptime(b, FMT), f"{name}: end {rows[-1][0]}")
        mism, na_split, na_raw = 0, {c: 0 for c in MAP}, {c: 0 for c in MAP}
        for t, row in rows:
            src = raw.get(t)
            if not check(src is not None, f"{name}: {t} not in raw"):
                continue
            for c, s in MAP.items():
                if row[c] != src[s]:
                    mism += 1
                na_split[c] += row[c] == "NA"
                na_raw[c] += src[s] == "NA"
        check(mism == 0, f"{name}: {mism} cell mismatches vs raw")
        check(na_split == na_raw, f"{name}: NA counts differ")
        spans.append((name, rows[0][0], rows[-1][0]))
        print(f"[3-6] {name}: {len(rows)} rows {rows[0][0]} .. {rows[-1][0]} | cell mismatches {mism} | NA {na_split}")
    for (n1, _, e1), (n2, s2, _) in zip(spans, spans[1:]):
        check(s2 - e1 == STEP, f"gap/overlap between {n1} and {n2}")
    print("[5] splits contiguous, no overlap" if not errors else "")
    if errors:
        print("FAILED:")
        for e in errors[:50]:
            print("  ", e)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
