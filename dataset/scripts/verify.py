"""Kiem chung toan ven dataset (chi dung thu vien chuan Python).

Kiem tra:
  1. SHA-256 cua cac file theo doi khop manifest.
  2. File nguon: dung so dong, luoi 15 phut deu, khong trung moc.
  3. base_splits: dung tieu de, khoang thoi gian, so dong; luoi 15 phut lien tuc; ba tap noi tiep, khong chong lan.
  4. Tung o so voi nguon: co 0 trung nguyen van; co 1 la o NA trong nguon; co 2 la gia tri ngoai mien
     hoac diem DO bat thuong da biet; co 3 de NA.
  5. Moi doan co 1 va 2 dai khong qua 32 moc va bang noi suy tuyen tinh giua hai so do ke ben (sai so lam tron 1e-4);
     moi doan co 3 dai hon 32 moc hoac cham dau, cuoi tap.
  6. So o theo co va so cua so bi bo khop so lieu trong README.md va LAM_SACH.md.
Chay tu thu muc dataset: python scripts/verify.py           (kiem chung)
                         python scripts/verify.py --write   (ghi lai manifest)
"""
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta

from common import BASE, CONFIGS, FEATURES, FLAGS, FMT, MAX_GAP, NAMES, RANGES, RAW, ROOT, SOURCE

MANIFEST = os.path.join(ROOT, "manifest_sha256.json")
STEP = timedelta(minutes=15)
RAW_ROWS = 631835
EXPECTED = {"train": ("2019-01-01 00:00:00", "2024-05-31 23:45:00", 189888),
            "val": ("2024-06-01 00:00:00", "2025-02-28 23:45:00", 26208),
            "test": ("2025-03-01 00:00:00", "2025-11-30 23:45:00", 26400)}
SPIKES = {"2019-08-19 00:00:00", "2021-09-08 22:15:00", "2021-10-28 17:30:00", "2023-07-11 08:45:00"}
FLAG_COUNTS = {"train": {1: 2457, 2: 4, 3: 0}, "val": {1: 305, 2: 0, 3: 907}, "test": {1: 171, 2: 0, 3: 588}}
# cau hinh -> {tap: (so cua so co the tao, so cua so bi bo)}
WINDOWS = {"24h_6h": {"train": (189769, 0), "val": (26185, 720), "test": (26377, 217)},
           "7d_24h": {"train": (189121, 0), "val": (26113, 2016), "test": (26305, 865)}}
FIXED = ["raw/cemp_RRI_2007_2025_source.csv", "metadata/cemp_station_metadata.csv",
         "metadata/cemp_edi_1177_9_eml.xml", "README.md", "LAM_SACH.md"]

errors = []


def check(cond, msg):
    if not cond:
        errors.append(msg)
    return cond


def tracked():
    files = FIXED + [f"base_splits/{n}.csv" for n in NAMES]
    for d in ("scripts", "analysis", "analysis/results"):
        files += sorted(f"{d}/{f}" for f in os.listdir(os.path.join(ROOT, d))
                        if os.path.isfile(os.path.join(ROOT, d, f)) and not f.endswith(".pyc"))
    return files


def sha256(path):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def runs(mask):
    """Cac doan True lien tiep: list (bat dau, do dai)."""
    out, a = [], None
    for i, m in enumerate(mask + [False]):
        if m and a is None:
            a = i
        elif not m and a is not None:
            out.append((a, i - a))
            a = None
    return out


def check_split(name, raw):
    a_exp, b_exp, n_exp = EXPECTED[name]
    with open(os.path.join(BASE, f"{name}.csv"), newline="") as f:
        rd = csv.reader(f)
        check(next(rd) == ["datetime_pst"] + FEATURES + FLAGS, f"{name}: header")
        rows = list(rd)
    times = [r[0] for r in rows]
    vals = [r[1:7] for r in rows]
    flags = [[int(x) for x in r[7:13]] for r in rows]
    check(len(rows) == n_exp, f"{name}: rows {len(rows)} != {n_exp}")
    check(times[0] == a_exp and times[-1] == b_exp, f"{name}: span {times[0]} .. {times[-1]}")
    dt = [datetime.strptime(t, FMT) for t in times]
    check(all(y - x == STEP for x, y in zip(dt, dt[1:])), f"{name}: 15-min grid broken")
    # 4. tung o so voi nguon
    count = {k: 0 for k in range(4)}
    for t, v, fl in zip(times, vals, flags):
        src = raw.get(t)
        if not check(src is not None, f"{name}: {t} not in raw"):
            continue
        for j, c in enumerate(FEATURES):
            k, s, x = fl[j], src[SOURCE[c]], v[j]
            count[k] += 1
            if k == 0:
                check(x == s, f"{name} {t} {c}: {x} != raw {s}")
            elif k == 1:
                check(s == "NA" and x != "NA", f"{name} {t} {c}: flag 1 but raw {s}, value {x}")
            elif k == 2:
                lo, hi = RANGES[c]
                known = (s != "NA" and not lo <= float(s) <= hi) or (c == "do_mg_l" and name == "train" and t in SPIKES)
                check(known and x != "NA", f"{name} {t} {c}: unexpected flag 2")
            else:
                check(k == 3 and x == "NA", f"{name} {t} {c}: flag {k}, value {x}")
    spikes = {t for t, fl in zip(times, flags) if fl[0] == 2}
    check(spikes == (SPIKES if name == "train" else set()), f"{name}: DO flag-2 points {sorted(spikes)}")
    check({k: count[k] for k in (1, 2, 3)} == FLAG_COUNTS[name], f"{name}: flag counts {count}")
    # 5. doan dien va doan trong dai
    n = len(rows)
    for j, c in enumerate(FEATURES):
        for a, ln in runs([fl[j] in (1, 2) for fl in flags]):
            if not check(ln <= MAX_GAP and a > 0 and a + ln < n and flags[a - 1][j] == 0 and flags[a + ln][j] == 0,
                         f"{name} {c} {times[a]}: filled run of {ln} without two observed ends"):
                continue
            y0, y1 = float(vals[a - 1][j]), float(vals[a + ln][j])
            for i in range(a, a + ln):
                lin = y0 + (y1 - y0) * (i - a + 1) / (ln + 1)
                check(abs(float(vals[i][j]) - lin) <= 1e-4, f"{name} {c} {times[i]}: not linear")
        for a, ln in runs([fl[j] == 3 for fl in flags]):
            check(ln > MAX_GAP or a == 0 or a + ln == n, f"{name} {c} {times[a]}: short run flagged 3")
    print(f"[3-5] {name}: {n} rows {times[0]} .. {times[-1]} | flag 1/2/3: {count[1]}/{count[2]}/{count[3]}")
    return dt, flags


def check_windows(splits):
    """Dem cua so theo quy tac cua common.window_starts, cai dat doc lap bang thu vien chuan."""
    flags = [fl for _, f in splits for fl in f]
    long_ = [0]
    obs = [0]
    for fl in flags:
        long_.append(long_[-1] + (3 in fl))
        obs.append(obs[-1] + (fl[0] == 0))
    offs = [0]
    for _, f in splits:
        offs.append(offs[-1] + len(f))
    for cfg, exp in WINDOWS.items():
        lin, lout = CONFIGS[cfg]
        got = {}
        for si, name in enumerate(NAMES):
            ts = range(max(offs[si], lin), offs[si + 1] - lout + 1)
            drop = sum(1 for t in ts if long_[t + lout] - long_[t - lin] > 0 or obs[t + lout] - obs[t] == 0)
            got[name] = (len(ts), drop)
        check(got == exp, f"windows {cfg}: {got} != {exp}")
        print(f"[6] windows {cfg} (total, dropped): {got}")


def main():
    if "--write" in sys.argv:
        man = {p: {"sha256": sha256(p), "bytes": os.path.getsize(os.path.join(ROOT, p))} for p in tracked()}
        json.dump(man, open(MANIFEST, "w"), indent=1)
        print("manifest written")
    man = json.load(open(MANIFEST))
    check(set(man) == set(tracked()), f"manifest file set differs: {sorted(set(man) ^ set(tracked()))}")
    for p, m in man.items():
        check(os.path.exists(os.path.join(ROOT, p)) and sha256(p) == m["sha256"], f"sha256 mismatch: {p}")
    print(f"[1] sha256: {len(man)} files checked")
    raw, prev = {}, None
    with open(RAW, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            t = f"{row['date']} {row['time']}"
            check(t not in raw, f"raw duplicate {t}")
            dt = datetime.strptime(t, FMT)
            if prev is not None:
                check(dt - prev == STEP, f"raw step error {prev} -> {dt}")
            raw[t], prev = row, dt
    check(len(raw) == RAW_ROWS, f"raw rows {len(raw)} != {RAW_ROWS}")
    print(f"[2] raw: {len(raw)} rows")
    splits = [check_split(n, raw) for n in NAMES]
    for (d1, _), (d2, _) in zip(splits, splits[1:]):
        check(d2[0] - d1[-1] == STEP, "splits not contiguous")
    check_windows(splits)
    if errors:
        print("FAILED:")
        for e in errors[:50]:
            print("  ", e)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
