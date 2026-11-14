"""Kiểm chứng toàn vẹn dataset, chỉ dùng thư viện chuẩn Python (độc lập với numpy, pandas dùng khi xử lý).

Kiểm tra:
  1. SHA-256 của các file theo dõi (dataset/, scripts/, artifacts/dataset/) khớp manifest.
  2. File nguồn: đúng số dòng, lưới 15 phút đều, không trùng mốc.
  3. base_splits: đúng tiêu đề, khoảng thời gian, số dòng; lưới 15 phút liên tục; ba tập nối tiếp, không chồng lấn.
  4. Từng ô so với nguồn, trên chuỗi nối ba tập: số đo hợp lệ (khác NA, trong miền RANGES) có cờ 0 và trùng
     nguyên văn; ô còn lại có cờ 1 và trùng nguyên văn số đo hợp lệ gần nhất phía trước nếu số đo đó cách không quá
     MAX_GAP mốc, ngược lại có cờ -1 và ghi NA.
  5. Số ô cờ 1 và cờ -1 của từng tập khớp FLAG_COUNTS (dataset/LAM_SACH.md mục 4.5).
  6. Số cửa sổ bị bỏ khớp WINDOWS (dataset/README.md mục 5.2, dataset/LAM_SACH.md mục 4.5).
Chạy từ gốc repo: uv run python scripts/verify_dataset.py           (kiểm chứng)
                  uv run python scripts/verify_dataset.py --write   (ghi lại manifest)
"""
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta

from common import BASE, CONFIGS, FEATURES, FLAGS, FMT, MAX_GAP, NAMES, RANGES, RAW, REPO, SOURCE

MANIFEST = os.path.join(REPO, "dataset", "manifest_sha256.json")   # đường dẫn trong manifest tính từ gốc repo
STEP = timedelta(minutes=15)
RAW_ROWS = 631835
EXPECTED = {"train": ("2019-01-01 00:00:00", "2024-05-31 23:45:00", 189888),
            "val": ("2024-06-01 00:00:00", "2025-02-28 23:45:00", 26208),
            "test": ("2025-03-01 00:00:00", "2025-11-30 23:45:00", 26400)}
FLAG_COUNTS = {"train": {1: 2457, -1: 0}, "val": {1: 529, -1: 683}, "test": {1: 363, -1: 396}}
# cấu hình: {tập: (số cửa sổ có thể tạo, số cửa sổ bị bỏ)}, khớp README mục 5.2
WINDOWS = {"24h_6h": {"train": (189769, 0), "val": (26185, 665), "test": (26377, 194)},
           "7d_24h": {"train": (189121, 0), "val": (26113, 1952), "test": (26305, 833)}}
FIXED = ["dataset/raw/cemp_RRI_2007_2025_source.csv", "dataset/metadata/cemp_station_metadata.csv",
         "dataset/metadata/cemp_edi_1177_9_eml.xml", "dataset/README.md", "dataset/LAM_SACH.md"]

errors = []


def check(cond, msg):
    """Ghi lỗi msg nếu cond sai; trả về cond để dùng tiếp trong if."""
    if not cond:
        errors.append(msg)
    return cond


def tracked():
    """Danh sách file manifest phải theo dõi: file cố định, base_splits, mọi file trong scripts/ và artifacts/dataset/."""
    files = FIXED + [f"dataset/base_splits/{n}.csv" for n in NAMES]
    for d in ("scripts", "artifacts/dataset"):
        files += sorted(f"{d}/{f}" for f in os.listdir(os.path.join(REPO, d))
                        if os.path.isfile(os.path.join(REPO, d, f)) and not f.endswith(".pyc"))
    return files


def sha256(path):
    """Mã SHA-256 của một file, đường dẫn tính từ gốc repo."""
    h = hashlib.sha256()
    with open(os.path.join(REPO, path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_split(name):
    """Kiểm tra 3 cho một tập; trả về (thời điểm, giá trị, cờ, datetime) của từng dòng."""
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
    return times, vals, flags, dt


def check_cells(splits, raw):
    """Kiểm tra 4, 5: cài đặt lại quy tắc làm sạch theo từng ô, trên chuỗi nối ba tập theo thời gian."""
    times = [t for s in splits for t in s[0]]
    vals = [v for s in splits for v in s[1]]
    flags = [f for s in splits for f in s[2]]
    for j, c in enumerate(FEATURES):
        lo, hi = RANGES[c]
        last = None                                 # chỉ số dòng của số đo hợp lệ gần nhất
        for i, t in enumerate(times):
            s, x, k = raw[t][SOURCE[c]], vals[i][j], flags[i][j]
            if s != "NA" and lo <= float(s) <= hi:
                check(k == 0 and x == s, f"{t} {c}: observed raw {s}, got {x} flag {k}")
                last = i
            elif last is not None and i - last <= MAX_GAP:
                check(k == 1 and x == vals[last][j], f"{t} {c}: expected ffill {vals[last][j]}, got {x} flag {k}")
            else:
                check(k == -1 and x == "NA", f"{t} {c}: expected NA, got {x} flag {k}")
    for name, s in zip(NAMES, splits):
        fl = [k for f in s[2] for k in f]
        got = {k: fl.count(k) for k in (1, -1)}
        check(got == FLAG_COUNTS[name], f"{name}: flag counts {got} != {FLAG_COUNTS[name]}")
        print(f"[3-5] {name}: {len(s[0])} rows {s[0][0]} .. {s[0][-1]} | flag 1 / -1: {got[1]} / {got[-1]}")


def check_windows(splits):
    """Kiểm tra 6: đếm cửa sổ theo quy tắc của common.window_starts, cài đặt lại độc lập bằng thư viện chuẩn."""
    flags = [fl for s in splits for fl in s[2]]
    miss = [0]
    obs = [0]
    for fl in flags:
        miss.append(miss[-1] + (-1 in fl))
        obs.append(obs[-1] + (fl[0] == 0))
    offs = [0]
    for s in splits:
        offs.append(offs[-1] + len(s[2]))
    for cfg, exp in WINDOWS.items():
        lin, lout = CONFIGS[cfg]
        got = {}
        for si, name in enumerate(NAMES):
            ts = range(max(offs[si], lin), offs[si + 1] - lout + 1)
            drop = sum(1 for t in ts if miss[t + lout] - miss[t - lin] > 0 or obs[t + lout] - obs[t] == 0)
            got[name] = (len(ts), drop)
        check(got == exp, f"windows {cfg}: {got} != {exp}")
        print(f"[6] windows {cfg} (total, dropped): {got}")


def main():
    """Chạy kiểm tra 1 đến 6 theo thứ tự; thoát mã 1 và in tối đa 50 lỗi nếu có lỗi."""
    if "--write" in sys.argv:
        man = {p: {"sha256": sha256(p), "bytes": os.path.getsize(os.path.join(REPO, p))} for p in tracked()}
        json.dump(man, open(MANIFEST, "w"), indent=1)
        print("manifest written")
    man = json.load(open(MANIFEST))
    check(set(man) == set(tracked()), f"manifest file set differs: {sorted(set(man) ^ set(tracked()))}")
    for p, m in man.items():
        check(os.path.exists(os.path.join(REPO, p)) and sha256(p) == m["sha256"], f"sha256 mismatch: {p}")
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
    n_err = len(errors)
    splits = [check_split(n) for n in NAMES]
    for s1, s2 in zip(splits, splits[1:]):
        check(s2[3][0] - s1[3][-1] == STEP, "splits not contiguous")
    if len(errors) == n_err:                    # kiểm tra từng ô cần ba tập đúng cấu trúc
        check_cells(splits, raw)
    check_windows(splits)
    if errors:
        print("FAILED:")
        for e in errors[:50]:
            print("  ", e)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
