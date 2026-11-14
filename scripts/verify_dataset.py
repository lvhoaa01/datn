"""Kiểm chứng toàn vẹn dataset, chỉ dùng thư viện chuẩn Python (độc lập với numpy, pandas dùng khi xử lý).

Kiểm tra:
  1. SHA-256 của các file theo dõi (dataset/, scripts/, artifacts/dataset/) khớp manifest.
  2. File nguồn: đúng số dòng, lưới 15 phút đều, không trùng mốc.
  3. base_splits: đúng tiêu đề, khoảng thời gian, số dòng; lưới 15 phút liên tục; ba tập nối tiếp, không chồng lấn.
  4. Từng ô so với nguồn: cờ 0 trùng nguyên văn; cờ 1 là ô NA trong nguồn; cờ 2 là giá trị ngoài miền
     hoặc điểm DO bất thường đã biết; cờ 3 để NA.
  5. Mỗi đoạn cờ 1 và 2 dài không quá 32 mốc và bằng nội suy tuyến tính giữa hai số đo kề bên (sai số làm tròn 1e-4);
     mỗi đoạn cờ 3 dài hơn 32 mốc hoặc chạm đầu, cuối tập.
  6. Số ô theo cờ và số cửa sổ bị bỏ khớp số liệu trong dataset/README.md và dataset/LAM_SACH.md.
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
SPIKES = {"2019-08-19 00:00:00", "2021-09-08 22:15:00", "2021-10-28 17:30:00", "2023-07-11 08:45:00"}
FLAG_COUNTS = {"train": {1: 2457, 2: 4, 3: 0}, "val": {1: 305, 2: 0, 3: 907}, "test": {1: 171, 2: 0, 3: 588}}
# cấu hình: {tập: (số cửa sổ có thể tạo, số cửa sổ bị bỏ)}, khớp README mục 5.2
WINDOWS = {"24h_6h": {"train": (189769, 0), "val": (26185, 720), "test": (26377, 217)},
           "7d_24h": {"train": (189121, 0), "val": (26113, 2016), "test": (26305, 865)}}
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


def runs(mask):
    """Các đoạn True liên tiếp: list (bắt đầu, độ dài)."""
    out, a = [], None
    for i, m in enumerate(mask + [False]):
        if m and a is None:
            a = i
        elif not m and a is not None:
            out.append((a, i - a))
            a = None
    return out


def check_split(name, raw):
    """Kiểm tra 3, 4, 5 cho một tập; trả về (danh sách thời điểm, cờ từng dòng) cho kiểm tra 6."""
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
    # 4. từng ô so với nguồn
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
    # 5. đoạn điền và đoạn trống dài
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
    """Kiểm tra 6: đếm cửa sổ theo quy tắc của common.window_starts, cài đặt lại độc lập bằng thư viện chuẩn."""
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
