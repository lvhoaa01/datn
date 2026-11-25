"""Kiểm chứng toàn vẹn dataset, chỉ dùng thư viện chuẩn Python (độc lập với numpy, pandas dùng khi xử lý).

Kiểm tra:
  1. SHA-256 của các file theo dõi (dataset/, ba tài liệu trong docs/, scripts/, artifacts/dataset/) khớp manifest.
  2. File nguồn: kích thước và MD5 khớp metadata EDI (cemp_edi_1177_9_eml.xml), đúng số dòng, lưới 15 phút đều,
     không trùng mốc.
  3. base_splits: đúng tiêu đề, khoảng thời gian, số dòng; lưới 15 phút liên tục; ba tập nối tiếp, không chồng lấn.
  4. Từng ô so với nguồn, trên chuỗi nối ba tập: số đo hợp lệ (khác NA, trong miền RANGES) có cờ 0 và trùng
     nguyên văn; ô còn lại có cờ 1 và trùng nguyên văn số đo hợp lệ gần nhất phía trước (phải có số đo như vậy).
  5. Số ô cờ 1 của từng tập khớp FLAG_COUNTS (docs/02_DATA_CLEANING.md mục 4.3).
  6. Số cửa sổ bị bỏ khớp WINDOWS (docs/04_MODEL_TRAINING_EVALUATION.md mục 2, docs/02_DATA_CLEANING.md mục 4.3).
  7. base_domain_splits: hồi quy giữa các đêm train tính lại cho đúng k trong artifacts/dataset/reaeration_k.csv;
     13 cột đầu trùng nguyên văn base_splits; 24 đặc trưng miền tính lại độc lập (định nghĩa của
     common.domain_features viết lại bằng math) lệch không quá 2e-6.
Chạy từ gốc repo: uv run python scripts/verify_dataset.py           (kiểm chứng)
                  uv run python scripts/verify_dataset.py --write   (ghi lại manifest)
"""
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timedelta

from common import (BASE, CHL_FLOOR, CONFIGS, DAY, DOMAIN, DOMAIN_FEATURES, FEATURES, FLAGS, FMT, LAT, LON, M2_EPOCH,
                    M2_HOURS, NAMES, NIGHT_MIN_STEPS, RANGES, RAW, REAERATION, REPO, SOURCE, SPLITS, read_k)

MANIFEST = os.path.join(REPO, "dataset", "manifest_sha256.json")   # đường dẫn trong manifest tính từ gốc repo
STEP = timedelta(minutes=15)
RAW_ROWS = 631735
# kích thước (byte) và MD5 của file MSD ghi trong metadata EDI edi.1177.9
RAW_BYTES, RAW_MD5 = 33877088, "9b3482d7f375014273bac0088a8d95d8"
EXPECTED = {"train": ("2019-01-01 00:00:00", "2024-05-31 23:45:00", 189888),
            "val": ("2024-06-01 00:00:00", "2025-02-28 23:45:00", 26208),
            "test": ("2025-03-01 00:00:00", "2025-11-30 23:45:00", 26400)}
FLAG_COUNTS = {"train": 4098, "val": 212, "test": 107}
# cấu hình: {tập: (số cửa sổ có thể tạo, số cửa sổ bị bỏ)}, khớp docs/04_MODEL_TRAINING_EVALUATION.md mục 2
WINDOWS = {"24h_6h": {"train": (189769, 424), "val": (26185, 0), "test": (26377, 0)},
           "7d_24h": {"train": (189121, 225), "val": (26113, 0), "test": (26305, 0)}}
FIXED = ["dataset/raw/cemp_MSD_2007_2025_source.csv", "dataset/metadata/cemp_station_metadata.csv",
         "dataset/metadata/cemp_edi_1177_9_eml.xml", "docs/01_DATASET_ANALYSIS.md", "docs/02_DATA_CLEANING.md",
         "docs/03_DOMAIN_FEATURES_ANALYSIS.md", "docs/04_MODEL_TRAINING_EVALUATION.md", "docs/README.md"]

errors = []


def check(cond, msg):
    """Ghi lỗi msg nếu cond sai; trả về cond để dùng tiếp trong if."""
    if not cond:
        errors.append(msg)
    return cond


def tracked():
    """Danh sách file manifest phải theo dõi: file cố định, base_splits, mọi file trong scripts/ và artifacts/dataset/."""
    files = FIXED + [f"dataset/{d}/{n}.csv" for d in ("base_splits", "base_domain_splits") for n in NAMES]
    for d in ("scripts", "artifacts/dataset"):
        files += sorted(f"{d}/{f}" for f in os.listdir(os.path.join(REPO, d))
                        if os.path.isfile(os.path.join(REPO, d, f)) and not f.endswith(".pyc"))
    return files


def sha256(path, algo="sha256"):
    """Mã băm (mặc định SHA-256) của một file, đường dẫn tính từ gốc repo."""
    h = hashlib.new(algo)
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
        last = None                                 # chuỗi nguyên văn của số đo hợp lệ gần nhất
        for i, t in enumerate(times):
            s, x, k = raw[t][SOURCE[c]], vals[i][j], flags[i][j]
            if s != "NA" and lo <= float(s) <= hi:
                check(k == 0 and x == s, f"{t} {c}: observed raw {s}, got {x} flag {k}")
                last = s
            else:
                check(last is not None and k == 1 and x == last, f"{t} {c}: expected ffill {last}, got {x} flag {k}")
    for name, s in zip(NAMES, splits):
        got = sum(k == 1 for f in s[2] for k in f)
        check(got == FLAG_COUNTS[name], f"{name}: flag 1 count {got} != {FLAG_COUNTS[name]}")
        print(f"[3-5] {name}: {len(s[0])} rows {s[0][0]} .. {s[0][-1]} | flag 1: {got}")


def check_windows(splits):
    """Kiểm tra 6: đếm cửa sổ theo quy tắc của common.window_starts, cài đặt lại độc lập bằng thư viện chuẩn."""
    obs = [0]                                   # tổng tích lũy số mốc DO cờ 0
    for s in splits:
        for fl in s[2]:
            obs.append(obs[-1] + (fl[0] == 0))
    offs = [0]
    for s in splits:
        offs.append(offs[-1] + len(s[2]))
    for cfg, exp in WINDOWS.items():
        lin, lout = CONFIGS[cfg]
        got = {}
        for si, name in enumerate(NAMES):
            ts = range(max(offs[si], lin), offs[si + 1] - lout + 1)
            drop = sum(1 for t in ts if obs[t + lout] - obs[t] == 0)
            got[name] = (len(ts), drop)
        check(got == exp, f"windows {cfg}: {got} != {exp}")
        print(f"[6] windows {cfg} (total, dropped): {got}")


def saturation(t, spc):
    """DO bão hòa (mg/L), Benson và Krause (1984), độ mặn từ độ dẫn theo Schemel (2001); cùng công thức common.do_saturation."""
    T = t + 273.15
    R = max(spc, 0.0) / 53087.0
    sal = 0.0120 - 0.2174 * R**0.5 + 25.3283 * R + 13.7714 * R**1.5 - 6.4788 * R**2 + 2.5842 * R**2.5
    return math.exp(-139.34411 + 1.575701e5 / T - 6.642308e7 / T**2 + 1.243800e10 / T**3 - 8.621949e11 / T**4
                    - sal * (1.7674e-2 - 10.754 / T + 2140.7 / T**2))


def solar_sin(dt):
    """sin độ cao mặt trời tại trạm (Spencer 1971); cùng công thức common.solar_elevation_sin."""
    hr = dt.hour + dt.minute / 60
    g = 2 * math.pi / 365 * (dt.timetuple().tm_yday - 1 + (hr - 12) / 24)
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g)
                    - 0.040849 * math.sin(2 * g))
    ha = math.radians((hr * 60 + eqt + 4 * LON + 480) / 4 - 180)
    lat = math.radians(LAT)
    return math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)


def expected_domain(base, dt, k):
    """24 đặc trưng miền cho chuỗi nối ba tập, tính lại từng dòng bằng math theo định nghĩa của common.domain_features."""
    n = len(base)
    do, temp, ph, spc, chl = ([float(b[j]) for b in base] for j in (1, 2, 3, 4, 6))
    ok = [b[7] == "0" for b in base]
    sat = [saturation(temp[i], spc[i]) for i in range(n)]
    D = [sat[i] - do[i] for i in range(n)]
    se = [solar_sin(x) for x in dt]
    dd = [0.0] + [(do[i] - do[i - 1]) / 0.25 for i in range(1, n)]
    Dm = [D[0]] + [(D[i] + D[i - 1]) / 2 for i in range(1, n)]
    night = [False] + [se[i] < 0 and se[i - 1] < 0 and ok[i] and ok[i - 1] for i in range(1, n)]
    epoch = datetime.strptime(M2_EPOCH, FMT)
    out, er = [], 0.0
    s_n = s_dm = s_dd = s_D = s_l = 0.0                     # tổng trượt 96 mốc
    for i in range(n):
        light = max(se[i], 0.0)
        s_n, s_dm, s_dd = s_n + night[i], s_dm + (Dm[i] if night[i] else 0.0), s_dd + (dd[i] if night[i] else 0.0)
        s_D, s_l = s_D + Dm[i], s_l + light
        if i >= DAY:
            j = i - DAY
            s_n, s_D, s_l = s_n - night[j], s_D - Dm[j], s_l - max(se[j], 0.0)
            if night[j]:
                s_dm, s_dd = s_dm - Dm[j], s_dd - dd[j]
        if s_n >= NIGHT_MIN_STEPS:
            er = (k * s_dm - s_dd) / s_n
        lag = lambda v, h: v[i] - v[i - h] if i >= h else 0.0   # noqa: E731
        gpp24 = lag(do, DAY) - k * s_D * 0.25 + 24 * er
        eff = gpp24 / max(s_l * 0.25, 1e-3) if i >= DAY else 0.0
        w = do[max(0, i - DAY + 1):i + 1]
        mean = sum(w) / len(w)
        std = math.sqrt(sum((x - mean) ** 2 for x in w) / (len(w) - 1)) if len(w) > 1 else 0.0
        day = (dt[i].hour + dt[i].minute / 60) / 24
        year = (dt[i].timetuple().tm_yday - 1 + day) / 365.25
        hours = (dt[i] - epoch).total_seconds() / 3600
        v = {"do_sat_mgl": sat[i], "do_sat_pct": 100 * do[i] / sat[i], "do_deficit": D[i], "light": light,
             "gpp_now": eff * light, "er_now": er, "do_change_24h": lag(do, DAY), "do_std_24h": std,
             "do_anom_24h": do[i] - mean, "do_change_1h": lag(do, 4), "do_change_3h": lag(do, 12),
             "temp_change_3h": lag(temp, 12), "ph_change_3h": lag(ph, 12), "ph_change_24h": lag(ph, DAY),
             "log_chl": math.log(max(chl[i], CHL_FLOOR)), "spc_change_24h": lag(spc, DAY),
             "m2_sin": math.sin(2 * math.pi * hours / M2_HOURS), "m2_cos": math.cos(2 * math.pi * hours / M2_HOURS),
             "hour_sin": math.sin(2 * math.pi * day), "hour_cos": math.cos(2 * math.pi * day),
             "hour2_sin": math.sin(4 * math.pi * day), "hour2_cos": math.cos(4 * math.pi * day),
             "doy_sin": math.sin(2 * math.pi * year), "doy_cos": math.cos(2 * math.pi * year)}
        out.append([v[c] for c in DOMAIN_FEATURES])
    return out, night, Dm, dd


def check_k(base, dt, night, Dm, dd):
    """Kiểm tra 7a: hồi quy giữa các đêm train (trung bình dd theo trung bình Dm) cho lại k, ER, r, số đêm của file."""
    acc = {}
    for i, b in enumerate(base):
        if night[i] and dt[i] < datetime.strptime(SPLITS["train"][1], FMT):
            a = acc.setdefault((dt[i] + timedelta(hours=12)).date(), [0.0, 0.0, 0])
            a[0], a[1], a[2] = a[0] + Dm[i], a[1] + dd[i], a[2] + 1
    xs, ys = zip(*[(a[0] / a[2], a[1] / a[2]) for a in acc.values() if a[2] >= NIGHT_MIN_STEPS])
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx, syy = sum((x - mx) ** 2 for x in xs), sum((y - my) ** 2 for y in ys)
    k, er, r = sxy / sxx, -(my - sxy / sxx * mx), sxy / math.sqrt(sxx * syy)
    with open(REAERATION, newline="") as f:
        got = next(csv.DictReader(f))
    check(abs(float(got["k"]) - k) <= 5e-7 and abs(float(got["er"]) - er) <= 5e-7
          and abs(float(got["r"]) - r) <= 5e-5 and int(got["n_nights"]) == len(xs),
          f"reaeration_k.csv {dict(got)} != k {k:.6f} er {er:.6f} r {r:.4f} n {len(xs)}")
    print(f"[7] k = {k:.6f} mỗi giờ từ {len(xs)} đêm train, khớp {os.path.relpath(REAERATION, REPO)}")


def check_domain(splits):
    """Kiểm tra 7: đọc base_domain_splits, so phần gốc với base_splits, tính lại k và 24 đặc trưng miền theo từng dòng."""
    base = [[t] + v + [str(k) for k in f] for s in splits for t, v, f in zip(s[0], s[1], s[2])]
    dt = [x for s in splits for x in s[3]]
    rows = []
    for name in NAMES:
        with open(os.path.join(DOMAIN, f"{name}.csv"), newline="") as f:
            rd = csv.reader(f)
            check(next(rd) == ["datetime_pst"] + FEATURES + FLAGS + DOMAIN_FEATURES, f"domain {name}: header")
            rows += list(rd)
    if not check(len(rows) == len(base), f"domain rows {len(rows)} != {len(base)}"):
        return
    exp, night, Dm, dd = expected_domain(base, dt, read_k())
    check_k(base, dt, night, Dm, dd)
    worst, n_bad = 0.0, 0
    for r, b, e in zip(rows, base, exp):
        if r[:13] != b:
            n_bad += 1
            continue
        worst = max(worst, max(abs(float(x) - y) for x, y in zip(r[13:], e)))
    check(n_bad == 0, f"domain: {n_bad} rows differ from base_splits")
    check(worst <= 2e-6, f"domain: max feature deviation {worst:.2e}")
    print(f"[7] base_domain_splits: {len(rows)} rows, base columns identical, max feature deviation {worst:.1e}")


def main():
    """Chạy kiểm tra 1 đến 7 theo thứ tự; thoát mã 1 và in tối đa 50 lỗi nếu có lỗi."""
    if "--write" in sys.argv:
        man = {p: {"sha256": sha256(p), "bytes": os.path.getsize(os.path.join(REPO, p))} for p in tracked()}
        with open(MANIFEST, "w") as f:
            json.dump(man, f, indent=1)
        print("manifest written")
    with open(MANIFEST) as f:
        man = json.load(f)
    check(set(man) == set(tracked()), f"manifest file set differs: {sorted(set(man) ^ set(tracked()))}")
    for p, m in man.items():
        check(os.path.exists(os.path.join(REPO, p)) and sha256(p) == m["sha256"], f"sha256 mismatch: {p}")
    print(f"[1] sha256: {len(man)} files checked")
    check(os.path.getsize(RAW) == RAW_BYTES, f"raw bytes {os.path.getsize(RAW)} != {RAW_BYTES}")
    check(sha256(os.path.relpath(RAW, REPO), "md5") == RAW_MD5, "raw MD5 differs from EDI metadata")
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
    print(f"[2] raw: {len(raw)} rows, {RAW_BYTES} bytes, MD5 khớp metadata EDI")
    n_err = len(errors)
    splits = [check_split(n) for n in NAMES]
    for s1, s2 in zip(splits, splits[1:]):
        check(s2[3][0] - s1[3][-1] == STEP, "splits not contiguous")
    if len(errors) == n_err:                    # kiểm tra từng ô cần ba tập đúng cấu trúc
        check_cells(splits, raw)
    check_windows(splits)
    check_domain(splits)
    if errors:
        print("FAILED:")
        for e in errors[:50]:
            print("  ", e)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
