"""Tao train/val/test tu file nguon CEMP RRI.

Gia tri duoc chep nguyen van dang chuoi (khong ep kieu so, khong lam tron, khong noi suy, giu NA).
Chi doi ten cot va gop date + time thanh mot cot thoi gian.
Chay tu thu muc dataset:  python scripts/make_splits.py
"""
import csv
import os
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "raw", "cemp_RRI_2007_2025_source.csv")
OUT = os.path.join(ROOT, "splits")

# Ranh gioi theo gio PST co dinh (UTC-8): bat dau tinh, ket thuc khong tinh.
SPLITS = [
    ("train", "2019-01-01 00:00:00", "2024-06-01 00:00:00"),
    ("val", "2024-06-01 00:00:00", "2025-03-01 00:00:00"),
    ("test", "2025-03-01 00:00:00", "2025-12-01 00:00:00"),
]
# cot nguon -> cot dau ra (thu tu dau ra)
COLUMNS = [
    ("dissolvedoxygen", "do_mg_l"),
    ("watertemperature", "water_temp_c"),
    ("ph", "ph"),
    ("spc", "spc_us_cm"),
    ("turbidity", "turbidity_fnu"),
    ("fluorescence", "chl_fluor_ug_l"),
]
FMT = "%Y-%m-%d %H:%M:%S"


def main():
    os.makedirs(OUT, exist_ok=True)
    bounds = [(n, datetime.strptime(a, FMT), datetime.strptime(b, FMT)) for n, a, b in SPLITS]
    writers, files, counts = {}, {}, {}
    for n, _, _ in bounds:
        f = open(os.path.join(OUT, f"{n}.csv"), "w", newline="")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["datetime_pst"] + [o for _, o in COLUMNS])
        writers[n], files[n], counts[n] = w, f, 0
    with open(RAW, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            t = datetime.strptime(f"{row['date']} {row['time']}", FMT)
            for n, a, b in bounds:
                if a <= t < b:
                    writers[n].writerow([t.strftime(FMT)] + [row[c] for c, _ in COLUMNS])
                    counts[n] += 1
                    break
    for f in files.values():
        f.close()
    for n in counts:
        print(f"{n}: {counts[n]} rows")


if __name__ == "__main__":
    main()
