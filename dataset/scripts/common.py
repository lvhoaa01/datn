"""Hang so va ham doc du lieu dung chung cho dataset CEMP RRI.

Phan dau chi dung thu vien chuan (cho verify.py); ham load_* can pandas.
"""
import csv
import os
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw", "cemp_RRI_2007_2025_source.csv")
BASE = os.path.join(ROOT, "base_splits")
FMT = "%Y-%m-%d %H:%M:%S"

# cot dau ra -> cot nguon, theo thu tu dau ra
SOURCE = {"do_mg_l": "dissolvedoxygen", "water_temp_c": "watertemperature", "ph": "ph",
          "spc_us_cm": "spc", "turbidity_fnu": "turbidity", "chl_fluor_ug_l": "fluorescence"}
FEATURES = list(SOURCE)
FLAGS = [f"flag_{c}" for c in FEATURES]
# ranh gioi theo gio PST co dinh (UTC-8), [bat dau, ket thuc)
SPLITS = {"train": ("2019-01-01 00:00:00", "2024-06-01 00:00:00"),
          "val": ("2024-06-01 00:00:00", "2025-03-01 00:00:00"),
          "test": ("2025-03-01 00:00:00", "2025-12-01 00:00:00")}
NAMES = list(SPLITS)
# mien vat ly chap nhan [min, max], don vi cua tung bien
RANGES = {"do_mg_l": (0, 20), "water_temp_c": (0, 35), "ph": (6, 10), "spc_us_cm": (20, 3000),
          "turbidity_fnu": (0, 1000), "chl_fluor_ug_l": (0, 200)}
MAX_GAP = 32              # khoang trong dai nhat duoc noi suy: 8 gio = 32 moc 15 phut
# cau hinh du bao: (do dai dau vao, do dai nhan) theo so moc; 24h_6h la cau hinh chinh
CONFIGS = {"24h_6h": (96, 24), "7d_24h": (672, 96)}


@lru_cache(maxsize=None)
def read_raw_split(name):
    """Cac dong nguon cua mot tap: tuple (thoi diem, tuple 6 chuoi gia tri nguyen van, 'NA' la trong)."""
    a, b = SPLITS[name]
    out = []
    with open(RAW, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            t = f"{r['date']} {r['time']}"          # chuoi ISO nen so sanh chuoi dung thu tu thoi gian
            if a <= t < b:
                out.append((t, tuple(r[SOURCE[c]] for c in FEATURES)))
    return tuple(out)


def load_raw_split(name):
    """Tap chua lam sach dang DataFrame: datetime_pst (datetime64) va 6 bien (float, NaN la trong)."""
    import pandas as pd
    rows = read_raw_split(name)
    d = pd.DataFrame([v for _, v in rows], columns=FEATURES).replace("NA", None).astype(float)
    d.insert(0, "datetime_pst", pd.to_datetime([t for t, _ in rows], format=FMT))
    return d


def load_base_split(name):
    """Tap da lam sach trong base_splits/: 6 bien (float) va 6 cot co (int 0 den 3)."""
    import pandas as pd
    return pd.read_csv(os.path.join(BASE, f"{name}.csv"), na_values=["NA"], parse_dates=["datetime_pst"],
                       dtype={f: "int8" for f in FLAGS})


def load_base_all():
    """Ba tap base_splits noi tiep nhau (chi so dong 0..n-1), them cot split."""
    import pandas as pd
    return pd.concat([load_base_split(n).assign(split=n) for n in NAMES], ignore_index=True)


def window_starts(d, n_in, n_out):
    """Thoi diem du bao hop le cua tung tap, la chi so dong trong d = load_base_all().

    Cua so t: dau vao [t - n_in, t), nhan [t, t + n_out). Nhan nam tron trong tap; dau vao duoc lay du lieu cuoi
    cua tap lien truoc. Giu cua so khi dau vao va nhan khong co o co 3 va nhan co it nhat mot moc DO co 0.
    """
    import numpy as np
    fl = d[FLAGS].to_numpy()
    long_gap = np.concatenate([[0], np.cumsum((fl == 3).any(axis=1))])
    observed = np.concatenate([[0], np.cumsum(fl[:, 0] == 0)])
    split = d["split"].to_numpy()
    out = {}
    for n in NAMES:
        idx = np.flatnonzero(split == n)
        t = np.arange(max(idx[0], n_in), idx[-1] + 2 - n_out)
        ok = (long_gap[t + n_out] - long_gap[t - n_in] == 0) & (observed[t + n_out] - observed[t] > 0)
        out[n] = t[ok]
    return out
