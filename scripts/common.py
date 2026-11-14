"""Hằng số, đường dẫn và hàm đọc dữ liệu dùng chung cho dataset CEMP RRI.

Phần khai báo và read_raw_split chỉ dùng thư viện chuẩn (cho verify_dataset.py); các hàm load_* cần pandas,
window_starts cần numpy.
"""
import csv
import os
from functools import lru_cache

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(REPO, "dataset")
RAW = os.path.join(DATASET, "raw", "cemp_RRI_2007_2025_source.csv")
BASE = os.path.join(DATASET, "base_splits")
STATS = os.path.join(REPO, "artifacts", "dataset")   # thống kê và hình của dataset
FMT = "%Y-%m-%d %H:%M:%S"

# tên cột đầu ra: tên cột trong file nguồn, theo thứ tự đầu ra
SOURCE = {"do_mg_l": "dissolvedoxygen", "water_temp_c": "watertemperature", "ph": "ph",
          "spc_us_cm": "spc", "turbidity_fnu": "turbidity", "chl_fluor_ug_l": "fluorescence"}
FEATURES = list(SOURCE)
FLAGS = [f"flag_{c}" for c in FEATURES]
# ranh giới theo giờ PST cố định (UTC-8), [bắt đầu, kết thúc)
SPLITS = {"train": ("2019-01-01 00:00:00", "2024-06-01 00:00:00"),
          "val": ("2024-06-01 00:00:00", "2025-03-01 00:00:00"),
          "test": ("2025-03-01 00:00:00", "2025-12-01 00:00:00")}
NAMES = list(SPLITS)
# miền vật lý chấp nhận [min, max], theo đơn vị của từng biến
RANGES = {"do_mg_l": (0, 20), "water_temp_c": (0, 35), "ph": (6, 10), "spc_us_cm": (20, 3000),
          "turbidity_fnu": (0, 1000), "chl_fluor_ug_l": (0, 200)}
MAX_GAP = 32              # khoảng trống dài nhất được nội suy: 8 giờ = 32 mốc 15 phút
# cấu hình dự báo: (độ dài đầu vào, độ dài nhãn) tính bằng số mốc; 24h_6h là cấu hình chính
CONFIGS = {"24h_6h": (96, 24), "7d_24h": (672, 96)}


@lru_cache(maxsize=None)
def read_raw_split(name):
    """Các dòng nguồn của một tập: tuple (thời điểm, tuple 6 chuỗi giá trị nguyên văn, 'NA' là trống)."""
    a, b = SPLITS[name]
    out = []
    with open(RAW, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            t = f"{r['date']} {r['time']}"          # chuỗi ISO nên so sánh chuỗi đúng thứ tự thời gian
            if a <= t < b:
                out.append((t, tuple(r[SOURCE[c]] for c in FEATURES)))
    return tuple(out)


def load_raw_split(name):
    """Một tập chưa làm sạch dạng DataFrame: datetime_pst (datetime64) và 6 biến (float, NaN là trống)."""
    import pandas as pd
    rows = read_raw_split(name)
    d = pd.DataFrame([v for _, v in rows], columns=FEATURES).replace("NA", None).astype(float)
    d.insert(0, "datetime_pst", pd.to_datetime([t for t, _ in rows], format=FMT))
    return d


def load_base_split(name):
    """Một tập đã làm sạch trong base_splits/: 6 biến (float, NaN ở ô cờ 3) và 6 cột cờ (int8, 0 đến 3)."""
    import pandas as pd
    return pd.read_csv(os.path.join(BASE, f"{name}.csv"), na_values=["NA"], parse_dates=["datetime_pst"],
                       dtype={f: "int8" for f in FLAGS})


def load_base_all():
    """Ba tập base_splits nối tiếp theo thời gian, chỉ số dòng 0 đến n - 1, thêm cột split."""
    import pandas as pd
    return pd.concat([load_base_split(n).assign(split=n) for n in NAMES], ignore_index=True)


def window_starts(d, n_in, n_out):
    """Thời điểm dự báo hợp lệ của từng tập: {tên tập: mảng chỉ số dòng t trong d = load_base_all()}.

    Cửa sổ t: đầu vào [t - n_in, t), nhãn [t, t + n_out). Nhãn nằm trọn trong tập; đầu vào được lấy dữ liệu cuối
    của tập liền trước. Giữ cửa sổ khi đầu vào và nhãn không có ô cờ 3 và nhãn có ít nhất một mốc DO cờ 0.
    Đếm bằng tổng tích lũy nên mỗi cửa sổ kiểm tra trong O(1).
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
