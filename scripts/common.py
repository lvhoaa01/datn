"""Hằng số, đường dẫn và hàm đọc dữ liệu dùng chung cho dataset CEMP MSD.

Phần khai báo và read_raw_split chỉ dùng thư viện chuẩn (cho verify_dataset.py); các hàm còn lại import numpy,
pandas, scikit-learn bên trong hàm nên verify_dataset.py không phụ thuộc các gói này.
"""
import csv
import hashlib
import os
from functools import lru_cache

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(REPO, "dataset")
RAW = os.path.join(DATASET, "raw", "cemp_MSD_2007_2025_source.csv")
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
DAY = 96                  # số mốc 15 phút trong một ngày
OBSERVED, FILLED = 0, 1    # giá trị cờ: số đo thật, ô điền bằng số đo thật gần nhất phía trước (ffill)
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


def runs(mask):
    """Vị trí bắt đầu và độ dài các đoạn True liên tiếp trong mảng boolean (cần numpy)."""
    import numpy as np
    d = np.diff(np.concatenate([[0], np.asarray(mask, bool).astype(np.int8), [0]]))
    s = np.flatnonzero(d == 1)
    return s, np.flatnonzero(d == -1) - s


def load_raw_split(name):
    """Một tập chưa làm sạch dạng DataFrame: datetime_pst (datetime64) và 6 biến (float, NaN là trống)."""
    import pandas as pd
    rows = read_raw_split(name)
    d = pd.DataFrame([v for _, v in rows], columns=FEATURES).replace("NA", None).astype(float)
    d.insert(0, "datetime_pst", pd.to_datetime([t for t, _ in rows], format=FMT))
    return d


def data_fingerprint():
    """Mã SHA-256 (16 ký tự đầu) của ba file base_splits; run huấn luyện ghi mã này để biết đã học trên dữ liệu nào."""
    h = hashlib.sha256()
    for n in NAMES:
        with open(os.path.join(BASE, f"{n}.csv"), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def load_base_split(name):
    """Một tập đã làm sạch trong base_splits/: 6 biến (float, không có ô trống) và 6 cột cờ (int8: 0, 1)."""
    import pandas as pd
    return pd.read_csv(os.path.join(BASE, f"{name}.csv"), parse_dates=["datetime_pst"],
                       dtype={f: "int8" for f in FLAGS})


def load_base_all():
    """Ba tập base_splits nối tiếp theo thời gian, chỉ số dòng 0 đến n - 1, thêm cột split."""
    import pandas as pd
    return pd.concat([load_base_split(n).assign(split=n) for n in NAMES], ignore_index=True)


def window_starts(d, n_in, n_out):
    """Thời điểm dự báo hợp lệ của từng tập: {tên tập: mảng chỉ số dòng t trong d = load_base_all()}.

    Cửa sổ t: đầu vào [t - n_in, t), nhãn [t, t + n_out). Nhãn nằm trọn trong tập; đầu vào được lấy dữ liệu cuối
    của tập liền trước. Giữ cửa sổ khi nhãn có ít nhất một mốc DO cờ 0 (có DO thật để tính sai số).
    Đếm bằng tổng tích lũy nên mỗi cửa sổ kiểm tra trong O(1).
    """
    import numpy as np
    observed = np.concatenate([[0], np.cumsum(d["flag_do_mg_l"].to_numpy() == OBSERVED)])
    split = d["split"].to_numpy()
    out = {}
    for n in NAMES:
        idx = np.flatnonzero(split == n)
        t = np.arange(max(idx[0], n_in), idx[-1] + 2 - n_out)
        out[n] = t[observed[t + n_out] - observed[t] > 0]
    return out


def fit_scaler(d, splits=("train",)):
    """StandardScaler (trừ trung bình, chia độ lệch chuẩn tổng thể) khớp trên 6 biến của các tập splits trong d.

    Mặc định chỉ train (README mục 5.3); khi huấn luyện lại trên train và val để chấm test thì truyền ("train", "val").
    """
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit(d.loc[d["split"].isin(splits), FEATURES].to_numpy())


def metrics(pred, true, mask):
    """MAE, RMSE (mg/L) và R2 trên các mốc mask (DO cờ 0); R2 = 1 - SSE / SST, SST quanh trung bình DO thật đã chọn."""
    import numpy as np
    e, y = (pred - true)[mask], true[mask]
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                R2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()))


def inverse_do(scaler, z):
    """Đổi DO đã chuẩn hóa (mảng bất kỳ hình dạng) về mg/L; DO là cột 0 của FEATURES."""
    return z * scaler.scale_[0] + scaler.mean_[0]
