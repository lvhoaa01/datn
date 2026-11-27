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
DOMAIN = os.path.join(DATASET, "base_domain_splits")   # base_splits cộng đặc trưng miền
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
# miền chấp nhận [min, max] theo đơn vị của từng biến: rộng hơn mọi số đo đã thẩm định của cả file nguồn 2007 đến 2025
# (artifacts/dataset/cleaning_summary_train.csv, cột nguon_min, nguon_max), đủ hẹp để bắt lỗi đơn vị, lỗi cảm biến
RANGES = {"do_mg_l": (0, 30), "water_temp_c": (0, 35), "ph": (6, 10), "spc_us_cm": (20, 3000),
          "turbidity_fnu": (0, 1000), "chl_fluor_ug_l": (0, 200)}
DAY = 96                  # số mốc 15 phút trong một ngày
OBSERVED, FILLED = 0, 1    # giá trị cờ: số đo thật, ô điền bằng số đo thật gần nhất phía trước (ffill)
# cấu hình dự báo: (độ dài đầu vào, độ dài nhãn) tính bằng số mốc; 24h_6h là cấu hình chính
CONFIGS = {"24h_6h": (96, 24), "7d_24h": (672, 96)}
# tên chuẩn (tiếng Việt) của sáu biến, dùng thống nhất trong hình và tài liệu
VAR_NAMES = {"do_mg_l": "DO", "water_temp_c": "Nhiệt độ nước", "ph": "pH", "spc_us_cm": "Độ dẫn điện",
             "turbidity_fnu": "Độ đục", "chl_fluor_ug_l": "Chlorophyll"}
UNITS = {"do_mg_l": "mg/L", "water_temp_c": "độ C", "ph": "", "spc_us_cm": "uS/cm", "turbidity_fnu": "FNU",
         "chl_fluor_ug_l": "ug/L"}
# 10 nhóm của 24 đặc trưng miền (docs/03_DOMAIN_FEATURES_ANALYSIS.md mục 4): (khóa, tên chuẩn, các cột)
DOMAIN_GROUPS = [("bao_hoa", "Bão hòa oxy", ["do_sat_mgl", "do_sat_pct", "do_deficit"]),
                 ("anh_sang", "Ánh sáng", ["light"]),
                 ("trao_doi_chat", "Quang hợp và hô hấp", ["gpp_now", "er_now"]),
                 ("do_24h", "Thống kê DO 24 giờ", ["do_change_24h", "do_std_24h", "do_anom_24h"]),
                 ("ngan_han", "Biến thiên ngắn hạn", ["do_change_1h", "do_change_3h", "temp_change_3h"]),
                 ("bien_thien_ph", "Biến thiên pH", ["ph_change_3h", "ph_change_24h"]),
                 ("log_chl", "Logarit chlorophyll", ["log_chl"]),
                 ("bien_thien_spc", "Biến thiên độ dẫn điện", ["spc_change_24h"]),
                 ("trieu", "Thủy triều M2", ["m2_sin", "m2_cos"]),
                 ("lich", "Lịch", ["hour_sin", "hour_cos", "hour2_sin", "hour2_cos", "doy_sin", "doy_cos"])]
DOMAIN_FEATURES = [c for _, _, cols in DOMAIN_GROUPS for c in cols]
LAT, LON = 37.786, -121.306          # tọa độ trạm MSD (độ), dùng tính độ cao mặt trời
M2_HOURS = 12.4206                   # chu kỳ triều bán nhật mặt trăng M2 (giờ)
M2_EPOCH = "2019-01-01 00:00:00"     # mốc tính pha M2, đầu chuỗi dùng cho mô hình
CHL_FLOOR = 0.1                      # chl nhỏ nhất có trong dữ liệu (ug/L), chặn dưới trước khi lấy ln
NIGHT_MIN_STEPS = 16                 # đêm dùng cho cân bằng oxy cần ít nhất 16 bước 15 phút (4 giờ) DO đo thật
REAERATION = os.path.join(STATS, "reaeration_k.csv")   # k ước lượng trên train (build_domain_splits.py)


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


def data_fingerprint(folder=BASE):
    """Mã SHA-256 (16 ký tự đầu) của ba file trong folder; run huấn luyện ghi mã này để biết đã học trên dữ liệu nào."""
    h = hashlib.sha256()
    for n in NAMES:
        with open(os.path.join(folder, f"{n}.csv"), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def load_base_split(name, folder=BASE):
    """Một tập đã làm sạch: 6 biến (float, không có ô trống), 6 cột cờ (int8: 0, 1) và, với DOMAIN, 24 đặc trưng miền."""
    import pandas as pd
    return pd.read_csv(os.path.join(folder, f"{name}.csv"), parse_dates=["datetime_pst"],
                       dtype={f: "int8" for f in FLAGS})


def load_base_all(folder=BASE):
    """Ba tập trong folder nối tiếp theo thời gian, chỉ số dòng 0 đến n - 1, thêm cột split."""
    import pandas as pd
    return pd.concat([load_base_split(n, folder).assign(split=n) for n in NAMES], ignore_index=True)


def salinity(spc_us_cm):
    """Độ mặn từ độ dẫn quy về 25 độ C (uS/cm), theo công thức rút gọn của Schemel (2001), R = SpC / 53087."""
    import numpy as np
    R = np.clip(spc_us_cm, 0, None) / 53087.0
    return 0.0120 - 0.2174 * R**0.5 + 25.3283 * R + 13.7714 * R**1.5 - 6.4788 * R**2 + 2.5842 * R**2.5


def do_saturation(temp_c, spc_us_cm, sal=None):
    """DO bão hòa (mg/L) theo nhiệt độ (độ C) và độ dẫn (uS/cm), áp suất 1 atm.

    Công thức Benson và Krause (1980, 1984), phương trình 7 và 8 của USGS (2011, Technical Memorandum 2011.03);
    độ mặn theo salinity (Schemel 2001), hoặc truyền sẵn qua sal.
    """
    import numpy as np
    T = temp_c + 273.15
    sal = salinity(spc_us_cm) if sal is None else sal
    lnc = (-139.34411 + 1.575701e5 / T - 6.642308e7 / T**2 + 1.243800e10 / T**3 - 8.621949e11 / T**4
           - sal * (1.7674e-2 - 10.754 / T + 2140.7 / T**2))
    return np.exp(lnc)


def solar_elevation_sin(t):
    """sin độ cao mặt trời tại trạm theo giờ PST (chuỗi datetime), theo NOAA GMD (General Solar Position Calculations).

    Xích vĩ và phương trình thời gian là chuỗi Fourier của Spencer (1971); dùng 365 ngày cho mọi năm (NOAA dùng 366 ở năm
    nhuận, light lệch tối đa 0.004).
    """
    import numpy as np
    doy = t.dt.day_of_year.to_numpy()
    hr = (t.dt.hour + t.dt.minute / 60).to_numpy()
    g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)            # góc ngày (radian)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g)
                    - 0.040849 * np.sin(2 * g))                  # phương trình thời gian (phút)
    tst = hr * 60 + eqt + 4 * LON + 480                           # giờ mặt trời thật (phút); PST = UTC - 8 giờ
    ha = np.radians(tst / 4 - 180)                                # góc giờ
    lat = np.radians(LAT)
    return np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)


def oxygen_steps(d, se):
    """Đại lượng theo bước 15 phút cho cân bằng oxy: thiếu hụt D tại mốc, tốc độ đổi DO dd (mg/L mỗi giờ) từ mốc trước,
    thiếu hụt giữa bước Dm, và night: bước nằm trọn trong đêm (mặt trời dưới chân trời ở cả hai đầu) với DO đo thật
    ở cả hai đầu."""
    import numpy as np
    do = d["do_mg_l"].to_numpy()
    D = do_saturation(d["water_temp_c"], d["spc_us_cm"]).to_numpy() - do
    ok = d["flag_do_mg_l"].to_numpy() == OBSERVED
    dd = np.r_[0.0, np.diff(do)] / 0.25
    Dm = np.r_[D[0], (D[1:] + D[:-1]) / 2]
    night = (se < 0) & np.r_[False, se[:-1] < 0] & np.r_[False, ok[1:] & ok[:-1]]
    return D, dd, Dm, night


def night_table(d, se=None):
    """Mỗi đêm của train một dòng: trung bình dd và Dm trên các bước đêm, số bước, nhiệt độ TB; nhãn đêm là ngày của
    mốc cộng 12 giờ (đêm 14 sang 15 mang nhãn ngày 15); chỉ giữ đêm có ít nhất NIGHT_MIN_STEPS bước."""
    import pandas as pd
    se = solar_elevation_sin(d["datetime_pst"]) if se is None else se
    _, dd, Dm, night = oxygen_steps(d, se)
    m = night & (d["split"].to_numpy() == "train")
    lab = (d["datetime_pst"] + pd.Timedelta(hours=12)).dt.date.to_numpy()
    g = pd.DataFrame({"night": lab[m], "dd": dd[m], "D": Dm[m], "T": d["water_temp_c"].to_numpy()[m]}).groupby("night")
    g = g.agg(dd=("dd", "mean"), D=("D", "mean"), T=("T", "mean"), n=("dd", "size"))
    return g[g.n >= NIGHT_MIN_STEPS]


def fit_reaeration(nights):
    """Hồi quy tuyến tính giữa các đêm: dd = k * D - ER; trả về k (mỗi giờ), ER (mg/L mỗi giờ), hệ số tương quan r."""
    import numpy as np
    k, a = np.polyfit(nights.D, nights.dd, 1)
    return dict(k=float(k), er=float(-a), r=float(np.corrcoef(nights.D, nights.dd)[0, 1]), n_nights=len(nights))


def read_k():
    """k (mỗi giờ) đã ước lượng trên train, đọc từ artifacts/dataset/reaeration_k.csv."""
    with open(REAERATION, newline="") as f:
        return float(next(csv.DictReader(f))["k"])


def domain_features(d, k):
    """24 đặc trưng miền DOMAIN_FEATURES cho chuỗi đã làm sạch d (liên tục 15 phút), k là hệ số trao đổi khí (mỗi giờ).

    Nhân quả: mỗi mốc chỉ dùng chính nó và các mốc trước. Tổng, trung bình, độ lệch chuẩn trượt lấy trên 96 mốc đến hết
    mốc hiện tại (ít hơn ở đầu chuỗi); hiệu với 1, 3, 24 giờ trước bằng 0 khi chưa có mốc đó.
    Cân bằng oxy (Odum 1956): dDO/dt = GPP - ER + k * D, D = DO bão hòa - DO.
      er_now: ER của các bước đêm trong 24 giờ qua, ER = k * TB(Dm) - TB(dd); chưa đủ bước đêm thì giữ giá trị trước.
      gpp_now: GPP 24 giờ qua (thay đổi DO 24 giờ, trừ trao đổi khí, cộng 24 giờ hô hấp) chia lượng ánh sáng 24 giờ,
               nhân ánh sáng hiện tại; 24 giờ đầu chuỗi bằng 0.
    """
    import numpy as np
    import pandas as pd
    t = d["datetime_pst"]
    day = (t.dt.hour + t.dt.minute / 60) / 24                   # phần ngày đã qua
    year = (t.dt.day_of_year - 1 + day) / 365.25                  # phần năm đã qua
    hours = (t - pd.Timestamp(M2_EPOCH)).dt.total_seconds() / 3600
    do, temp, ph, spc = d["do_mg_l"], d["water_temp_c"], d["ph"], d["spc_us_cm"]
    sat = do_saturation(temp, spc)
    roll = do.rolling(DAY, min_periods=1)
    se = solar_elevation_sin(t)
    light = np.clip(se, 0, None)
    _, dd, Dm, night = oxygen_steps(d, se)
    S = lambda v: pd.Series(v).rolling(DAY, min_periods=1).sum().to_numpy()   # tổng 96 mốc gần nhất  # noqa: E731
    n_night = S(night.astype(float))
    er = (k * S(np.where(night, Dm, 0)) - S(np.where(night, dd, 0))) / np.maximum(n_night, 1)
    er = pd.Series(np.where(n_night >= NIGHT_MIN_STEPS, er, np.nan)).ffill().fillna(0.0).to_numpy()
    gpp24 = do.diff(DAY).fillna(0.0).to_numpy() - k * S(Dm) * 0.25 + 24 * er     # mg/L trong 24 giờ
    eff = gpp24 / np.maximum(S(light) * 0.25, 1e-3)                              # GPP trên một đơn vị ánh sáng
    eff[:DAY] = 0.0
    return pd.DataFrame({
        "do_sat_mgl": sat, "do_sat_pct": 100 * do / sat, "do_deficit": sat - do,
        "light": light, "gpp_now": eff * light, "er_now": er,
        "do_change_24h": do.diff(DAY).fillna(0.0), "do_std_24h": roll.std().fillna(0.0), "do_anom_24h": do - roll.mean(),
        "do_change_1h": do.diff(4).fillna(0.0), "do_change_3h": do.diff(12).fillna(0.0),
        "temp_change_3h": temp.diff(12).fillna(0.0),
        "ph_change_3h": ph.diff(12).fillna(0.0), "ph_change_24h": ph.diff(DAY).fillna(0.0),
        "log_chl": np.log(np.clip(d["chl_fluor_ug_l"], CHL_FLOOR, None)),
        "spc_change_24h": spc.diff(DAY).fillna(0.0),
        "m2_sin": np.sin(2 * np.pi * hours / M2_HOURS), "m2_cos": np.cos(2 * np.pi * hours / M2_HOURS),
        "hour_sin": np.sin(2 * np.pi * day), "hour_cos": np.cos(2 * np.pi * day),
        "hour2_sin": np.sin(4 * np.pi * day), "hour2_cos": np.cos(4 * np.pi * day),
        "doy_sin": np.sin(2 * np.pi * year), "doy_cos": np.cos(2 * np.pi * year)}, index=d.index)[DOMAIN_FEATURES]


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


def fit_scaler(d, splits=("train",), cols=FEATURES):
    """StandardScaler (trừ trung bình, chia độ lệch chuẩn tổng thể) khớp trên các cột cols của các tập splits trong d.

    Mặc định chỉ train (docs/04_MODEL_TRAINING_EVALUATION.md mục 6); khi huấn luyện lại trên train và val để chấm test thì truyền ("train", "val").
    """
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit(d.loc[d["split"].isin(splits), list(cols)].to_numpy())


def metrics(pred, true, mask):
    """MAE, RMSE (mg/L) và R2 trên các mốc mask (DO cờ 0); R2 = 1 - SSE / SST, SST quanh trung bình DO thật đã chọn."""
    import numpy as np
    e, y = (pred - true)[mask], true[mask]
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                R2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()))


def read_json(path):
    """Đọc một file JSON (config.json, metrics.json của các lần chạy)."""
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path, **kw):
    """Ghi obj ra file JSON; kw chuyển thẳng cho json.dump (indent, ensure_ascii)."""
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, **kw)


def inverse_do(scaler, z):
    """Đổi DO đã chuẩn hóa (mảng bất kỳ hình dạng) về mg/L; DO là cột 0 của FEATURES."""
    return z * scaler.scale_[0] + scaler.mean_[0]
