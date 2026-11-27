"""Căn cứ cho kiểm tra miền giá trị và cho quyết định giữ nguyên điểm DO đột biến một mốc (docs/02_DATA_CLEANING.md, mục 2 và 5).

Ghi vào artifacts/dataset/:
  cleaning_summary_train.csv: số giá trị ngoài miền (cả ba tập), nhỏ nhất và lớn nhất của cả file nguồn 2007 đến
    2025 (nguon_min, nguon_max: căn cứ đặt miền), phân vị 50, 99, 99.9 của độ lớn bước thay đổi
    15 phút trên train và ngưỡng bước lớn bất thường (buoc_lon = 2 lần phân vị 99), theo biến.
  do_reversal_points.csv: điểm DO đột biến một mốc (bước vào và bước ra đều lớn hơn buoc_lon của DO, ngược dấu) và
    phân loại: tu_nhien_khoi_nuoc (bước của nhiệt độ hoặc pH cùng mốc lớn hơn buoc_lon của biến đó),
    loi_thiet_bi (cảm biến khác NA trong vòng 1 giờ), chua_ro. Không điểm nào bị thay: phát hiện cần số đo sau
    điểm đó (không nhân quả).
  reversal_effect.csv: theo tập, số điểm đột biến, số mốc DO hợp lệ dưới 5 và dưới 6 mg/L và bao nhiêu mốc trong
    đó là điểm đột biến.
Chỉ đọc file nguồn (raw/).
Chạy từ gốc repo: uv run python scripts/cleaning_study.py
"""
import os

import pandas as pd

from common import FEATURES as F, NAMES, RANGES, RAW, SOURCE, STATS as OUT, load_raw_split

BIG_STEP = 2   # bước 15 phút lớn bất thường: lớn hơn BIG_STEP lần phân vị 99 của bước trên train, cùng quy tắc mọi biến


def summary(S):
    """Bảng cleaning_summary_train: số giá trị ngoài miền RANGES, nhỏ nhất và lớn nhất của cả file nguồn 2007 đến 2025,
    phân vị 50, 99, 99.9, giá trị lớn nhất của bước 15 phút và ngưỡng bước lớn buoc_lon = BIG_STEP lần phân vị 99."""
    d = S["train"]
    src = pd.read_csv(RAW, usecols=list(SOURCE.values()))
    rows = []
    for c in F:
        lo, hi = RANGES[c]
        oob = sum(int(((S[n][c] < lo) | (S[n][c] > hi)).sum()) for n in NAMES)
        step = d[c].diff().abs()
        p99 = round(step.quantile(.99), 3)
        rows.append(dict(feature=c, out_of_bounds_all=oob, nguon_min=src[SOURCE[c]].min(), nguon_max=src[SOURCE[c]].max(),
                         step_p50=step.median(), step_p99=p99, step_p999=step.quantile(.999),
                         step_max=step.max(), buoc_lon=BIG_STEP * p99))
    return pd.DataFrame(rows).round(3)


def reversals(S, big):
    """Bảng do_reversal_points: mọi điểm DO đột biến một mốc trong ba tập, kèm phân loại; big: buoc_lon theo biến."""
    rows = []
    others = [c for c in F if c != "do_mg_l"]
    for n in NAMES:
        x = S[n]
        s = x.do_mg_l
        i, o = s - s.shift(1), s - s.shift(-1)
        rev = (i.abs() > big["do_mg_l"] + 1e-9) & (o.abs() > big["do_mg_l"] + 1e-9) & (i * o > 0)
        dT = (x.water_temp_c - x.water_temp_c.shift(1)).abs()
        dP = (x.ph - x.ph.shift(1)).abs()
        comove = (dT > big["water_temp_c"] + 1e-9) | (dP > big["ph"] + 1e-9)
        na_near = x[others].isna().any(axis=1).astype(int).rolling(9, center=True, min_periods=1).max() > 0
        for t in s.index[rev]:
            cls = "tu_nhien_khoi_nuoc" if comove[t] else ("loi_thiet_bi" if na_near[t] else "chua_ro")
            rows.append(dict(split=n, time=str(t), prev=s.shift(1)[t], value=s[t], next=s.shift(-1)[t],
                             dT=dT[t], dpH=dP[t], other_sensor_na_1h=bool(na_near[t]), classification=cls))
    return pd.DataFrame(rows)


def reversal_effect(S, rev):
    """Bảng reversal_effect: theo tập, số điểm đột biến rev và số mốc DO hợp lệ dưới 5, dưới 6 mg/L là điểm đột biến."""
    rows = []
    for n in NAMES:
        do = S[n]["do_mg_l"]
        do = do[do.between(*RANGES["do_mg_l"])]                  # chỉ số đo hợp lệ (cờ 0 trong base_splits)
        is_rev = do.index.astype(str).isin(rev.loc[rev["split"] == n, "time"])
        rows.append(dict(tap=n, so_diem=int((rev["split"] == n).sum()),
                         moc_DO_duoi_5=int((do < 5).sum()), moc_DO_duoi_5_la_dot_bien=int(((do < 5) & is_rev).sum()),
                         moc_DO_duoi_6=int((do < 6).sum()), moc_DO_duoi_6_la_dot_bien=int(((do < 6) & is_rev).sum())))
    return pd.DataFrame(rows)


def main():
    """Ghi ba bảng căn cứ."""
    os.makedirs(OUT, exist_ok=True)
    S = {n: load_raw_split(n).set_index("datetime_pst") for n in NAMES}
    sm = summary(S)
    sm.to_csv(os.path.join(OUT, "cleaning_summary_train.csv"), index=False)
    rev = reversals(S, dict(zip(sm.feature, sm.buoc_lon)))
    rev.to_csv(os.path.join(OUT, "do_reversal_points.csv"), index=False)
    eff = reversal_effect(S, rev)
    eff.to_csv(os.path.join(OUT, "reversal_effect.csv"), index=False)
    print(rev.groupby(["split", "classification"]).size().to_string())
    print(eff.to_string(index=False))
    print("done")


if __name__ == "__main__":
    main()
