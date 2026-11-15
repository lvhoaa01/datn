"""Căn cứ cho kiểm tra miền giá trị và cho quyết định giữ nguyên điểm DO đột biến một mốc (dataset/LAM_SACH.md, mục 3 và 6).

Ghi vào artifacts/dataset/:
  cleaning_summary_train.csv: số giá trị ngoài miền (cả ba tập), phân vị bước thay đổi 15 phút và
    đoạn giá trị không đổi trên train, theo biến.
  do_reversal_points.csv: điểm DO đột biến một mốc (vào và ra quá 0.8 mg/L, ngược dấu) và phân loại:
    tu_nhien_khoi_nuoc (nhiệt độ đổi từ 0.3 độ C hoặc pH đổi từ 0.05 cùng mốc, giữ nguyên),
    loi_thiet_bi (cảm biến khác NA trong vòng 1 giờ), chua_ro. Không điểm nào bị thay: phát hiện cần số đo sau
    điểm đó (không nhân quả).
  reversal_effect.csv: theo tập, số điểm đột biến, số mốc DO dưới 5 và dưới 6 mg/L là điểm đột biến, và MAE của
    cách cùng giờ hôm qua (cấu hình 24h_6h, cửa sổ và mặt nạ như mô hình) khi giữ và khi bỏ các mốc nhãn là điểm
    đột biến. Cần base_splits (chạy sau build_base_splits.py).
Chạy từ gốc repo: uv run python scripts/cleaning_study.py
"""
import os

import numpy as np
import pandas as pd

from common import (CONFIGS, FEATURES as F, NAMES, OBSERVED, RANGES, STATS as OUT, load_base_all, load_raw_split,
                    runs, window_starts)

# điểm DO đột biến một mốc: bước vào và bước ra tối thiểu, mg/L; nhiệt độ (độ C) hoặc pH đổi cùng mốc từ ngưỡng
# tương ứng cho thấy khối nước khác đi qua, là hiện tượng thật
SPIKE_MG_L, COMOVE_TEMP_C, COMOVE_PH = 0.8, 0.3, 0.05


def summary(S):
    """Bảng cleaning_summary_train: số giá trị ngoài miền RANGES, phân vị 99 và 99.9 của bước 15 phút, đoạn không đổi dài nhất."""
    d = S["train"]
    rows = []
    for c in F:
        lo, hi = RANGES[c]
        oob = sum(int(((S[n][c] < lo) | (S[n][c] > hi)).sum()) for n in NAMES)
        step = d[c].diff().abs()
        flat_h = (runs((d[c].diff() == 0).to_numpy())[1] + 1) * 0.25
        rows.append(dict(feature=c, out_of_bounds_all=oob, step_p99=step.quantile(.99), step_p999=step.quantile(.999),
                         step_max=step.max(), longest_flat_h=float(flat_h.max()), flat_runs_ge_6h=int((flat_h >= 6).sum())))
    return pd.DataFrame(rows).round(3)


def reversals(S):
    """Bảng do_reversal_points: mọi điểm DO đột biến một mốc trong ba tập, kèm phân loại."""
    rows = []
    others = [c for c in F if c != "do_mg_l"]
    for n in NAMES:
        x = S[n]
        s = x.do_mg_l
        i, o = s - s.shift(1), s - s.shift(-1)
        rev = (i.abs() > SPIKE_MG_L) & (o.abs() > SPIKE_MG_L) & (i * o > 0)
        dT = (x.water_temp_c - x.water_temp_c.shift(1)).abs()
        dP = (x.ph - x.ph.shift(1)).abs()
        comove = (dT >= COMOVE_TEMP_C) | (dP >= COMOVE_PH + 1e-9)
        na_near = x[others].isna().any(axis=1).astype(int).rolling(9, center=True, min_periods=1).max() > 0
        for t in s.index[rev]:
            cls = "tu_nhien_khoi_nuoc" if comove[t] else ("loi_thiet_bi" if na_near[t] else "chua_ro")
            rows.append(dict(split=n, time=str(t), prev=s.shift(1)[t], value=s[t], next=s.shift(-1)[t],
                             dT=dT[t], dpH=dP[t], other_sensor_na_1h=bool(na_near[t]), classification=cls))
    return pd.DataFrame(rows)


def reversal_effect(rev):
    """Bảng reversal_effect: ảnh hưởng của các điểm đột biến rev lên mốc DO thấp và lên sai số cùng giờ hôm qua."""
    d = load_base_all()
    t = d["datetime_pst"].astype(str).to_numpy()
    is_rev = np.isin(t, rev["time"].to_numpy())
    do = d["do_mg_l"].to_numpy()
    obs = d["flag_do_mg_l"].to_numpy() == OBSERVED
    n_in, n_out = CONFIGS["24h_6h"]
    starts = window_starts(d, n_in, n_out)
    rows = []
    for n in NAMES:
        sel = (d["split"] == n).to_numpy()
        idx = starts[n][:, None] + np.arange(n_out)
        e, m = np.abs(do[idx - 96] - do[idx]), obs[idx]
        rows.append(dict(tap=n, so_diem=int((is_rev & sel).sum()),
                         moc_DO_duoi_5=int((sel & obs & (do < 5)).sum()),
                         moc_DO_duoi_5_la_dot_bien=int((sel & obs & (do < 5) & is_rev).sum()),
                         moc_DO_duoi_6=int((sel & obs & (do < 6)).sum()),
                         moc_DO_duoi_6_la_dot_bien=int((sel & obs & (do < 6) & is_rev).sum()),
                         MAE_cung_gio_hom_qua=e[m].mean(), MAE_bo_moc_dot_bien=e[m & ~is_rev[idx]].mean()))
    return pd.DataFrame(rows).round(4)


def main():
    """Ghi ba bảng căn cứ."""
    os.makedirs(OUT, exist_ok=True)
    S = {n: load_raw_split(n).set_index("datetime_pst") for n in NAMES}
    summary(S).to_csv(os.path.join(OUT, "cleaning_summary_train.csv"), index=False)
    rev = reversals(S)
    rev.to_csv(os.path.join(OUT, "do_reversal_points.csv"), index=False)
    eff = reversal_effect(rev)
    eff.to_csv(os.path.join(OUT, "reversal_effect.csv"), index=False)
    print(rev.groupby(["split", "classification"]).size().to_string())
    print(eff.to_string(index=False))
    print("done")


if __name__ == "__main__":
    main()
