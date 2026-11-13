"""Thong so chuan hoa, kiem tra ngoai suy, phan loai diem DO bat thuong (can pandas, numpy).
Chay tu thu muc dataset: python analysis/window_scaler_study.py
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "results")
F = ["do_mg_l", "water_temp_c", "ph", "spc_us_cm", "turbidity_fnu", "chl_fluor_ug_l"]
NAMES = ["train", "val", "test"]


def main():
    S = {n: pd.read_csv(os.path.join(ROOT, "splits", f"{n}.csv"), na_values=["NA"],
                        parse_dates=["datetime_pst"]).set_index("datetime_pst") for n in NAMES}
    # 1. thong so chuan hoa tren train va kiem tra ngoai suy
    d = S["train"]
    rows = []
    for c in F:
        v = d[c].dropna()
        m, s = v.mean(), v.std()
        lv = np.log1p(v)
        r = dict(feature=c, mean=m, sd=s, median=v.median(), iqr=v.quantile(.75) - v.quantile(.25),
                 log1p_mean=lv.mean(), log1p_sd=lv.std())
        for n in NAMES:
            r[f"{n}_z_min"] = (S[n][c].min() - m) / s
            r[f"{n}_z_max"] = (S[n][c].max() - m) / s
        rows.append(r)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "scaler_stats_train.csv"), index=False)
    # 2. diem DO dao chieu don le (vao va ra > 0.8 mg/L, nguoc dau), phan loai theo ba tieu chi:
    #    (a) dong bien: nhiet do doi tu 0.3 do C hoac pH doi tu 0.05 tai cung moc -> khoi nuoc khac, tu nhien, giu nguyen
    #    (b) va (c) duoc thay bang trung binh hai moc lan can (chi trong train)
    #    (b) khong dong bien va cam bien khac bi NA trong vong 1 gio quanh diem -> loi thiet bi
    #    (c) khong dong bien, khong co bang chung thiet bi -> chua ro nguyen nhan (co the do suc khi)
    rows = []
    others = [c for c in F if c != "do_mg_l"]
    for n in NAMES:
        x = S[n]
        s = x.do_mg_l
        i, o = s - s.shift(1), s - s.shift(-1)
        rev = (i.abs() > 0.8) & (o.abs() > 0.8) & (i * o > 0)
        dT = (x.water_temp_c - x.water_temp_c.shift(1)).abs()
        dP = (x.ph - x.ph.shift(1)).abs()
        comove = (dT >= 0.3) | (dP >= 0.05 + 1e-9)
        na_near = x[others].isna().any(axis=1).astype(int).rolling(9, center=True, min_periods=1).max() > 0
        for t in s.index[rev]:
            if comove[t]:
                cls = "tu_nhien_khoi_nuoc"
            elif na_near[t]:
                cls = "loi_thiet_bi"
            else:
                cls = "chua_ro"
            rows.append(dict(split=n, time=str(t), prev=s.shift(1)[t], value=s[t], next=s.shift(-1)[t],
                             dT=dT[t], dpH=dP[t], other_sensor_na_1h=bool(na_near[t]), classification=cls))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "do_reversal_points.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
