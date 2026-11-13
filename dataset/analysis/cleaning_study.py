"""Nghien cuu lam sach va phan phoi tren tap train (can pandas, numpy, scipy).
Chay tu thu muc dataset: python analysis/cleaning_study.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import skew

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "results")
F = ["do_mg_l", "water_temp_c", "ph", "spc_us_cm", "turbidity_fnu", "chl_fluor_ug_l"]
# mien vat ly hop ly cho song ngot (gioi han rong, chi bat loi ro rang)
BOUNDS = {"do_mg_l": (0, 20), "water_temp_c": (0, 35), "ph": (6, 10), "spc_us_cm": (20, 3000),
          "turbidity_fnu": (0, 1000), "chl_fluor_ug_l": (0, 200)}
# nguong tuyet doi toi thieu cho spike (don vi cua bien)
FLOOR = {"do_mg_l": 1.0, "water_temp_c": 1.5, "ph": 0.3, "spc_us_cm": 100, "turbidity_fnu": 10, "chl_fluor_ug_l": 3}


def runs(mask):
    m = np.asarray(mask, bool)
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return s, e - s


def hampel_reversal(s, floor, k=6, w=25):
    med = s.rolling(w, center=True, min_periods=w // 2).median()
    mad = (s - med).abs().rolling(w, center=True, min_periods=w // 2).median()
    thr = np.maximum(floor, k * 1.4826 * mad)
    rev = (s - s.shift(1)) * (s - s.shift(-1)) > 0
    return ((s - med).abs() > thr) & rev


def main():
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(os.path.join(ROOT, "splits", "train.csv"), na_values=["NA"], parse_dates=["datetime_pst"]).set_index("datetime_pst")
    rows, spikes = [], []
    for c in F:
        s = d[c]
        v = s.dropna()
        lo, hi = BOUNDS[c]
        sp = hampel_reversal(s, FLOOR[c])
        same = (s.diff() == 0).to_numpy()
        st, ln = runs(same)
        flat_h = (ln + 1) * 0.25
        dif = s.diff().abs()
        r = dict(feature=c, out_of_bounds=int(((v < lo) | (v > hi)).sum()), spikes_isolated=int(sp.sum()),
                 spikes_per_10k=round(sp.sum() / len(v) * 1e4, 2), longest_flat_h=float(flat_h.max()) if len(ln) else 0,
                 flat_runs_ge_6h=int((flat_h >= 6).sum()), step_p999=dif.quantile(.999), step_max=dif.max(),
                 skew_raw=skew(v), skew_log1p=skew(np.log1p(v)) if v.min() > -1 else np.nan)
        rows.append(r)
        for t in s.index[sp]:
            i = d.index.get_loc(t)
            spikes.append(dict(feature=c, time=str(t), prev=s.iloc[i - 1], value=s.iloc[i], next=s.iloc[i + 1],
                               temp=d.water_temp_c.iloc[i], ph=d.ph.iloc[i]))
    pd.DataFrame(rows).round(3).to_csv(os.path.join(OUT, "cleaning_summary_train.csv"), index=False)
    pd.DataFrame(spikes).to_csv(os.path.join(OUT, "spike_candidates_train.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
