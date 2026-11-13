"""Phan bo do dai khoang trong theo bien va buoc thay doi tu nhien tren tap train (can pandas, numpy).
Chay tu thu muc dataset: python analysis/gap_distribution_study.py
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "results")
F = ["do_mg_l", "water_temp_c", "ph", "spc_us_cm", "turbidity_fnu", "chl_fluor_ug_l"]


def runs(mask):
    m = np.asarray(mask, bool)
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return s, e - s


def main():
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(os.path.join(ROOT, "splits", "train.csv"), na_values=["NA"], parse_dates=["datetime_pst"])
    d = d.set_index("datetime_pst")
    # 1. phan bo do dai khuyet
    rows = []
    bins = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 24), (25, 10**9)]
    for c in F:
        st, ln = runs(d[c].isna().to_numpy())
        r = {"feature": c, "na_cells": int(d[c].isna().sum()), "runs": len(ln)}
        for lo, hi in bins:
            sel = (ln >= lo) & (ln <= hi)
            r[f"runs_{lo}_{hi}"] = int(sel.sum())
            r[f"cells_{lo}_{hi}"] = int(ln[sel].sum())
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "gap_length_distribution_train.csv"), index=False)
    # sai so dao dong tu nhien trong 15 phut (tham chieu): |x(t+1)-x(t)|
    ref = pd.DataFrame([dict(feature=c, step_abs_change_median=d[c].diff().abs().median(),
                             step_abs_change_p95=d[c].diff().abs().quantile(.95), sd=d[c].std()) for c in F])
    ref.to_csv(os.path.join(OUT, "natural_step_change_train.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
