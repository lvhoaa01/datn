"""Muc sai so tham chieu cua hai cau hinh du bao tren base_splits (README.md muc 5.3); can pandas, numpy.

Cung cua so va cung mat na nhu khi danh gia mo hinh: cua so tu common.window_starts, sai so chi tinh tren
moc nhan co DO co 0. Hai cach du bao khong can mo hinh:
  giu_nguyen: moi buoc nhan bang DO o moc cuoi cua dau vao (t - 1).
  cung_gio_hom_qua: buoc t + k bang DO tai t + k - 96 (luon nam trong dau vao vi dau vao toi thieu 96 moc).
R2 = 1 - SSE / SST, SST tinh quanh trung binh DO that tren cac moc duoc tinh sai so cua tap.
Ghi analysis/results/baselines.csv. Chay tu thu muc dataset: python analysis/baselines.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from common import CONFIGS, NAMES, load_base_all, window_starts  # noqa: E402

OUT = os.path.join(ROOT, "analysis", "results")
DAY = 96


def main():
    d = load_base_all()
    do = d["do_mg_l"].to_numpy()
    obs = d["flag_do_mg_l"].to_numpy() == 0
    rows = []
    for cfg, (n_in, n_out) in CONFIGS.items():
        starts = window_starts(d, n_in, n_out)
        for n in NAMES:
            t = starts[n]
            idx = t[:, None] + np.arange(n_out)
            y, m = do[idx], obs[idx]
            preds = {"giu_nguyen": np.repeat(do[t - 1][:, None], n_out, axis=1), "cung_gio_hom_qua": do[idx - DAY]}
            sst = ((y[m] - y[m].mean()) ** 2).sum()
            for meth, p in preds.items():
                e = (p - y)[m]
                rows.append(dict(cau_hinh=cfg, tap=n, phuong_phap=meth, so_cua_so=len(t), so_moc_tinh=int(m.sum()),
                                 MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), R2=1 - (e ** 2).sum() / sst))
    r = pd.DataFrame(rows).round(3)
    r.to_csv(os.path.join(OUT, "baselines.csv"), index=False)
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
