"""Mức sai số tham chiếu của hai cấu hình dự báo trên base_splits (dataset/README.md mục 5.3).

Cùng cửa sổ và cùng mặt nạ như khi đánh giá mô hình: cửa sổ từ common.window_starts, sai số chỉ tính trên
mốc nhãn có DO cờ 0. Hai cách dự báo không cần mô hình:
  giu_nguyen: mọi bước nhãn bằng DO ở mốc cuối của đầu vào (t - 1).
  cung_gio_hom_qua: bước t + k bằng DO tại t + k - 96 (luôn nằm trong đầu vào vì đầu vào tối thiểu 96 mốc).
R2 = 1 - SSE / SST, SST tính quanh trung bình DO thật trên các mốc được tính sai số của tập.
Ghi artifacts/dataset/baselines.csv. Chạy từ gốc repo: uv run python scripts/baselines.py
"""
import os

import numpy as np
import pandas as pd

from common import CONFIGS, NAMES, STATS as OUT, load_base_all, window_starts

DAY = 96                  # số mốc 15 phút trong một ngày


def main():
    """Tính MAE, RMSE, R2 của hai cách cho mọi cấu hình và tập, ghi baselines.csv."""
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
