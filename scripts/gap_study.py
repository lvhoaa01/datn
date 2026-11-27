"""Phân bố khoảng trống thật của dataset (docs/02_DATA_CLEANING.md mục 3.1).

Tính trên sáu biến và cả ba tập, trước làm sạch. Ghi vào artifacts/dataset/:
  gap_length_histogram.csv: số khoảng trống và phần trăm thời gian trống theo nhóm độ dài; cận trên của nhóm tăng gấp
    đôi từ 0.25 giờ (cột can_tren_gio); pct_so_khoang_cong_don là phần trăm số khoảng không dài hơn cận trên.
  fig_khoang_trong.png: hình của bảng trên.
  gap_events.csv: mọi khoảng trống từ 2 giờ trở lên, kèm số biến trống suốt khoảng đó (6 là cả đầu đo ngừng ghi,
    1 là trục trặc riêng một cảm biến).
Chạy từ gốc repo: uv run python scripts/gap_study.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FEATURES as F, NAMES, STATS as OUT, load_raw_split, runs
from plot_style import BLUE, panel, save

# nhóm độ dài (giờ): cận trên tăng gấp đôi từ 0.25 giờ (một mốc); nhóm gồm khoảng dài hơn cận trên của nhóm trước
EDGES = [0.25 * 2 ** i for i in range(10)]          # 0.25, 0.5, 1, ..., 128 giờ


def gap_lengths():
    """Độ dài (giờ) của mọi khoảng trống thật, mọi biến, cả ba tập."""
    L = []
    for n in NAMES:
        X = load_raw_split(n)[F].to_numpy(float)
        for j in range(len(F)):
            L += list(runs(np.isnan(X[:, j]))[1])
    return np.array(L) / 4


def histogram(h):
    """Số khoảng trống, số ô trống, phần trăm thời gian trống theo nhóm độ dài; nhóm i gồm (EDGES[i-1], EDGES[i]]."""
    rows = []
    for i, hi in enumerate(EDGES):
        lo = EDGES[i - 1] if i else 0
        m = (h > lo) & (h <= hi)
        rows.append(dict(can_tren_gio=hi, so_khoang=int(m.sum()), pct_so_khoang=round(100 * m.mean(), 2),
                         pct_so_khoang_cong_don=round(100 * (h <= hi).mean(), 2),
                         so_o_trong=int(round(h[m].sum() * 4)), pct_thoi_gian=round(100 * h[m].sum() / h.sum(), 2)))
    assert h.max() <= EDGES[-1]
    return pd.DataFrame(rows)


def figure(t, path):
    """Hai khung cột chung trục nhóm độ dài: (a) số khoảng trống, thang log; (b) phần trăm thời gian trống; lưu PNG."""
    x = np.arange(len(t))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 4.2), sharex=True)
    a1.bar(x, t["so_khoang"], width=0.65, color=BLUE)
    a1.set_yscale("log")
    a1.set_ylim(0.8, 1000)
    a1.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    a1.minorticks_off()
    a1.set_ylabel("Số khoảng trống")
    a2.bar(x, t["pct_thoi_gian"], width=0.65, color=BLUE)
    a2.set_ylabel("Thời gian trống (%)")
    a2.set_xticks(x, [f"{v:g}" for v in t["can_tren_gio"]])
    a2.set_xlabel("Cận trên của nhóm độ dài khoảng trống (giờ)")
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout()
    save(fig, path)


def gap_events(min_h=2):
    """Mọi khoảng trống thật dài từ min_h giờ, kèm số biến trống suốt khoảng đó (cả ba tập)."""
    d = pd.concat([load_raw_split(n).assign(tap=n) for n in NAMES], ignore_index=True)
    X = d[F].to_numpy(float)
    rows = []
    for j, c in enumerate(F):
        for a, ln in zip(*runs(np.isnan(X[:, j]))):
            if ln / 4 >= min_h:
                rows.append(dict(tap=d["tap"][a], bien=c, bat_dau=str(d["datetime_pst"][a]), gio=ln / 4,
                                 so_bien_cung_trong=int(np.isnan(X[a:a + ln]).all(0).sum())))
    return pd.DataFrame(rows).sort_values(["bat_dau", "bien"])


def main():
    """Ghi bảng tần suất, biểu đồ và danh sách khoảng trống dài."""
    os.makedirs(OUT, exist_ok=True)
    t = histogram(gap_lengths())
    t.to_csv(os.path.join(OUT, "gap_length_histogram.csv"), index=False)
    figure(t, os.path.join(OUT, "fig_khoang_trong.png"))
    g = gap_events()
    g.to_csv(os.path.join(OUT, "gap_events.csv"), index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
