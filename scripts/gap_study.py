"""Phân bố khoảng trống thật của dataset (docs/02_DATA_CLEANING.md mục 4.1).

Tính trên sáu biến và cả ba tập, trước làm sạch. Ghi vào artifacts/dataset/:
  gap_length_histogram.csv và .png: số khoảng trống và phần thời gian trống theo nhóm độ dài.
  gap_events.csv: mọi khoảng trống từ 2 giờ trở lên, kèm số biến trống suốt khoảng đó (6 là cả đầu đo ngừng ghi,
    1 là trục trặc riêng một cảm biến).
Chạy từ gốc repo: uv run python scripts/gap_study.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import FEATURES as F, NAMES, STATS as OUT, load_raw_split, runs  # noqa: E402

# nhóm độ dài (giờ): (nhãn trong CSV, nhãn trên hình, cận dưới, cận trên], cận dưới loại trừ
BINS = [("15 phut", "15 phút", 0, .25), ("30 phut - 1 gio", "30 phút\nđến 1 giờ", .25, 1),
        ("1 - 2 gio", "1 đến 2\ngiờ", 1, 2), ("2 - 6 gio", "2 đến 6\ngiờ", 2, 6),
        ("6 - 12 gio", "6 đến 12\ngiờ", 6, 12), ("12 - 24 gio", "12 đến 24\ngiờ", 12, 24),
        ("24 - 48 gio", "24 đến 48\ngiờ", 24, 48), ("tren 48 gio", "trên 48\ngiờ", 48, 1e9)]


def gap_lengths():
    """Độ dài (giờ) của mọi khoảng trống thật, mọi biến, cả ba tập."""
    L = []
    for n in NAMES:
        X = load_raw_split(n)[F].to_numpy(float)
        for j in range(len(F)):
            L += list(runs(np.isnan(X[:, j]))[1])
    return np.array(L) / 4


def histogram(h):
    """Số khoảng trống, số ô trống và phần trăm thời gian trống theo nhóm độ dài."""
    rows = []
    for name, _, lo, hi in BINS:
        m = (h > lo) & (h <= hi)
        rows.append(dict(do_dai=name, so_khoang=int(m.sum()), pct_so_khoang=round(100 * m.mean(), 2),
                         so_o_trong=int(round(h[m].sum() * 4)), pct_thoi_gian=round(100 * h[m].sum() / h.sum(), 2)))
    return pd.DataFrame(rows)


def figure(t, path):
    """Hai khung cột cạnh nhau: số khoảng trống (thang log) và phần trăm thời gian trống, cùng nhóm độ dài; lưu PNG."""
    ink, ink2, grid, surf, bar = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb", "#2a78d6"
    labels = [b[1] for b in BINS]
    x = np.arange(len(t))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=150)
    fig.patch.set_facecolor(surf)
    panels = [("so_khoang", "Số khoảng trống (thang log)", True, "{:d}"),
              ("pct_thoi_gian", "Phần trăm thời gian trống", False, "{:.1f}%")]
    for ax, (col, title, log, fmt) in zip(axes, panels):
        v = t[col]
        ax.set_facecolor(surf)
        ax.bar(x, v.where(v > 0, np.nan), width=0.64, color=bar, edgecolor=surf, linewidth=2, zorder=2)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(0.7, 3000)
            ax.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
            ax.minorticks_off()
        else:
            ax.set_ylim(0, v.max() * 1.18)
        for i, val in enumerate(v):
            if val > 0:
                ax.text(i, val * (1.12 if log else 1) + (0 if log else v.max() * 0.015), fmt.format(val),
                        ha="center", va="bottom", fontsize=8.5, color=ink)
        ax.set_xticks(x, labels, fontsize=8, color=ink2)
        ax.set_title(title, fontsize=10.5, color=ink, loc="left", pad=10)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color("#c9c8c3")
        ax.tick_params(axis="y", colors=ink2, labelsize=8.5)
        ax.yaxis.grid(True, color=grid, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    short = t["pct_so_khoang"].iloc[:2].sum()
    long_n, long_pct = t["so_khoang"].iloc[-2:].sum(), t["pct_thoi_gian"].iloc[-2:].sum()
    axes[0].text(0.98, 0.96, f"{short:.0f}% số khoảng trống không quá 1 giờ", transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=9, color=ink2)
    axes[1].text(0.02, 0.96, f"{long_pct:.0f}% thời gian trống thuộc {long_n} khoảng dài hơn 1 ngày",
                 transform=axes[1].transAxes, ha="left", va="top", fontsize=9, color=ink2)
    fig.suptitle(f"Khoảng trống trong dataset MSD theo độ dài ({int(t['so_khoang'].sum())} khoảng, "
                 f"{int(t['so_o_trong'].sum())} ô trống, 6 biến, 2019-01 đến 2025-11)",
                 x=0.01, ha="left", fontsize=11.5, color=ink)
    fig.tight_layout()
    fig.savefig(path, facecolor=surf)


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
    figure(t, os.path.join(OUT, "gap_length_histogram.png"))
    g = gap_events()
    g.to_csv(os.path.join(OUT, "gap_events.csv"), index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
