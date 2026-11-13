"""Ve bieu do tan suat do dai khoang trong (can pandas, matplotlib).
Chay tu thu muc dataset, sau analysis/threshold_study.py: python analysis/gap_histogram_figure.py
Doc analysis/results/gap_length_histogram.csv, ghi analysis/results/gap_length_histogram.png.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "analysis", "results")
INK, INK2, SURF, BAR = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6"
LABELS = ["15 phút", "30 phút\nđến 1 giờ", "1 đến 2\ngiờ", "2 đến 4\ngiờ", "4 đến 8\ngiờ",
          "8 đến 16\ngiờ", "16 đến 24\ngiờ", "24 đến 48\ngiờ", "trên 48\ngiờ"]


def main():
    h = pd.read_csv(os.path.join(RES, "gap_length_histogram.csv"))
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    x = range(len(h))
    counts = h["so_khoang"].clip(lower=0)
    ax.bar(x, counts.where(counts > 0, 0.0001), width=0.62, color=BAR, edgecolor=SURF, linewidth=2)
    ax.set_yscale("log")
    ax.set_ylim(0.7, 25000)
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["1", "10", "100", "1000"])
    ax.minorticks_off()
    for i, k in enumerate(h["so_khoang"]):
        ax.text(i, max(k, 0.8) * 1.15, str(k), ha="center", va="bottom", fontsize=9, color=INK)
    ax.axvline(4.5, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(4.6, 1200, "ngưỡng 8 giờ", color=INK2, fontsize=9, ha="left", va="center")
    ax.text(2.3, 20000, "không quá 8 giờ: 99.45%, nội suy tuyến tính", color=INK2, fontsize=9, ha="center", va="top")
    ax.text(7.0, 20000, "dài hơn 8 giờ: 3 sự kiện, cắt bỏ", color=INK2, fontsize=9, ha="center", va="top")
    ax.set_xticks(list(x))
    ax.set_xticklabels(LABELS, fontsize=8.5, color=INK2)
    ax.set_ylabel("Số khoảng trống (thang log)", fontsize=9, color=INK2)
    ax.set_title("Độ dài khoảng trống trong dataset RRI (2377 khoảng, 6 biến, cả ba tập)", fontsize=10.5,
                 color=INK, loc="left", pad=12)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(axis="y", colors=INK2, labelsize=8.5)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "gap_length_histogram.png"), facecolor=SURF)
    print("done")


if __name__ == "__main__":
    main()
