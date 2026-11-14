"""Căn cứ chọn ngưỡng điền 8 giờ (dataset/LAM_SACH.md mục 4.1, 4.3, 4.5).
Chạy từ gốc repo: uv run python scripts/gap_threshold_study.py

Phần 0. Tần suất độ dài khoảng trống thật (mọi biến, cả ba tập): gap_length_histogram.csv và biểu đồ .png.
Phần 1. Độ nhạy theo ngưỡng T, với quy tắc nhân quả của build_base_splits: ô cách số đo hợp lệ gần nhất phía trước
  hơn T giờ thì để trống; cửa sổ chạm ô trống, hoặc có nhãn không còn mốc DO đo thật, bị bỏ. T = 0 là không điền
  ô nào. Ghi threshold_sensitivity.csv (số ô để trống, số cửa sổ bị bỏ theo tập và cấu hình).
Phần 2. Đợt DO thấp bị mất khỏi đánh giá với base_splits: một đợt bị mất khi không mốc nào của nó nằm trong nhãn
  của cửa sổ được giữ. Ghi episode_coverage.csv.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import (CONFIGS, FEATURES as F, NAMES, RANGES, STATS as OUT, load_base_all, load_raw_split, runs,  # noqa: E402
                    window_starts)

THRESH_H = [0, 4, 6, 7.75, 8, 12, 21]                  # ngưỡng thử ở phần 1, giờ; 0 là không điền
EPISODE_MG_L = [6, 5]                                  # ngưỡng đợt DO thấp, mg/L
MIN_STEPS = MERGE_STEPS = 4                            # đợt dài ít nhất 1 giờ; gộp hai đoạn cách nhau không quá 1 giờ


def histogram():
    """Phần 0: số khoảng trống thật theo nhóm độ dài, kèm tỉ lệ tích lũy."""
    L = []
    for nm in NAMES:
        X = load_raw_split(nm)[F].to_numpy(float)
        for j in range(6):
            L += list(runs(np.isnan(X[:, j]))[1])
    h = np.array(L) / 4
    bins = [("15 phut", 0, .25), ("30 phut - 1 gio", .25, 1), ("1 - 2 gio", 1, 2), ("2 - 4 gio", 2, 4),
            ("4 - 8 gio", 4, 8), ("8 - 16 gio", 8, 16), ("16 - 24 gio", 16, 24), ("24 - 48 gio", 24, 48),
            ("tren 48 gio", 48, 1e9)]
    rows, cum = [], 0
    for name, lo, hi in bins:
        k = int(((h > lo) & (h <= hi)).sum())
        cum += k
        rows.append(dict(do_dai=name, so_khoang=k, tich_luy_pct=round(100 * cum / len(h), 2)))
    return pd.DataFrame(rows)


def figure(h, path):
    """Biểu đồ cột (thang log) của bảng tần suất h, kèm đường ngưỡng 8 giờ; lưu PNG."""
    ink, ink2, surf, bar = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6"
    labels = ["15 phút", "30 phút\nđến 1 giờ", "1 đến 2\ngiờ", "2 đến 4\ngiờ", "4 đến 8\ngiờ",
              "8 đến 16\ngiờ", "16 đến 24\ngiờ", "24 đến 48\ngiờ", "trên 48\ngiờ"]
    k = h["so_khoang"]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    fig.patch.set_facecolor(surf)
    ax.set_facecolor(surf)
    ax.bar(range(len(h)), k.where(k > 0, 0.0001), width=0.62, color=bar, edgecolor=surf, linewidth=2)
    ax.set_yscale("log")
    ax.set_ylim(0.7, 25000)
    ax.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    ax.minorticks_off()
    for i, n in enumerate(k):
        ax.text(i, max(n, 0.8) * 1.15, str(n), ha="center", va="bottom", fontsize=9, color=ink)
    ax.axvline(4.5, color=ink2, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(4.6, 1200, "ngưỡng 8 giờ", color=ink2, fontsize=9, ha="left", va="center")
    ax.text(2.3, 20000, "không quá 8 giờ: 99.45%, điền bằng số đo trước", color=ink2, fontsize=9, ha="center", va="top")
    ax.text(7.0, 20000, "dài hơn 8 giờ: 3 sự kiện, để trống sau 8 giờ đầu", color=ink2, fontsize=9, ha="center", va="top")
    ax.set_xticks(range(len(h)), labels, fontsize=8.5, color=ink2)
    ax.set_ylabel("Số khoảng trống (thang log)", fontsize=9, color=ink2)
    ax.set_title(f"Độ dài khoảng trống trong dataset RRI ({k.sum()} khoảng, 6 biến, cả ba tập)", fontsize=10.5,
                 color=ink, loc="left", pad=12)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(axis="y", colors=ink2, labelsize=8.5)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, facecolor=surf)


def steps_since_valid(X):
    """(số mốc từ số đo hợp lệ gần nhất phía trước, mặt nạ số đo hợp lệ) cho từng ô của X (n x 6).

    Hợp lệ là khác NaN và nằm trong RANGES; khoảng cách bằng 0 tại số đo hợp lệ, vô cùng nếu chưa có số đo nào.
    """
    lo, hi = np.array([RANGES[c] for c in F]).T
    valid = (X >= lo) & (X <= hi)                      # so sánh với NaN cho False
    i = np.arange(len(X))[:, None]
    last = np.maximum.accumulate(np.where(valid, i, -1), axis=0)
    return np.where(last >= 0, i - last, np.inf), valid


def sensitivity():
    """Phần 1: số ô để trống và số cửa sổ bị bỏ theo ngưỡng T, cho từng cấu hình và tập."""
    S = [load_raw_split(n) for n in NAMES]
    X = pd.concat(S, ignore_index=True)[F].to_numpy(float)
    offs = np.cumsum([0] + [len(s) for s in S])
    dist, valid = steps_since_valid(X)
    obs = np.concatenate([[0], np.cumsum(valid[:, 0])])
    rows = []
    for T in THRESH_H:
        miss = dist > T * 4
        cm = np.concatenate([[0], np.cumsum(miss.any(1))])
        for cfg, (lin, lout) in CONFIGS.items():
            r = dict(nguong_gio=T, cau_hinh=cfg, so_o_trong=int(miss.sum()))
            for si, nm in enumerate(NAMES):
                t = np.arange(max(offs[si], lin), offs[si + 1] - lout + 1)
                ok = (cm[t + lout] - cm[t - lin] == 0) & (obs[t + lout] - obs[t] > 0)
                r[f"cat_{nm}"] = int((~ok).sum())
                r[f"pct_{nm}"] = round(100 * (~ok).mean(), 2)
            rows.append(r)
    return pd.DataFrame(rows)


def episode_spans(do, thr):
    """Các đợt DO dưới thr: list (bắt đầu, kết thúc) theo chỉ số trong do; NaN coi như không dưới ngưỡng."""
    b = np.nan_to_num(do, nan=np.inf) < thr
    for a, ln in zip(*runs(~b)):
        if 0 < a and a + ln < len(b) and ln <= MERGE_STEPS:
            b[a:a + ln] = True
    return [(a, a + ln) for a, ln in zip(*runs(b)) if ln >= MIN_STEPS]


def episode_coverage():
    """Phần 2: số đợt DO thấp của từng tập và số đợt không nằm trong nhãn của cửa sổ nào được giữ."""
    d = load_base_all()
    do = np.where(d["flag_do_mg_l"].to_numpy() == 0, d["do_mg_l"].to_numpy(), np.nan)
    split = d["split"].to_numpy()
    rows = []
    for cfg, (lin, lout) in CONFIGS.items():
        starts = window_starts(d, lin, lout)
        for nm in NAMES:
            idx = np.flatnonzero(split == nm)
            cov = np.zeros(len(d) + 1, int)
            np.add.at(cov, starts[nm], 1)
            np.add.at(cov, starts[nm] + lout, -1)
            covered = np.cumsum(cov)[:-1] > 0          # mốc nằm trong nhãn của ít nhất một cửa sổ được giữ
            for thr in EPISODE_MG_L:
                sp = [(a + idx[0], b + idx[0]) for a, b in episode_spans(do[idx], thr)]
                lost = [a for a, b in sp if not covered[a:b].any()]
                rows.append(dict(cau_hinh=cfg, tap=nm, nguong_mg_l=thr, so_dot=len(sp), so_dot_mat=len(lost),
                                 dot_mat=" ".join(str(d["datetime_pst"][a])[:16] for a in lost)))
    return pd.DataFrame(rows)


def main():
    """Chạy ba phần, ghi ba bảng và biểu đồ."""
    os.makedirs(OUT, exist_ok=True)
    h = histogram()
    h.to_csv(os.path.join(OUT, "gap_length_histogram.csv"), index=False)
    figure(h, os.path.join(OUT, "gap_length_histogram.png"))
    s = sensitivity()
    s.to_csv(os.path.join(OUT, "threshold_sensitivity.csv"), index=False)
    e = episode_coverage()
    e.to_csv(os.path.join(OUT, "episode_coverage.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(s.to_string(index=False))
    print(e.to_string(index=False))


if __name__ == "__main__":
    main()
