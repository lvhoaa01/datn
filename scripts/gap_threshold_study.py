"""Căn cứ chọn ngưỡng 8 giờ giữa khoảng trống được nội suy và khoảng trống bị cắt (dataset/LAM_SACH.md mục 5.2, 5.4).
Chạy từ gốc repo: uv run python scripts/gap_threshold_study.py

Phần 0. Bảng tần suất độ dài khoảng trống thật (mọi biến, cả ba tập): gap_length_histogram.csv.
Phần 1. Điểm chuyển: nội suy tuyến tính còn tốt nhất đến độ dài nào?
  Trên toàn bộ train, xóa nhân tạo cả 6 biến cùng lúc (như khi đầu đo ngừng ghi) với độ dài 3 đến 16 giờ,
  so nội suy tuyến tính với các cách điền theo nhịp ngày đêm. Khoảng tin cậy 95% của chênh lệch MAE bằng lấy mẫu lại
  (bootstrap, 4000 lần) theo từng khoảng trống. Ghi threshold_crossover.csv.
Phần 2. Độ nhạy: nếu đổi ngưỡng T, số khoảng trống dài và số cửa sổ bị cắt thay đổi thế nào?
  Cắt mọi cửa sổ (đầu vào và nhãn) chạm vào khoảng trống dài hơn T của bất kỳ biến nào. Ghi threshold_sensitivity.csv.
"""
import os

import numpy as np
import pandas as pd

from common import STATS as OUT, load_raw_split
from imputation_benchmark import F, fill_simple

NAMES = ["train", "val", "test"]
# độ dài khoảng trống thử ở phần 1 (giờ), số khoảng mỗi độ dài; các cách điền so với nội suy tuyến tính
LENS_H = [3, 4, 5, 6, 7, 8, 9, 10, 12, 16]
N_GAPS = 400
ALT = ["hom_qua_offset_2_dau", "cung_gio_hom_qua", "hom_qua_offset_1_dau", "giu_so_truoc", "pchip"]
# các ngưỡng T thử ở phần 2 (giờ)
THRESH_H = [2, 4, 6, 7.75, 8, 10, 12, 16, 20, 21]
CONFIGS = {"24h_6h": (96, 24), "7d_24h": (672, 96)}
rng = np.random.default_rng(20261113)


def runs(m):
    """Vị trí bắt đầu và độ dài các đoạn True liên tiếp."""
    d = np.diff(np.concatenate([[0], np.asarray(m, int), [0]]))
    s = np.flatnonzero(d == 1)
    return s, np.flatnonzero(d == -1) - s


def part1(X):
    """Phần 1 trên mảng train X (n x 6): mỗi biến, mỗi độ dài, so MAE nội suy tuyến tính với cách khác tốt nhất."""
    n = len(X)
    cs = np.concatenate([[0], np.cumsum(np.isnan(X).any(1))])
    rows = []
    for h in LENS_H:
        L = int(h * 4)
        cand = np.arange(300, n - L - 20)
        cand = cand[(cs[cand + L + 10] - cs[cand - 300]) == 0]
        used = np.zeros(n, bool)
        st = []
        for a in rng.permutation(cand):
            if not used[a - 1:a + L + 1].any():
                used[a - 1:a + L + 1] = True
                st.append(a)
                if len(st) == N_GAPS:
                    break
        for j, c in enumerate(F):
            v = X[:, j]
            sd = np.nanstd(v)
            err = {m: np.array([np.abs(fill_simple(v, a, L, m) - v[a:a + L]).mean() for a in st])
                   for m in ["noi_thang"] + ALT}
            best_alt = min(ALT, key=lambda m: err[m].mean())
            dlt = err[best_alt] - err["noi_thang"]          # dương: nội suy tuyến tính tốt hơn
            bs = np.array([rng.choice(dlt, len(dlt)).mean() for _ in range(4000)])
            lo, hi = np.quantile(bs, [0.025, 0.975])
            verdict = "noi thang tot hon" if lo > 0 else ("cach khac tot hon" if hi < 0 else "ngang nhau")
            rows.append(dict(bien=c, gio=h, so_khoang=len(st), MAE_noi_thang=err["noi_thang"].mean(),
                             cach_khac_tot_nhat=best_alt, MAE_cach_khac=err[best_alt].mean(),
                             chenh=dlt.mean(), tin_cay_duoi=lo, tin_cay_tren=hi, ket_luan=verdict,
                             pct_sd_noi_thang=100 * err["noi_thang"].mean() / sd))
    return pd.DataFrame(rows)


def part2():
    """Phần 2 trên cả ba tập nguồn: số khoảng trống dài và số cửa sổ bị cắt theo từng ngưỡng; kèm độ dài các khoảng trên 2 giờ."""
    S = [load_raw_split(n) for n in NAMES]
    X = pd.concat(S, ignore_index=True)[F].to_numpy(float)
    n_all = len(X)
    offs = np.cumsum([0] + [len(s) for s in S])
    gaps = [(j, a + offs[si], l) for si in range(3) for j in range(6)
            for a, l in zip(*runs(np.isnan(X[offs[si]:offs[si + 1], j])))]
    cl = np.concatenate([[0], np.cumsum(np.isnan(X[:, 0]))])
    rows = []
    for T in THRESH_H:
        long_gaps = [(j, a, l) for j, a, l in gaps if l > T * 4]
        for cfg, (lin, lout) in CONFIGS.items():
            bad = np.zeros(n_all + 1, bool)
            for j, a, l in long_gaps:
                bad[max(a - lout + 1, 0):min(a + l + lin, n_all)] = True   # đầu vào [t - lin, t) chạm [a, a + l) khi t <= a + l + lin - 1
            r = dict(nguong_gio=T, cau_hinh=cfg, so_khoang_dai=len(long_gaps))
            for si, nm in enumerate(NAMES):
                t = np.arange(max(offs[si], lin), offs[si + 1] - lout + 1)
                ok = ((cl[t + lout] - cl[t]) < lout) & ~bad[t]
                r[f"cat_{nm}"] = int((~ok).sum())
                r[f"pct_{nm}"] = round(100 * (~ok).mean(), 2)
            rows.append(r)
    lens = pd.DataFrame([dict(bien=F[j], gio=l / 4) for j, a, l in gaps if l > 8]).sort_values("gio")
    return pd.DataFrame(rows), lens


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


def main():
    """Chạy phần 0, 1, 2 và ghi ba bảng."""
    os.makedirs(OUT, exist_ok=True)
    d = load_raw_split("train")
    histogram().to_csv(os.path.join(OUT, "gap_length_histogram.csv"), index=False)
    p1 = part1(d[F].to_numpy(float))
    p1.round(4).to_csv(os.path.join(OUT, "threshold_crossover.csv"), index=False)
    p2, lens = part2()
    p2.to_csv(os.path.join(OUT, "threshold_sensitivity.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(p1[["bien", "gio", "MAE_noi_thang", "cach_khac_tot_nhat", "MAE_cach_khac", "chenh", "tin_cay_duoi",
              "tin_cay_tren", "ket_luan"]].round(3).to_string(index=False))
    print(p2.to_string(index=False))
    print("Các khoảng trống thật dài hơn 2 giờ:", lens.gio.tolist())


if __name__ == "__main__":
    main()
