"""Tạo dataset/base_splits/: ba tập train, val, test đã làm sạch, kèm cột cờ cho từng biến.

Quy trình (dataset/LAM_SACH.md), áp dụng riêng từng tập, không dùng giá trị của tập khác:
  1. Giá trị ngoài miền vật lý bị loại (cờ 2).
  2. Chỉ train: điểm DO đảo chiều đơn lẻ không kèm biến đổi nhiệt độ hoặc pH bị loại (cờ 2).
  3. Khoảng trống không quá 8 giờ và có số đo ở hai đầu: nội suy tuyến tính (cờ 1, giữ cờ 2 nếu đã có).
     Khoảng trống dài hơn, hoặc chạm đầu, cuối tập: để NA (cờ 3).
Cờ 0 là số đo thật, giữ nguyên văn chuỗi nguồn.
Chạy từ gốc repo: uv run python scripts/build_base_splits.py
"""
import csv
import os

import numpy as np

from common import BASE, FEATURES, FLAGS, MAX_GAP, NAMES, RANGES, read_raw_split

SPIKE_MG_L = 0.8          # bước vào và bước ra tối thiểu của điểm DO đảo chiều, mg/L
COMOVE_TEMP_C = 0.3       # nhiệt độ đổi cùng mốc từ 0.3 độ C: khối nước khác đi qua, là hiện tượng thật
COMOVE_PH = 0.05          # tương tự, ngưỡng thay đổi pH cùng mốc
OBSERVED, INTERP, REJECTED, LONG_GAP = 0, 1, 2, 3


def do_spikes(X):
    """Chỉ số các mốc DO lệch quá SPIKE_MG_L so với cả hai mốc kề, cùng chiều lệch (đỉnh hoặc đáy nhọn),
    và không có thay đổi nhiệt độ hoặc pH đi kèm."""
    do, tc, ph = X[:, 0], X[:, 1], X[:, 2]
    i, o = do[1:-1] - do[:-2], do[1:-1] - do[2:]
    rev = (np.abs(i) > SPIKE_MG_L) & (np.abs(o) > SPIKE_MG_L) & (i * o > 0)
    comove = (np.abs(tc[1:-1] - tc[:-2]) >= COMOVE_TEMP_C) | (np.abs(ph[1:-1] - ph[:-2]) >= COMOVE_PH + 1e-9)
    return np.flatnonzero(rev & ~comove) + 1


def runs(mask):
    """Vị trí bắt đầu và độ dài các đoạn True liên tiếp trong mảng boolean."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    s = np.flatnonzero(d == 1)
    return s, np.flatnonzero(d == -1) - s


def clean(X, is_train):
    """Làm sạch một tập (mảng n x 6, NaN là trống) theo ba bước ở đầu file; trả về (giá trị, cờ)."""
    X = X.copy()
    flag = np.where(np.isnan(X), INTERP, OBSERVED).astype(np.int8)
    for j, c in enumerate(FEATURES):
        lo, hi = RANGES[c]
        bad = (X[:, j] < lo) | (X[:, j] > hi)
        X[bad, j], flag[bad, j] = np.nan, REJECTED
    if is_train:
        k = do_spikes(X)
        X[k, 0], flag[k, 0] = np.nan, REJECTED
    n = len(X)
    for j in range(X.shape[1]):
        for a, ln in zip(*runs(np.isnan(X[:, j]))):
            if ln <= MAX_GAP and a > 0 and a + ln < n:
                X[a:a + ln, j] = np.interp(np.arange(a, a + ln), [a - 1, a + ln], [X[a - 1, j], X[a + ln, j]])
            else:
                flag[a:a + ln, j] = LONG_GAP
    return X, flag


def main():
    """Ghi ba file base_splits/; ô cờ 0 chép nguyên chuỗi nguồn, ô cờ 1, 2 làm tròn 4 chữ số, ô cờ 3 ghi NA."""
    os.makedirs(BASE, exist_ok=True)
    for name in NAMES:
        rows = read_raw_split(name)
        raw = [v for _, v in rows]
        X = np.array([[np.nan if s == "NA" else float(s) for s in v] for v in raw])
        Y, flag = clean(X, name == "train")
        with open(os.path.join(BASE, f"{name}.csv"), "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["datetime_pst"] + FEATURES + FLAGS)
            for (t, v), y, fl in zip(rows, Y, flag):
                vals = [v[j] if fl[j] == OBSERVED else ("NA" if fl[j] == LONG_GAP else repr(round(float(y[j]), 4)))
                        for j in range(len(FEATURES))]
                w.writerow([t] + vals + [int(x) for x in fl])
        cnt = {k: int((flag == k).sum()) for k in (INTERP, REJECTED, LONG_GAP)}
        print(f"{name}: {len(rows)} dòng | nội suy {cnt[INTERP]} | loại rồi điền {cnt[REJECTED]} | trống dài {cnt[LONG_GAP]}")


if __name__ == "__main__":
    main()
