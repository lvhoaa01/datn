"""Tạo dataset/base_splits/: ba tập train, val, test đã làm sạch, kèm cột cờ cho từng biến.

Quy trình (docs/02_DATA_CLEANING.md), nhân quả: mỗi ô chỉ phụ thuộc số đo tại chính nó và trước nó.
  1. Giá trị ngoài miền vật lý RANGES bị loại, coi như ô trống.
  2. Mọi ô trống được điền bằng số đo hợp lệ gần nhất phía trước (ffill), không giới hạn độ dài khoảng trống; chuỗi
     chạy liền mạch qua ranh giới train, val, test vì tập sau được dùng quá khứ của tập trước. Nếu ô trống nằm ngay
     đầu chuỗi (chưa có số đo nào phía trước) thì báo lỗi; dataset MSD không có trường hợp này.
Cờ: 0 số đo thật (chép nguyên văn chuỗi nguồn); 1 ô được điền (chép nguyên văn chuỗi của số đo nguồn).
Chạy từ gốc repo: uv run python scripts/build_base_splits.py
"""
import csv
import os

import numpy as np
import pandas as pd

from common import BASE, FEATURES, FILLED, FLAGS, NAMES, OBSERVED, RANGES, read_raw_split


def source_index(valid):
    """Với mỗi ô: chỉ số của số đo hợp lệ gần nhất phía trước (chính nó nếu hợp lệ); lỗi nếu ô đầu chuỗi trống."""
    idx = pd.Series(np.where(valid, np.arange(len(valid)), np.nan)).ffill()
    if idx.isna().any():
        raise ValueError(f"{int(idx.isna().sum())} ô trống ở đầu chuỗi, chưa có số đo nào phía trước để chép")
    return idx.to_numpy(int)


def main():
    """Làm sạch chuỗi nối ba tập, ghi ba file base_splits/ và in số ô theo cờ."""
    os.makedirs(BASE, exist_ok=True)
    parts = [read_raw_split(n) for n in NAMES]
    rows = [r for p in parts for r in p]
    S = np.array([v for _, v in rows], dtype=object)            # chuỗi nguyên văn, n x 6
    out, flag = np.empty(S.shape, dtype=object), np.zeros(S.shape, np.int8)
    for j, c in enumerate(FEATURES):
        lo, hi = RANGES[c]
        valid = np.array([s != "NA" and lo <= float(s) <= hi for s in S[:, j]])
        out[:, j] = S[source_index(valid), j]
        flag[:, j] = np.where(valid, OBSERVED, FILLED)
    a = 0
    for name, p in zip(NAMES, parts):
        b = a + len(p)
        with open(os.path.join(BASE, f"{name}.csv"), "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["datetime_pst"] + FEATURES + FLAGS)
            for (t, _), v, fl in zip(p, out[a:b], flag[a:b]):
                w.writerow([t] + list(v) + [int(x) for x in fl])
        fl = flag[a:b]
        print(f"{name}: {b - a} dòng | điền {int((fl == FILLED).sum())} ô")
        a = b


if __name__ == "__main__":
    main()
