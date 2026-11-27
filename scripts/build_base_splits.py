"""Tạo dataset/base_splits/: ba tập train, val, test đã làm sạch, kèm cột cờ cho từng biến.

Quy trình (docs/02_DATA_CLEANING.md), nhân quả: mỗi ô chỉ phụ thuộc số đo tại chính nó và trước nó.
  1. Giá trị ngoài miền vật lý RANGES bị loại, coi như ô trống.
  2. Mọi ô trống được điền bằng số đo hợp lệ gần nhất phía trước (ffill), không giới hạn độ dài khoảng trống; chuỗi
     chạy liền mạch qua ranh giới train, val, test vì tập sau được dùng quá khứ của tập trước. Nếu ô trống nằm ngay
     đầu chuỗi (chưa có số đo nào phía trước) thì báo lỗi; dataset MSD không có trường hợp này.
Cờ: 0 số đo thật (chép nguyên văn chuỗi nguồn); 1 ô được điền (chép nguyên văn chuỗi của số đo nguồn).
Ghi thêm artifacts/dataset/split_counts.csv: theo tập, số dòng, số ô, số ô được điền và phần trăm, số cửa sổ có thể
tạo và số cửa sổ bị bỏ của từng cấu hình (common.window_starts); dùng ở docs/02 Bảng 2 và docs/04 Bảng 3.
Chạy từ gốc repo: uv run python scripts/build_base_splits.py
"""
import csv
import os

import numpy as np
import pandas as pd

from common import (BASE, CONFIGS, FEATURES, FILLED, FLAGS, NAMES, OBSERVED, RANGES, STATS, load_base_all,
                    read_raw_split, window_starts)


def source_index(valid):
    """Với mỗi ô: chỉ số của số đo hợp lệ gần nhất phía trước (chính nó nếu hợp lệ); lỗi nếu ô đầu chuỗi trống."""
    idx = pd.Series(np.where(valid, np.arange(len(valid)), np.nan)).ffill()
    if idx.isna().any():
        raise ValueError(f"{int(idx.isna().sum())} ô trống ở đầu chuỗi, chưa có số đo nào phía trước để chép")
    return idx.to_numpy(int)


def main():
    """Làm sạch chuỗi nối ba tập, ghi ba file base_splits/ và in số ô theo cờ."""
    os.makedirs(BASE, exist_ok=True)
    os.makedirs(STATS, exist_ok=True)
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
    split_counts()


def split_counts():
    """Ghi split_counts.csv: số ô được điền và số cửa sổ (có thể tạo, bị bỏ) của từng tập, đọc lại từ base_splits/."""
    d = load_base_all()
    rows = []
    for n in NAMES:
        x = d[d["split"] == n]
        filled = int((x[FLAGS].to_numpy() == FILLED).sum())
        r = dict(tap=n, so_dong=len(x), so_o=len(x) * len(FEATURES), o_dien=filled,
                 pct_o_dien=round(100 * filled / (len(x) * len(FEATURES)), 6))
        rows.append(r)
    for cfg, (n_in, n_out) in CONFIGS.items():
        kept = window_starts(d, n_in, n_out)
        for r in rows:
            idx = d.index[d["split"] == r["tap"]]
            total = idx[-1] + 2 - n_out - max(idx[0], n_in)       # mọi t có nhãn nằm trọn trong tập
            r[f"cua_so_{cfg}"], r[f"bo_{cfg}"] = int(total), int(total - len(kept[r["tap"]]))
    pd.DataFrame(rows).to_csv(os.path.join(STATS, "split_counts.csv"), index=False)


if __name__ == "__main__":
    main()
