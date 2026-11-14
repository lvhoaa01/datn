"""Tao base_splits/: ba tap train, val, test da lam sach, kem cot co cho tung bien (can numpy).

Quy trinh (LAM_SACH.md), ap dung rieng tung tap, khong dung gia tri cua tap khac:
  1. Gia tri ngoai mien vat ly bi loai (co 2).
  2. Chi train: diem DO dao chieu don le khong kem bien doi nhiet do hoac pH bi loai (co 2).
  3. Khoang trong khong qua 8 gio va co so do o hai dau: noi suy tuyen tinh (co 1, giu co 2 neu da co).
     Khoang trong dai hon, hoac cham dau, cuoi tap: de NA (co 3).
Co 0 la so do that, giu nguyen van chuoi nguon.
Chay tu thu muc dataset: python scripts/build_base.py
"""
import csv
import os

import numpy as np

from common import BASE, FEATURES, FLAGS, MAX_GAP, NAMES, RANGES, read_raw_split

SPIKE_MG_L = 0.8          # buoc vao va ra toi thieu cua diem DO dao chieu
COMOVE_TEMP_C = 0.3       # bien doi nhiet do cung moc cho thay khoi nuoc khac (hien tuong that)
COMOVE_PH = 0.05
OBSERVED, INTERP, REJECTED, LONG_GAP = 0, 1, 2, 3


def do_spikes(X):
    """Chi so cac moc DO lech qua SPIKE_MG_L so voi ca hai moc ke, nguoc dau, khong dong bien nhiet do hoac pH."""
    do, tc, ph = X[:, 0], X[:, 1], X[:, 2]
    i, o = do[1:-1] - do[:-2], do[1:-1] - do[2:]
    rev = (np.abs(i) > SPIKE_MG_L) & (np.abs(o) > SPIKE_MG_L) & (i * o > 0)
    comove = (np.abs(tc[1:-1] - tc[:-2]) >= COMOVE_TEMP_C) | (np.abs(ph[1:-1] - ph[:-2]) >= COMOVE_PH + 1e-9)
    return np.flatnonzero(rev & ~comove) + 1


def runs(mask):
    """Vi tri bat dau va do dai cac doan True lien tiep."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    s = np.flatnonzero(d == 1)
    return s, np.flatnonzero(d == -1) - s


def clean(X, is_train):
    """Tra ve (gia tri da lam sach, co); NaN la trong."""
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
        print(f"{name}: {len(rows)} dong | noi suy {cnt[INTERP]} | loai roi dien {cnt[REJECTED]} | trong dai {cnt[LONG_GAP]}")


if __name__ == "__main__":
    main()
