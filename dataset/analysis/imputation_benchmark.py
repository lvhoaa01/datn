"""Benchmark cac phuong phap dien du lieu khuyet tren tap train RRI.

Chay tu thu muc dataset:  python analysis/imputation_benchmark.py [--no-saits]
Can: pandas, numpy, scipy, scikit-learn. SAITS can them torch va pypots (tu bo qua neu thieu).

Cach lam (xoa nhan tao roi dien lai):
1. Doan cham: 15% cuoi cua train (2023-07 den 2024-05). 85% dau chi dung de huan luyen SAITS.
2. Voi moi do dai khoang trong L, chon ngau nhien cac vi tri a sao cho tu a-200 den a+L+150 moc
   deu co du 6 bien (de biet gia tri that va du ngu canh). Cac khoang trong cung do dai khong chong nhau.
3. Xoa [a, a+L) theo hai kieu: A = ca 6 bien cung trong (giong thuc te khi DO trong);
   B = chi mot bien trong, 5 bien kia con (danh gia rieng tung bien).
4. Dien lai bang tung phuong phap, so voi gia tri that da xoa.
   MAE = trung binh |dien - that|; RMSE = can bac hai trung binh binh phuong;
   pct_sd = MAE chia do lech chuan cua bien tren train (de so sanh giua cac bien);
   gay_pct = % so khoang trong co buoc nhay o mot trong hai dau (o dien dau tien so voi so do truoc,
   so do sau so voi o dien cuoi) lon hon phan vi 99 cua buoc thay doi 15 phut tu nhien cua bien do.
Cac phuong phap mot bien (1-6) cho ket qua nhu nhau o kieu A va B vi chi dung chinh bien do.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "results")
F = ["do_mg_l", "water_temp_c", "ph", "spc_us_cm", "turbidity_fnu", "chl_fluor_ug_l"]
LENS = [1, 2, 4, 8, 12, 24, 48, 85, 98]          # so moc 15 phut; 85 va 98 = do dai hai lan bao tri that
N_BY_LEN = {1: 300, 2: 300, 4: 300, 8: 300, 12: 300, 24: 200, 48: 200, 85: 100, 98: 100}
EVAL_FRAC, CTX_L, CTX_R, T_SAITS, DAY = 0.15, 200, 150, 288, 96
SEED = 20261113
METHODS = ["giu_so_truoc", "cung_gio_hom_qua", "hom_qua_offset_1_dau", "noi_thang", "pchip",
           "hom_qua_offset_2_dau", "missforest", "saits"]


def ref_index(i, a, L):
    """O cung gio hom qua; neu o do nam trong khoang trong thi lui them mot ngay."""
    j = i - DAY
    while a <= j < a + L:
        j -= DAY
    return j


def fill_simple(v, a, L, m):
    idx = np.arange(a, a + L)
    left, right = a - 1, a + L
    if m == "giu_so_truoc":
        return np.full(L, v[left])
    if m == "noi_thang":
        return np.interp(idx, [left, right], [v[left], v[right]])
    if m == "pchip":
        k = np.r_[np.arange(left - 7, left + 1), np.arange(right, right + 8)]
        return PchipInterpolator(k, v[k])(idx)
    ref = np.array([v[ref_index(i, a, L)] for i in idx])
    if m == "cung_gio_hom_qua":
        return ref
    o0 = v[left] - v[ref_index(left, a, L)]
    if m == "hom_qua_offset_1_dau":
        return ref + o0
    if m == "hom_qua_offset_2_dau":
        o1 = v[right] - v[ref_index(right, a, L)]
        return ref + o0 + (o1 - o0) * (idx - left) / (right - left)
    raise ValueError(m)


def pick_gaps(X, cut, rng):
    n = len(X)
    bad = np.isnan(X).any(1)
    cs = np.concatenate([[0], np.cumsum(bad)])
    gaps = {}
    for L in LENS:
        cand = np.arange(cut + CTX_L, n - L - CTX_R)
        cand = cand[(cs[cand + L + CTX_R] - cs[cand - CTX_L]) == 0]
        used = np.zeros(n, bool)
        sel = []
        for a in rng.permutation(cand):
            if not used[a - 1:a + L + 1].any():
                used[a - 1:a + L + 1] = True
                sel.append(int(a))
                if len(sel) == N_BY_LEN[L]:
                    break
        gaps[L] = sorted(sel)
    return gaps


def to_model_space(X):
    Z = X.copy()
    Z[:, 4:] = np.log1p(np.clip(Z[:, 4:], 0, None))
    return Z


def from_model_space(Z):
    X = Z.copy()
    X[:, 4:] = np.expm1(X[:, 4:])
    return X


def run_missforest(X, times, gaps, L, pattern, j_only=None):
    """MissForest (IterativeImputer + ExtraTrees), them dac trung gio trong ngay va ngay trong nam."""
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import ExtraTreesRegressor
    lo = int(len(X) * 0.70)                         # du lieu dung de khop: 30% cuoi train
    M = to_model_space(X[lo:])
    for a in gaps:
        if pattern == "A":
            M[a - lo:a - lo + L] = np.nan
        else:
            M[a - lo:a - lo + L, j_only] = np.nan
    hr = (times.dt.hour * 4 + times.dt.minute // 15).to_numpy()[lo:] / DAY * 2 * np.pi
    dy = times.dt.day_of_year.to_numpy()[lo:] / 365.25 * 2 * np.pi
    Zin = np.column_stack([M, np.sin(hr), np.cos(hr), np.sin(dy), np.cos(dy)])
    imp = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=50, min_samples_leaf=3, n_jobs=-1,
                                                         random_state=0), max_iter=5, random_state=0)
    out = from_model_space(imp.fit_transform(Zin)[:, :6])
    full = np.full_like(X, np.nan)
    full[lo:] = out
    return full


def train_saits(Z, cut):
    import torch
    from pypots.imputation import SAITS
    torch.manual_seed(0)
    np.random.seed(0)
    tr_end = int(cut * 0.9)
    tr = np.stack([Z[s:s + T_SAITS] for s in range(0, tr_end - T_SAITS, 24)])
    va = np.stack([Z[s:s + T_SAITS] for s in range(tr_end, cut - T_SAITS, 96)])
    va_in = va.copy()
    rng = np.random.default_rng(1)
    for w in va_in:                                  # mat na kiem dinh: khoi ca 6 bien, nhieu do dai
        L = rng.choice([1, 4, 8, 24, 48, 96])
        a = rng.integers(48, T_SAITS - 48 - L)
        w[a:a + L] = np.nan
    model = SAITS(n_steps=T_SAITS, n_features=6, n_layers=3, d_model=256, n_heads=4, d_k=64, d_v=64,
                  d_ffn=512, dropout=0.1, batch_size=64, epochs=300, patience=20,
                  device="cuda" if torch.cuda.is_available() else "cpu", verbose=False)
    model.fit({"X": tr}, {"X": va_in, "X_ori": va})
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-saits", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(os.path.join(ROOT, "splits", "train.csv"), na_values=["NA"], parse_dates=["datetime_pst"])
    X, times = d[F].to_numpy(float), d["datetime_pst"]
    n = len(X)
    cut = int(n * (1 - EVAL_FRAC))
    rng = np.random.default_rng(SEED)
    gaps = pick_gaps(X, cut, rng)
    sd = np.nanstd(X, 0)
    p99 = np.array([np.nanquantile(np.abs(np.diff(X[:, j])), 0.99) for j in range(6)])

    use_saits = not args.no_saits
    if use_saits:
        try:
            Zall = to_model_space(X)
            mu, sg = np.nanmean(Zall[:cut], 0), np.nanstd(Zall[:cut], 0)
            Zn = (Zall - mu) / sg
            t0 = time.time()
            saits = train_saits(Zn, cut)
            print(f"SAITS huan luyen xong sau {time.time() - t0:.0f} giay")
        except ImportError:
            print("Khong co torch/pypots: bo qua SAITS")
            use_saits = False

    rows = []

    def score(j, L, pattern, method, fills, starts):
        e = np.concatenate([fills[k] - X[a:a + L, j] for k, a in enumerate(starts)])
        jumps = [max(abs(fills[k][0] - X[a - 1, j]), abs(X[a + L, j] - fills[k][-1])) for k, a in enumerate(starts)]
        rows.append(dict(bien=F[j], kieu=pattern, so_moc=L, gio=L / 4, phuong_phap=method, so_khoang=len(starts),
                         MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), pct_sd=100 * np.abs(e).mean() / sd[j],
                         gay_pct=100 * np.mean(np.array(jumps) > p99[j])))

    for L in LENS:
        st = gaps[L]
        t0 = time.time()
        for j in range(6):                             # phuong phap mot bien: dung chung cho kieu A va B
            for m in METHODS[:6]:
                fills = [fill_simple(X[:, j], a, L, m) for a in st]
                for pattern in ("A", "B"):
                    score(j, L, pattern, m, fills, st)
        mfA = run_missforest(X, times, st, L, "A")
        for j in range(6):
            score(j, L, "A", "missforest", [mfA[a:a + L, j] for a in st], st)
            mfB = run_missforest(X, times, st, L, "B", j_only=j)
            score(j, L, "B", "missforest", [mfB[a:a + L, j] for a in st], st)
        if use_saits:
            for pattern in ("A", "B"):
                for j in range(6):
                    if pattern == "A" and j > 0:
                        break
                    W, off = [], (T_SAITS - L) // 2
                    for a in st:
                        w = Zn[a - off:a - off + T_SAITS].copy()
                        if pattern == "A":
                            w[off:off + L] = np.nan
                        else:
                            w[off:off + L, j] = np.nan
                        W.append(w)
                    P = saits.predict({"X": np.stack(W)})["imputation"] * sg + mu
                    P = np.stack([from_model_space(p) for p in P])
                    cols = range(6) if pattern == "A" else [j]
                    for jj in cols:
                        score(jj, L, pattern, "saits", [p[off:off + L, jj] for p in P], st)
        print(f"L = {L / 4:5.2f} gio: {len(st)} khoang, {time.time() - t0:.0f} giay")

    r = pd.DataFrame(rows)
    for c in ["MAE", "RMSE", "pct_sd", "gay_pct"]:
        r[c] = r[c].round(4 if c in ("MAE", "RMSE") else 2)
    r.to_csv(os.path.join(OUT, "imputation_benchmark.csv"), index=False)
    pd.DataFrame([dict(so_moc=L, bat_dau=str(times[a])) for L in LENS for a in gaps[L]]).to_csv(
        os.path.join(OUT, "imputation_benchmark_gaps.csv"), index=False)
    pd.DataFrame(dict(bien=F, do_lech_chuan=sd, buoc_15p_p99=p99)).round(4).to_csv(
        os.path.join(OUT, "imputation_benchmark_reference.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    for pattern in ("A", "B"):
        q = r[(r.kieu == pattern) & (r.bien == "do_mg_l")]
        print(f"\nDO, kieu {pattern}, MAE (mg/L)")
        print(q.pivot_table(index="gio", columns="phuong_phap", values="MAE")[[m for m in METHODS if m in set(q.phuong_phap)]].round(3))
    print("done")


if __name__ == "__main__":
    main()
