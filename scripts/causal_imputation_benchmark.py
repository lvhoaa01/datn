"""So sánh các phương pháp điền khuyết nhân quả trên train RRI (dataset/LAM_SACH.md mục 4.2, 4.3).

Chạy từ gốc repo: uv run python scripts/causal_imputation_benchmark.py

Hàm điền chỉ nhận h = chuỗi trước khoảng trống (h[-1] là số đo cuối) và k = 1..L bước kể từ số đo cuối; không biết
độ dài L, giống khi chạy thật. ref(k) là giá trị cùng giờ hôm qua, o = h[-1] - ref(0) là độ lệch tại số đo cuối.
  ffill:             h[-1]
  sn1:               ref(k)
  sn1_offset:        ref(k) + o
  sn1_offset_decay:  ref(k) + o exp(-k / tau)
  ffill_to_sn1:      w h[-1] + (1 - w)(ref(k) + o), w = exp(-k / tau)
  linear:            nội suy tuyến tính, dùng số đo sau khoảng trống; chỉ để tham khảo, không nhân quả.
Thiết kế: tau chọn theo từng biến trên 85% đầu train, chấm trên 15% cuối. Khoảng trống nhân tạo dài 1, 2, 4, 8, 16,
32 mốc (N_GAPS mỗi độ dài), đặt ngẫu nhiên nơi ngày hôm trước và mốc ngay sau đều đủ số đo, không chồng nhau.
Sai số tổng hợp: MAE trung bình theo tỉ lệ ô trống thật (khoảng trống không quá 8 giờ, mọi biến, cả ba tập) ở từng
độ dài. Khoảng tin cậy 95% của chênh lệch so với phương pháp nhân quả tốt nhất: bootstrap ghép cặp, lấy mẫu lại
khoảng trống trong từng độ dài.
Ghi vào artifacts/dataset/: causal_imputation_benchmark.csv (MAE theo biến, phương pháp, độ dài),
causal_imputation_summary.csv (tổng hợp, tau, khoảng tin cậy), ffill_by_length.csv (MAE của ffill với khoảng trống
từ 15 phút đến 24 giờ trên toàn train).
"""
import os

import numpy as np
import pandas as pd

from common import FEATURES as F, MAX_GAP, NAMES, STATS as OUT, load_raw_split, runs

DAY = 96
LENS = [1, 2, 4, 8, 16, 32]              # độ dài khoảng trống, mốc 15 phút
FFILL_LENS = [1, 4, 32, 48, 84, 96]      # 15 phút, 1, 8, 12, 21, 24 giờ
N_GAPS, N_BOOT = 400, 2000
TAUS = [1, 2, 4, 8, 16, 32, 64, 96, 192]
TUNE_FRAC = 0.85
SEED = 20261114
FIXED = ["ffill", "sn1", "sn1_offset"]
TUNED = ["sn1_offset_decay", "ffill_to_sn1"]


def fill(h, L, method, tau=None):
    """Điền L ô sau h bằng method; chỉ dùng h và k = 1..L (k <= DAY nên ref luôn nằm trong h)."""
    k = np.arange(1, L + 1)
    if method == "ffill":
        return np.full(L, h[-1])
    ref = h[len(h) - 1 + k - DAY]
    if method == "sn1":
        return ref
    o = h[-1] - h[len(h) - 1 - DAY]
    if method == "sn1_offset":
        return ref + o
    w = np.exp(-k / tau)
    if method == "sn1_offset_decay":
        return ref + o * w
    if method == "ffill_to_sn1":
        return w * h[-1] + (1 - w) * (ref + o)
    raise ValueError(method)


def pick(X, lo, hi, L, rng, n=N_GAPS):
    """Chọn n vị trí a trong [lo, hi) sao cho [a - DAY - 1, a + L] đủ 6 biến; các khoảng không chồng nhau."""
    cs = np.concatenate([[0], np.cumsum(np.isnan(X).any(1))])
    cand = np.arange(lo + DAY + 1, hi - L - 1)
    cand = cand[cs[cand + L + 1] - cs[cand - DAY - 1] == 0]
    used = np.zeros(len(X) + 1, bool)
    out = []
    for a in rng.permutation(cand):
        if not used[a - 1:a + L + 1].any():
            used[a - 1:a + L + 1] = True
            out.append(int(a))
            if len(out) == n:
                break
    return out


def errors(v, gaps, method, tau=None):
    """{L: mảng MAE từng khoảng trống} của method trên chuỗi một biến v."""
    out = {}
    for L, st in gaps.items():
        if method == "linear":
            f = [np.interp(np.arange(a, a + L), [a - 1, a + L], [v[a - 1], v[a + L]]) for a in st]
        else:
            f = [fill(v[:a], L, method, tau) for a in st]
        out[L] = np.array([np.abs(fi - v[a:a + L]).mean() for fi, a in zip(f, st)])
    return out


def cell_weights():
    """Tỉ lệ ô trống thật theo độ dài LENS; khoảng trống không quá MAX_GAP mốc, gán vào độ dài LENS gần nhất phía trên."""
    ls = []
    for n in NAMES:
        Y = load_raw_split(n)[F].to_numpy(float)
        for j in range(len(F)):
            ls += [x for x in runs(np.isnan(Y[:, j]))[1] if x <= MAX_GAP]
    ls = np.array(ls)
    b = np.searchsorted(LENS, ls)
    return {L: float(ls[b == i].sum() / ls.sum()) for i, L in enumerate(LENS)}


def main():
    """Chọn tau trên đoạn chỉnh, chấm mọi phương pháp trên đoạn chấm, đo ffill theo độ dài, ghi ba bảng."""
    os.makedirs(OUT, exist_ok=True)
    X = load_raw_split("train")[F].to_numpy(float)
    cut = int(len(X) * TUNE_FRAC)
    rng = np.random.default_rng(SEED)
    tune = {L: pick(X, 0, cut, L, rng) for L in LENS}
    test = {L: pick(X, cut, len(X), L, rng) for L in LENS}
    w = cell_weights()

    def score(e, idx=None):
        """MAE tổng hợp theo trọng số w; idx là chỉ số lấy mẫu lại cho bootstrap."""
        return sum(w[L] * (e[L] if idx is None else e[L][idx[L]]).mean() for L in LENS)

    rows, summ = [], []
    for j, c in enumerate(F):
        v = X[:, j]
        tau = {m: min(TAUS, key=lambda t: score(errors(v, tune, m, t))) for m in TUNED}
        E = {m: errors(v, test, m, tau.get(m)) for m in FIXED + TUNED + ["linear"]}
        causal = FIXED + TUNED
        best = min(causal, key=lambda m: score(E[m]))
        diff = {m: [] for m in causal}
        for _ in range(N_BOOT):
            idx = {L: rng.integers(0, len(test[L]), len(test[L])) for L in LENS}
            sb = score(E[best], idx)
            for m in causal:
                diff[m].append(score(E[m], idx) - sb)
        for m, e in E.items():
            rows += [dict(bien=c, phuong_phap=m, so_moc=L, gio=L / 4, so_khoang=len(test[L]), MAE=e[L].mean())
                     for L in LENS]
            lo, hi = np.quantile(diff[m], [0.025, 0.975]) if m in diff else (np.nan, np.nan)
            summ.append(dict(bien=c, phuong_phap=m, tau_moc=tau.get(m), MAE_tong_hop=score(e),
                             chenh_so_voi_tot_nhat=score(e) - score(E[best]), tin_cay_duoi=lo, tin_cay_tren=hi))
    pd.DataFrame(rows).round(5).to_csv(os.path.join(OUT, "causal_imputation_benchmark.csv"), index=False)
    s = pd.DataFrame(summ).round(5)
    s.to_csv(os.path.join(OUT, "causal_imputation_summary.csv"), index=False)
    ff = [dict(bien=c, so_moc=L, gio=L / 4, MAE=np.mean([np.abs(X[a - 1, j] - X[a:a + L, j]).mean() for a in st]))
          for L in FFILL_LENS for st in [pick(X, 0, len(X), L, rng, 300)] for j, c in enumerate(F)]
    pd.DataFrame(ff).round(5).to_csv(os.path.join(OUT, "ffill_by_length.csv"), index=False)
    print("tỉ lệ ô trống thật theo độ dài:", {L: round(x, 4) for L, x in w.items()})
    pd.set_option("display.width", 250)
    print(s.sort_values(["bien", "MAE_tong_hop"]).to_string(index=False))


if __name__ == "__main__":
    main()
