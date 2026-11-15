"""So sánh các phương pháp điền khoảng trống nhân quả trên train (dataset/LAM_SACH.md mục 4.2).

Chạy từ gốc repo: uv run python scripts/causal_imputation_benchmark.py   (sau build_base_splits.py)

Mỗi phương pháp chỉ nhận h = chuỗi trước khoảng trống (h[-1] là số đo cuối) và k = 1..L bước kể từ số đo cuối; không
biết độ dài L, giống khi chạy thật. ref(k) là giá trị cùng giờ của ngày cuối trước khoảng trống (lặp lại nếu khoảng trống
dài hơn một ngày), o = h[-1] trừ giá trị cùng giờ hôm trước:
  ffill:             h[-1]
  sn1:               ref(k)
  sn1_offset:        ref(k) + o
  sn1_offset_decay:  ref(k) + o exp(-k / tau)
  ffill_to_sn1:      pha trộn: w h[-1] + (1 - w)(ref(k) + o), w = exp(-k / tau)
  linear:            nội suy tuyến tính, dùng số đo sau khoảng trống; chỉ để tham khảo, không nhân quả.
Thiết kế: khoảng trống nhân tạo dài LENS mốc (từ 15 phút đến 64.25 giờ, khoảng trống thật dài nhất), N_GAPS mỗi độ dài,
đặt ngẫu nhiên nơi một ngày trước khoảng trống và cả khoảng trống đều có đủ sáu biến. tau chọn theo từng biến trên 85% đầu
train, chấm trên 15% cuối. Sai số tổng hợp: MAE trung bình theo tỉ lệ ô trống thật (mọi khoảng trống, mọi biến, cả ba
tập), mỗi khoảng trống thật gán vào độ dài LENS gần nhất phía trên.
Ghi vào artifacts/dataset/: causal_imputation_benchmark.csv (MAE theo biến, phương pháp, độ dài),
causal_imputation_summary.csv (sai số tổng hợp, tau), linear_leak.csv (tỉ lệ cửa sổ 24h_6h được giữ mà nội suy tuyến
tính sẽ cho đầu vào thấy số đo sau thời điểm dự báo: đầu vào chứa ô điền của một khoảng trống kết thúc từ thời điểm dự
báo trở đi).
"""
import os

import numpy as np
import pandas as pd

from common import (CONFIGS, DAY, FEATURES as F, FILLED, FLAGS, NAMES, STATS as OUT, load_base_all, load_raw_split,
                    runs, window_starts)

LENS = [1, 4, 8, 16, 32, 96, 192, 257]   # độ dài khoảng trống, mốc 15 phút: 15 phút, 1, 2, 4, 8, 24, 48, 64.25 giờ
N_GAPS = 200
TAUS = [1, 2, 4, 8, 16, 32, 64, 96, 192, 384, 768]   # mốc 15 phút: từ 15 phút đến 8 ngày
TUNE_FRAC = 0.85
SEED = 20261115
FIXED = ["ffill", "sn1", "sn1_offset"]
TUNED = ["sn1_offset_decay", "ffill_to_sn1"]


def fill(h, L, method, tau=None):
    """Điền L ô sau h bằng method; h có ít nhất một ngày và một mốc, không có NaN."""
    if method == "ffill":
        return np.full(L, h[-1])
    k = np.arange(1, L + 1)
    ref = h[-DAY:][(k - 1) % DAY]
    o = h[-1] - h[-1 - DAY]
    if method == "ffill_to_sn1":
        w = np.exp(-k / tau)
        return w * h[-1] + (1 - w) * (ref + o)
    if method == "sn1":
        return ref
    if method == "sn1_offset":
        return ref + o
    if method == "sn1_offset_decay":
        return ref + o * np.exp(-k / tau)
    raise ValueError(method)


def pick(X, lo, hi, L, rng):
    """N_GAPS vị trí a trong [lo, hi) sao cho [a - DAY - 1, a + L] đủ sáu biến; các khoảng trống có thể chồng nhau."""
    cs = np.concatenate([[0], np.cumsum(np.isnan(X).any(1))])
    cand = np.arange(lo + DAY + 1, hi - L - 1)
    cand = cand[cs[cand + L + 1] - cs[cand - DAY - 1] == 0]
    return rng.choice(cand, min(N_GAPS, len(cand)), replace=False)


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
    """Tỉ lệ ô trống thật theo độ dài LENS: mọi khoảng trống, gán vào độ dài LENS gần nhất phía trên."""
    ls = []
    for n in NAMES:
        Y = load_raw_split(n)[F].to_numpy(float)
        for j in range(len(F)):
            ls += list(runs(np.isnan(Y[:, j]))[1])
    ls = np.array(ls)
    b = np.searchsorted(LENS, ls)
    return {L: float(ls[b == i].sum() / ls.sum()) for i, L in enumerate(LENS)}


def linear_leak():
    """Bảng linear_leak: theo tập, số và tỉ lệ cửa sổ 24h_6h được giữ mà nội suy tuyến tính dùng số đo từ thời điểm dự báo trở đi."""
    d = load_base_all()
    n_in, n_out = CONFIGS["24h_6h"]
    starts = window_starts(d, n_in, n_out)
    fl = d[FLAGS].to_numpy() == FILLED
    nxt = np.zeros(fl.shape, np.int64)                       # chỉ số số đo thật đầu tiên sau mỗi ô điền
    for j in range(len(F)):
        for a, k in zip(*runs(fl[:, j])):
            nxt[a:a + k, j] = a + k
    last_end = nxt.max(1)                                    # với mỗi mốc: số đo thật xa nhất mà nội suy cần
    rows = []
    for n in NAMES:
        t = starts[n]
        leak = np.array([(last_end[x - n_in:x] >= x).any() for x in t])
        rows.append(dict(tap=n, so_cua_so=len(t), so_cua_so_thay_tuong_lai=int(leak.sum()),
                         pct=round(100 * leak.mean(), 2)))
    return pd.DataFrame(rows)


def main():
    """Chọn tau trên đoạn chỉnh, chấm mọi phương pháp trên đoạn chấm, ghi ba bảng."""
    os.makedirs(OUT, exist_ok=True)
    X = load_raw_split("train")[F].to_numpy(float)
    cut = int(len(X) * TUNE_FRAC)
    rng = np.random.default_rng(SEED)
    tune = {L: pick(X, 0, cut, L, rng) for L in LENS}
    test = {L: pick(X, cut, len(X), L, rng) for L in LENS}
    w = cell_weights()

    def score(e):
        """MAE tổng hợp theo trọng số w."""
        return sum(w[L] * e[L].mean() for L in LENS)

    rows, summ = [], []
    for j, c in enumerate(F):
        v = X[:, j]
        tau = {m: min(TAUS, key=lambda t: score(errors(v, tune, m, t))) for m in TUNED}
        for m in FIXED + TUNED + ["linear"]:
            e = errors(v, test, m, tau.get(m))
            rows += [dict(bien=c, phuong_phap=m, so_moc=L, gio=L / 4, so_khoang=len(test[L]), MAE=e[L].mean())
                     for L in LENS]
            summ.append(dict(bien=c, phuong_phap=m, tau_moc=tau.get(m), MAE_tong_hop=score(e)))
    pd.DataFrame(rows).round(5).to_csv(os.path.join(OUT, "causal_imputation_benchmark.csv"), index=False)
    s = pd.DataFrame(summ).round(5)
    s.to_csv(os.path.join(OUT, "causal_imputation_summary.csv"), index=False)
    lk = linear_leak()
    lk.to_csv(os.path.join(OUT, "linear_leak.csv"), index=False)
    print("tỉ lệ ô trống thật theo độ dài:", {L: round(x, 4) for L, x in w.items()})
    pd.set_option("display.width", 250)
    print(s.sort_values(["bien", "MAE_tong_hop"]).to_string(index=False))
    print(lk.to_string(index=False))


if __name__ == "__main__":
    main()
