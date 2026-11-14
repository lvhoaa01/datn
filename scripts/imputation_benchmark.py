"""So sánh các phương pháp điền dữ liệu khuyết trên tập train RRI (dataset/LAM_SACH.md mục 5.3, phương pháp A đến H).

Chạy từ gốc repo: uv run python scripts/imputation_benchmark.py [--no-saits]
SAITS cần nhóm gói benchmark (uv sync --group benchmark); thiếu PyPOTS thì tự bỏ qua SAITS.

Cách làm (xóa nhân tạo rồi điền lại):
1. Đoạn chấm: 15% cuối của train (2023-08-09 đến 2024-05-31). 85% đầu chỉ dùng để huấn luyện SAITS.
2. Với mỗi độ dài khoảng trống L, chọn ngẫu nhiên các vị trí a sao cho từ a - 200 đến a + L + 150 mốc
   đều có đủ 6 biến (để biết giá trị thật và đủ ngữ cảnh). Các khoảng trống cùng độ dài không chồng nhau.
3. Xóa [a, a + L) theo hai kiểu: A là cả 6 biến cùng trống (giống thực tế khi DO trống);
   B là chỉ một biến trống, 5 biến kia còn (đánh giá riêng từng biến).
4. Điền lại bằng từng phương pháp, so với giá trị thật đã xóa.
   MAE: trung bình độ lệch tuyệt đối; RMSE: căn bậc hai trung bình bình phương độ lệch;
   pct_sd: MAE chia độ lệch chuẩn của biến trên train (để so sánh giữa các biến);
   gay_pct: phần trăm số khoảng trống có bước nhảy ở một trong hai đầu (ô điền đầu tiên so với số đo trước,
   số đo sau so với ô điền cuối) lớn hơn phân vị 99 của bước thay đổi 15 phút tự nhiên của biến đó.
Sáu phương pháp một biến cho kết quả như nhau ở kiểu A và B vì chỉ dùng chính biến đó.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from common import STATS as OUT, load_raw_split

F = ["do_mg_l", "water_temp_c", "ph", "spc_us_cm", "turbidity_fnu", "chl_fluor_ug_l"]
LENS = [1, 2, 4, 8, 12, 24, 48, 85, 98]          # số mốc 15 phút; 85 và 98 là độ dài hai lần bảo trì thật
# số khoảng trống thử cho mỗi độ dài
N_BY_LEN = {1: 300, 2: 300, 4: 300, 8: 300, 12: 300, 24: 200, 48: 200, 85: 100, 98: 100}
# tỉ lệ đoạn chấm; ngữ cảnh trái, phải (mốc); độ dài cửa sổ SAITS (3 ngày); số mốc một ngày
EVAL_FRAC, CTX_L, CTX_R, T_SAITS, DAY = 0.15, 200, 150, 288, 96
SEED = 20261113
# thứ tự khớp nhãn phương pháp A đến H trong LAM_SACH.md mục 5.3 (khác với kiểu xóa A, B ở trên)
METHODS = ["giu_so_truoc", "cung_gio_hom_qua", "hom_qua_offset_1_dau", "noi_thang", "pchip",
           "hom_qua_offset_2_dau", "missforest", "saits"]


def ref_index(i, a, L):
    """Chỉ số ô cùng giờ hôm qua của ô i; nếu ô đó nằm trong khoảng trống [a, a + L) thì lùi thêm từng ngày."""
    j = i - DAY
    while a <= j < a + L:
        j -= DAY
    return j


def fill_simple(v, a, L, m):
    """Điền khoảng trống [a, a + L) của chuỗi một biến v bằng phương pháp m (một trong sáu phương pháp đầu của METHODS).

    giu_so_truoc: lặp số đo trước. noi_thang: nội suy tuyến tính giữa hai đầu. pchip: nội suy PCHIP qua 8 số đo
    mỗi bên. cung_gio_hom_qua: chép ô cùng giờ hôm qua. hom_qua_offset_1_dau: như trên cộng độ lệch ở đầu trái.
    hom_qua_offset_2_dau: cộng độ lệch biến đổi tuyến tính từ đầu trái sang đầu phải.
    """
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
    """Chọn ngẫu nhiên vị trí khoảng trống nhân tạo trong đoạn chấm: {L: danh sách a}, đủ ngữ cảnh, không chồng nhau."""
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
    """Biến đổi log1p cho độ đục và huỳnh quang (phân bố lệch phải) trước khi đưa vào MissForest, SAITS."""
    Z = X.copy()
    Z[:, 4:] = np.log1p(np.clip(Z[:, 4:], 0, None))
    return Z


def from_model_space(Z):
    """Phép ngược của to_model_space."""
    X = Z.copy()
    X[:, 4:] = np.expm1(X[:, 4:])
    return X


def run_missforest(X, times, gaps, L, pattern, j_only=None):
    """MissForest (IterativeImputer với ExtraTrees), thêm đặc trưng giờ trong ngày và ngày trong năm dạng sin, cos.

    Xóa các khoảng trống theo kiểu pattern (A: cả 6 biến; B: chỉ biến j_only), điền lại; trả về mảng cùng cỡ X.
    """
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import ExtraTreesRegressor
    lo = int(len(X) * 0.70)                         # dữ liệu dùng để khớp: 30% cuối train
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
    """Huấn luyện SAITS trên Z[:cut] đã chuẩn hóa: 90% đầu làm cửa sổ huấn luyện, 10% sau làm kiểm định có mặt nạ."""
    import torch
    from pypots.imputation import SAITS
    torch.manual_seed(0)
    np.random.seed(0)
    tr_end = int(cut * 0.9)
    tr = np.stack([Z[s:s + T_SAITS] for s in range(0, tr_end - T_SAITS, 24)])
    va = np.stack([Z[s:s + T_SAITS] for s in range(tr_end, cut - T_SAITS, 96)])
    va_in = va.copy()
    rng = np.random.default_rng(1)
    for w in va_in:                                  # mặt nạ kiểm định: khối cả 6 biến, nhiều độ dài
        L = rng.choice([1, 4, 8, 24, 48, 96])
        a = rng.integers(48, T_SAITS - 48 - L)
        w[a:a + L] = np.nan
    model = SAITS(n_steps=T_SAITS, n_features=6, n_layers=3, d_model=256, n_heads=4, d_k=64, d_v=64,
                  d_ffn=512, dropout=0.1, batch_size=64, epochs=300, patience=20,
                  device="cuda" if torch.cuda.is_available() else "cpu", verbose=False)
    model.fit({"X": tr}, {"X": va_in, "X_ori": va})
    return model


def main():
    """Chạy cả tám phương pháp cho mọi độ dài L, ghi ba bảng kết quả."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-saits", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d = load_raw_split("train")
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
            print(f"SAITS huấn luyện xong sau {time.time() - t0:.0f} giây")
        except ImportError:
            print("Không có torch hoặc pypots: bỏ qua SAITS")
            use_saits = False

    rows = []

    def score(j, L, pattern, method, fills, starts):
        """Thêm một dòng kết quả (MAE, RMSE, pct_sd, gay_pct) cho biến j, độ dài L, kiểu pattern, phương pháp method."""
        e = np.concatenate([fills[k] - X[a:a + L, j] for k, a in enumerate(starts)])
        jumps = [max(abs(fills[k][0] - X[a - 1, j]), abs(X[a + L, j] - fills[k][-1])) for k, a in enumerate(starts)]
        rows.append(dict(bien=F[j], kieu=pattern, so_moc=L, gio=L / 4, phuong_phap=method, so_khoang=len(starts),
                         MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), pct_sd=100 * np.abs(e).mean() / sd[j],
                         gay_pct=100 * np.mean(np.array(jumps) > p99[j])))

    for L in LENS:
        st = gaps[L]
        t0 = time.time()
        for j in range(6):                             # phương pháp một biến: dùng chung cho kiểu A và B
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
        print(f"L = {L / 4:5.2f} giờ: {len(st)} khoảng, {time.time() - t0:.0f} giây")

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
        print(f"\nDO, kiểu {pattern}, MAE (mg/L)")
        print(q.pivot_table(index="gio", columns="phuong_phap", values="MAE")[[m for m in METHODS if m in set(q.phuong_phap)]].round(3))
    print("done")


if __name__ == "__main__":
    main()
