"""Phân tích ảnh hưởng của đặc trưng lên dự báo bằng GradientSHAP (expected gradients) trên test,
cho các mô hình đã huấn luyện lại (scripts/final_train.py); dùng cho docs/04_MODEL_TRAINING_EVALUATION.md mục 8.

Phần góp của ô i (một cột tại một mốc của đầu vào) vào dự báo f ở một bước:
  phi_i = TB trên các cặp (x', a) của (x_i - x'_i) x df/dx_i tại x' + a (x - x'),
với x' là cửa sổ train lấy ngẫu nhiên (tham chiếu), a lấy đều trên (0, 1). Tổng phi trên mọi ô xấp xỉ
f(x) - TB f(x') (tính đầy đủ), được kiểm tra và ghi lại. Phần góp đổi sang mg/L.
Cửa sổ được giải thích: cửa sổ test có thời điểm dự báo tròn giờ (mỗi giờ một cửa sổ), 32 tham chiếu mỗi cửa sổ,
cùng tham chiếu cho mọi bộ đặc trưng; ba seed của mỗi bộ, số liệu là trung bình trên ba seed.
Ghi vào artifacts/analysis/:
  attr_completeness.csv: tương quan giữa tổng phần góp và f(x) - TB f(x'), sai lệch tuyệt đối trung bình.
  attr_by_column.csv, attr_by_group.csv: mức ảnh hưởng = TB trên cửa sổ của |phần góp ròng| (cộng trên 96 mốc,
    cộng trên các cột của nhóm), kèm độ lệch chuẩn giữa seed và tỉ trọng (%), cho bước 15 phút, 3 giờ, 6 giờ.
  attr_by_lag.csv: tỉ trọng ảnh hưởng của từng giờ trong 24 giờ trước lúc dự báo, theo nhóm.
  attr_by_condition.csv: tỉ trọng nhóm theo trạng thái lúc dự báo: nhiệt độ nước, bão hòa từ 130% (tảo nở hoa),
    ban đêm, ban ngày, và các cửa sổ quanh đợt DO dưới 6 mg/L.
  attr_pairs.csv: hai nhóm mang thông tin trùng nhau (DO và Bão hòa, Nhiệt độ và Bão hòa): tương quan giữa phần góp
    ròng của hai nhóm qua các cửa sổ, mức ảnh hưởng của từng nhóm và của tổng hai nhóm.
  attr_dependence.csv: phần góp ròng TB theo giá trị hiện tại của một số đặc trưng (chia 10 nhóm phân vị).
  attr_example.csv: phần góp có dấu theo nhóm và mốc của một cửa sổ trước đợt DO thấp sâu nhất.
  fig_anh_huong_nhom.png, fig_anh_huong_theo_tre.png, fig_anh_huong_theo_dieu_kien.png, fig_phu_thuoc.png,
  fig_vi_du_anh_huong.png.
Chạy từ gốc repo (sau final_train.py): uv run python scripts/attribution.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import CONFIGS, FEATURES, FILLED, FLAGS, load_base_all, solar_elevation_sin, window_starts
from describe_dataset import episode_spans
from error_analysis import LEAD, LOW, OUT, SETS, window_conditions
from figures import AQUA, BLUE, INK2, ORANGE
from train import FEATURE_SETS, RNNForecaster, final_dir, iter_runs

N_REF = 32                                        # số tham chiếu mỗi cửa sổ
CHUNK = 64                                        # số cửa sổ mỗi lô
TARGETS = {"15 phút": 0, "3 giờ": 11, "6 giờ": 23}
LAG_BINS = [(4 * h, 4 * h + 4, h + 1) for h in range(24)]   # (mốc đầu, mốc cuối tính từ lúc dự báo, giờ thứ mấy trước)
VAR_NAME = {"do_mg_l": "DO", "water_temp_c": "Nhiệt độ", "ph": "pH", "spc_us_cm": "Độ dẫn",
            "turbidity_fnu": "Độ đục", "chl_fluor_ug_l": "Chlorophyll"}
DOMAIN_GROUPS = {                                 # nhóm cơ chế của 24 đặc trưng miền (03_DOMAIN_FEATURES_ANALYSIS mục 4)
    "Bão hòa": ["do_sat_mgl", "do_sat_pct", "do_deficit"],
    "Ánh sáng": ["light"],
    "Quang hợp, hô hấp": ["gpp_now", "er_now"],
    "Chu kỳ ngày của DO": ["do_change_24h", "do_std_24h", "do_anom_24h"],
    "Thay đổi ngắn hạn": ["do_change_1h", "do_change_3h", "temp_change_3h"],
    "Cacbonat": ["ph_change_3h", "ph_change_24h"],
    "Tảo": ["log_chl"],
    "Vận chuyển": ["spc_change_24h"],
    "Triều": ["m2_sin", "m2_cos"],
    "Lịch": ["hour_sin", "hour_cos", "hour2_sin", "hour2_cos", "doy_sin", "doy_cos"],
}
DEPENDENCE = ["do_anom_24h", "gpp_now", "do_sat_pct", "light"]   # đặc trưng vẽ quan hệ giá trị và phần góp
PAIRS = [("DO", "Bão hòa"), ("Nhiệt độ", "Bão hòa")]
TEMP_BINS = ([-np.inf, 18, 24, np.inf], ["dưới 18 độ C", "18 đến 24 độ C", "từ 24 độ C"])


def columns(fs):
    """Tên các cột đầu vào của bộ fs theo đúng thứ tự trong mô hình (cột số rồi cờ)."""
    return list(FEATURE_SETS[fs][1]) + FLAGS


def groups(fs):
    """{tên nhóm: chỉ số cột}: mỗi biến đo một nhóm, đặc trưng miền theo DOMAIN_GROUPS, sáu cờ một nhóm."""
    cols = columns(fs)
    g = {VAR_NAME[c]: [cols.index(c)] for c in FEATURES}
    if fs != "base":
        g.update({k: [cols.index(c) for c in v] for k, v in DOMAIN_GROUPS.items()})
    g["Cờ điền"] = [cols.index(c) for c in FLAGS]
    return g


def final_models(fs, model="lstm", cfg="24h_6h"):
    """Đường dẫn model.pt của ba seed đã chấm test của bộ fs, theo thứ tự seed."""
    found = {c["seed"]: os.path.join(d, "model.pt") for d, c in iter_runs(final_dir(model, cfg)) if c["features"] == fs}
    return [found[s] for s in sorted(found)]


def load_model(path, device):
    """Mô hình và tensor đầu vào chung X (chuẩn hóa bằng scaler lưu cùng mô hình)."""
    ck = torch.load(path, map_location=device)
    c = ck["config"]
    model = RNNForecaster(c["model"], c["n_feat"], c["n_out"], c["hidden"], c["layers"], c["dropout"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return model, np.array(ck["scaler_mean"]), np.array(ck["scaler_scale"])


def explain(model, X, t_explain, refs, alphas, n_in, sd, device, gidx):
    """Phần góp ròng theo cột (W, C, mục tiêu) và theo nhóm, mốc (W, G, n_in, mục tiêu), kèm kiểm tra tính đầy đủ."""
    # cuDNN chỉ lan truyền ngược RNN ở chế độ train; tắt dropout để chế độ train cho đúng kết quả như chế độ eval
    model.train()
    model.rnn.dropout = 0.0
    ai = torch.arange(-n_in, 0, device=device)
    win = lambda t: X[torch.as_tensor(t, device=device)[:, None] + ai]   # noqa: E731
    W, C, K, G = len(t_explain), X.shape[1], len(TARGETS), len(gidx)
    col = np.zeros((W, C, K), np.float32)
    lag = np.zeros((W, G, n_in, K), np.float32)
    gap = np.zeros((W, K, 2), np.float32)
    M = torch.zeros(C, G, device=device)
    for j, idx in enumerate(gidx):
        M[idx, j] = 1
    for i in range(0, W, CHUNK):
        x = win(t_explain[i:i + CHUNK])
        B = len(x)
        xr = x.repeat_interleave(N_REF, 0)
        xb = win(refs[i:i + B].ravel())
        a = torch.as_tensor(alphas[i:i + B].ravel(), device=device)[:, None, None]
        xi = (xb + a * (xr - xb)).requires_grad_(True)
        out = model(xi)
        with torch.no_grad():
            fx, fb = model(x), model(xb).view(B, N_REF, -1).mean(1)
        for k, j in enumerate(TARGETS.values()):
            gr, = torch.autograd.grad(out[:, j].sum(), xi, retain_graph=k < K - 1)
            p = ((xr - xb) * gr).view(B, N_REF, n_in, C).mean(1) * sd                 # (B, n_in, C), mg/L
            col[i:i + B, :, k] = p.sum(1).detach().cpu().numpy()
            lag[i:i + B, :, :, k] = (p @ M).permute(0, 2, 1).detach().cpu().numpy()
            gap[i:i + B, k, 0] = p.sum((1, 2)).detach().cpu().numpy()
            gap[i:i + B, k, 1] = ((fx[:, j] - fb[:, j]) * sd).cpu().numpy()
    model.eval()
    return col, lag, gap


def main():
    """Tính phần góp cho mọi bộ đặc trưng đã chấm test, ghi bảng và hình."""
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    n_in, n_out = CONFIGS["24h_6h"]
    sets = [fs for fs in SETS if os.path.isdir(FEATURE_SETS.get(fs, ("",))[0]) and final_models(fs)]
    d_all = {fs: load_base_all(FEATURE_SETS[fs][0]) for fs in sets}
    d0 = d_all[sets[0]]
    st = window_starts(d0, n_in, n_out)
    t_explain = st["test"][d0.datetime_pst[st["test"]].dt.minute.to_numpy() == 0]
    cond = window_conditions(d0, t_explain)
    cond["night"] = np.asarray(solar_elevation_sin(d0.datetime_pst.iloc[t_explain - 1])) <= 0   # mặt trời dưới chân trời
    test = d0[d0["split"] == "test"]
    events = [(test.index[0] + a, test.index[0] + a + ln - 1) for a, ln in episode_spans(test.do_mg_l.to_numpy() < LOW)]
    near_low = np.zeros(len(t_explain), bool)
    for s_, e_ in events:
        near_low |= (t_explain >= s_ - LEAD) & (t_explain <= e_)
    deepest = min(events, key=lambda se: d0.do_mg_l[se[0]:se[1] + 1].min())
    ex_i = int(np.argmin(np.abs(t_explain - (deepest[0] - LEAD))))
    res = {}
    for fs in sets:
        d = d_all[fs]
        gi = groups(fs)
        cols, gnames = columns(fs), list(gi)
        out = dict(col=[], lag=[], gap=[])
        for s, path in enumerate(final_models(fs)):
            model, mu, scale = load_model(path, device)
            num = FEATURE_SETS[fs][1]
            Z = (d[num].to_numpy() - mu) / scale
            X = torch.as_tensor(np.hstack([Z, d[FLAGS].to_numpy() == FILLED]), dtype=torch.float32, device=device)
            rng = np.random.default_rng(s)                                       # cùng tham chiếu cho mọi bộ
            refs = rng.choice(st["train"], (len(t_explain), N_REF))
            alphas = rng.random((len(t_explain), N_REF)).astype(np.float32)
            with torch.no_grad():                                                # chế độ train không dropout = eval
                model.eval()
                xe = X[torch.as_tensor(t_explain[:256], device=device)[:, None] + torch.arange(-n_in, 0, device=device)]
                ref_out = model(xe)
                model.train()
                model.rnn.dropout = 0.0
                assert torch.allclose(ref_out, model(xe), atol=1e-5)
            col, lag, gap = explain(model, X, t_explain, refs, alphas, n_in, scale[0], device, list(gi.values()))
            out["col"].append(col)
            out["lag"].append(lag)
            out["gap"].append(gap)
            print(f"{fs} seed {s}: xong", flush=True)
        res[fs] = dict(col=np.stack(out["col"]), lag=np.stack(out["lag"]), gap=np.stack(out["gap"]), cols=cols,
                       gnames=gnames, gidx=gi, d=d)
    write_tables(res, cond, near_low, ex_i, t_explain, n_in)


def share_rows(imp, names, fs, extra):
    """Dòng bảng: mức ảnh hưởng TB, độ lệch chuẩn giữa seed, tỉ trọng (%) cho từng tên và mục tiêu.
    imp: (seed, tên, mục tiêu)."""
    rows = []
    for k, tgt in enumerate(TARGETS):
        m, sd = imp[:, :, k].mean(0), imp[:, :, k].std(0, ddof=1)
        share = 100 * imp[:, :, k] / imp[:, :, k].sum(1, keepdims=True)
        for j, n in enumerate(names):
            rows.append(dict(bo=SETS[fs][0], muc_tieu=tgt, **extra(j), anh_huong_mgl=m[j], sd_giua_seed=sd[j],
                             ti_trong_pct=share[:, j].mean()))
    return rows


def write_tables(res, cond, near_low, ex_i, t_explain, n_in):
    """Gộp phần góp thành các bảng, hình của docstring."""
    comp, bycol, bygrp, bylag, bycond, dep, pairs = [], [], [], [], [], [], []
    for fs, r in res.items():
        col, lag, gap = r["col"], r["lag"], r["gap"]                 # (S, W, C, K), (S, W, G, L, K), (S, W, K, 2)
        for k, tgt in enumerate(TARGETS):
            for s in range(len(gap)):
                g = gap[s, :, k]
                comp.append(dict(bo=SETS[fs][0], muc_tieu=tgt, seed=s, tuong_quan=np.corrcoef(g[:, 0], g[:, 1])[0, 1],
                                 sai_lech_tb_mgl=np.abs(g[:, 0] - g[:, 1]).mean(), do_lon_tb_mgl=np.abs(g[:, 1]).mean()))
        group_of = {c: n for n, idx in r["gidx"].items() for c in [r["cols"][i] for i in idx]}
        imp_col = np.abs(col).mean(1)                                # (S, C, K)
        bycol += share_rows(imp_col, r["cols"], fs, lambda j: dict(cot=r["cols"][j], nhom=group_of[r["cols"][j]]))
        gnet = lag.sum(3)                                            # phần góp ròng theo nhóm (S, W, G, K)
        imp_g = np.abs(gnet).mean(1)
        bygrp += share_rows(imp_g, r["gnames"], fs, lambda j: dict(nhom=r["gnames"][j]))
        # theo khoảng trễ: |phần góp ròng của nhóm trong khoảng| rồi lấy tỉ trọng trên tổng các khoảng và nhóm
        for k, tgt in enumerate(TARGETS):
            tot = []
            for a, b, lab in LAG_BINS:
                v = np.abs(lag[:, :, :, n_in - b:n_in - a, k].sum(3)).mean(1)       # (S, G); mốc cuối là chỉ số n_in - 1
                tot.append(v)
            tot = np.stack(tot, 1)                                                  # (S, khoảng, G)
            share = 100 * tot / tot.sum((1, 2), keepdims=True)
            for bi, (_, _, lab) in enumerate(LAG_BINS):
                bylag.append(dict(bo=SETS[fs][0], muc_tieu=tgt, gio_truoc=lab, nhom="tất cả",
                                  ti_trong_pct=share[:, bi].sum(1).mean()))
                for j, n in enumerate(r["gnames"]):
                    bylag.append(dict(bo=SETS[fs][0], muc_tieu=tgt, gio_truoc=lab, nhom=n,
                                      ti_trong_pct=share[:, bi, j].mean()))
        # theo điều kiện
        tb = pd.cut(cond.water_temp_c, TEMP_BINS[0], right=False, labels=TEMP_BINS[1]).astype(str).to_numpy()
        sel = ([("toàn bộ test", np.ones(len(tb), bool))] + [(f"nước {b}", tb == b) for b in TEMP_BINS[1]]
               + [("bão hòa từ 130% (tảo nở hoa)", cond.do_sat_pct.to_numpy() >= 130),
                  ("ban đêm", cond.night.to_numpy()), ("ban ngày", ~cond.night.to_numpy()),
                  ("quanh đợt DO dưới 6 mg/L", near_low)])
        for name, m in sel:
            if not m.any():
                continue
            v = np.abs(gnet[:, m]).mean(1)                                           # (S, G, K)
            share = 100 * v / v.sum(1, keepdims=True)
            for k, tgt in enumerate(TARGETS):
                for j, n in enumerate(r["gnames"]):
                    bycond.append(dict(bo=SETS[fs][0], muc_tieu=tgt, dieu_kien=name, so_cua_so=int(m.sum()), nhom=n,
                                       ti_trong_pct=share[:, j, k].mean()))
        if fs != "base":
            gn = r["gnames"]
            for a, b in PAIRS:
                ia, ib = gn.index(a), gn.index(b)
                for k, tgt in enumerate(TARGETS):
                    va, vb = gnet[:, :, ia, k], gnet[:, :, ib, k]                   # (S, W)
                    pairs.append(dict(bo=SETS[fs][0], muc_tieu=tgt, nhom_a=a, nhom_b=b,
                                      tuong_quan=np.mean([np.corrcoef(va[s], vb[s])[0, 1] for s in range(len(va))]),
                                      anh_huong_a=np.abs(va).mean(), anh_huong_b=np.abs(vb).mean(),
                                      anh_huong_tong=np.abs(va + vb).mean()))
            d = r["d"]
            for c in DEPENDENCE:
                ci = r["cols"].index(c)
                val = d[c].to_numpy()[t_explain - 1]
                q = pd.qcut(val, 10, labels=False, duplicates="drop")
                for k, tgt in enumerate(TARGETS):
                    contrib = col[:, :, ci, k].mean(0)
                    for b in np.unique(q):
                        dep.append(dict(cot=c, muc_tieu=tgt, nhom_phan_vi=int(b), gia_tri_tb=val[q == b].mean(),
                                        phan_gop_tb_mgl=contrib[q == b].mean()))
    pd.DataFrame(comp).round(6).to_csv(os.path.join(OUT, "attr_completeness.csv"), index=False)
    bycol, bygrp = pd.DataFrame(bycol), pd.DataFrame(bygrp)
    bycol.round(6).to_csv(os.path.join(OUT, "attr_by_column.csv"), index=False)
    bygrp.round(6).to_csv(os.path.join(OUT, "attr_by_group.csv"), index=False)
    bylag = pd.DataFrame(bylag)
    bylag.round(6).to_csv(os.path.join(OUT, "attr_by_lag.csv"), index=False)
    bycond = pd.DataFrame(bycond)
    bycond.round(6).to_csv(os.path.join(OUT, "attr_by_condition.csv"), index=False)
    pd.DataFrame(pairs).round(6).to_csv(os.path.join(OUT, "attr_pairs.csv"), index=False)
    dep = pd.DataFrame(dep)
    dep.round(6).to_csv(os.path.join(OUT, "attr_dependence.csv"), index=False)
    # ví dụ một cửa sổ (bộ cuối cùng, thường là bộ nhiều đặc trưng nhất), bước 6 giờ
    fs = list(res)[-1]
    r = res[fs]
    ex = r["lag"][:, ex_i, :, :, 2].mean(0)                                          # (G, L)
    t0 = r["d"].datetime_pst[t_explain[ex_i]]
    exdf = pd.DataFrame(ex.T, columns=r["gnames"])
    exdf.insert(0, "gio_truoc", (np.arange(n_in)[::-1] + 1) * 0.25)
    exdf.round(5).to_csv(os.path.join(OUT, "attr_example.csv"), index=False)
    figures(bygrp, bylag, bycond, dep, ex, r["gnames"], t0, fs, n_in)
    print(pd.DataFrame(comp).groupby(["bo", "muc_tieu"])[["tuong_quan", "sai_lech_tb_mgl", "do_lon_tb_mgl"]].mean().round(3))
    print(bygrp[bygrp.muc_tieu == "6 giờ"].sort_values(["bo", "ti_trong_pct"], ascending=False).round(2).to_string())


def figures(bygrp, bylag, bycond, dep, ex, gnames, t0, fs, n_in):
    """Năm hình của phân tích ảnh hưởng."""
    tgt_color = {"15 phút": BLUE, "3 giờ": AQUA, "6 giờ": ORANGE}
    # 1. tỉ trọng nhóm, từng bộ, ba bước dự báo
    sets = list(bygrp.bo.unique())
    fig, axes = plt.subplots(1, len(sets), figsize=(6 * len(sets), 4.6), gridspec_kw={"width_ratios": [
        bygrp[bygrp.bo == b].nhom.nunique() for b in sets]} if len(sets) > 1 else None)
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, sets):
        g = bygrp[bygrp.bo == b]
        order = g[g.muc_tieu == "6 giờ"].sort_values("ti_trong_pct").nhom.tolist()
        y = np.arange(len(order))
        h = 0.8 / len(TARGETS)
        for k, tgt in enumerate(TARGETS):
            v = g[g.muc_tieu == tgt].set_index("nhom").loc[order, "ti_trong_pct"]
            ax.barh(y + (k - 1) * h, v, h * 0.9, color=tgt_color[tgt], label=f"dự báo {tgt}")
        ax.set_yticks(y, order)
        ax.set_xlabel("tỉ trọng ảnh hưởng (%)")
        ax.set_title(f"{b}: nhóm đầu vào theo mức ảnh hưởng")
        ax.grid(axis="x", color="#e6e5e0")
        ax.grid(axis="y", visible=False)
    axes[-1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_anh_huong_nhom.png"), dpi=150)
    plt.close(fig)
    # 2. theo giờ trước lúc dự báo, tổng mọi nhóm, từng bộ, ba bước
    fig, axes = plt.subplots(1, len(sets), figsize=(6 * len(sets), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, sets):
        g = bylag[(bylag.bo == b) & (bylag.nhom == "tất cả")]
        for tgt in TARGETS:
            v = g[g.muc_tieu == tgt].sort_values("gio_truoc")
            ax.plot(v.gio_truoc, v.ti_trong_pct, "o-", ms=3.5, lw=1.8, color=tgt_color[tgt], label=f"dự báo {tgt}")
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.set_xlabel("giờ thứ mấy trước lúc dự báo")
        ax.set_title(f"{b}: ảnh hưởng theo thời gian trong đầu vào")
    axes[0].set_ylabel("tỉ trọng ảnh hưởng (%)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_anh_huong_theo_tre.png"), dpi=150)
    plt.close(fig)
    # 3. theo trạng thái: bản đồ nhiệt tỉ trọng nhóm (hàng) theo trạng thái (cột), từng bộ, bước 6 giờ
    conds = list(dict.fromkeys(bycond.dieu_kien))
    fig, axes = plt.subplots(1, len(sets), figsize=(6.2 * len(sets), 6.4), gridspec_kw={"width_ratios": [1] * len(sets)})
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, sets):
        g = bycond[(bycond.bo == b) & (bycond.muc_tieu == "6 giờ")]
        order = g[g.dieu_kien == "toàn bộ test"].sort_values("ti_trong_pct", ascending=False).nhom.tolist()
        M = g.pivot(index="nhom", columns="dieu_kien", values="ti_trong_pct").loc[order, conds].to_numpy()
        ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=max(30, np.nanmax(M)))
        for (r_, c_), v in np.ndenumerate(M):
            ax.text(c_, r_, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color="white" if v > 20 else INK2)
        n = g.drop_duplicates("dieu_kien").set_index("dieu_kien").loc[conds, "so_cua_so"]
        ax.set_xticks(range(len(conds)), [f"{c}\n({k} cửa sổ)" for c, k in zip(conds, n)], rotation=35, ha="right",
                      fontsize=7.5)
        ax.set_yticks(range(len(order)), order)
        ax.set_title(f"{b}, dự báo 6 giờ: tỉ trọng ảnh hưởng (%) theo trạng thái")
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_anh_huong_theo_dieu_kien.png"), dpi=150)
    plt.close(fig)
    # 4. quan hệ giá trị đặc trưng và phần góp
    if len(dep):
        fig, axes = plt.subplots(1, len(DEPENDENCE), figsize=(12, 3.2))
        for ax, c in zip(axes, DEPENDENCE):
            for tgt in TARGETS:
                g = dep[(dep.cot == c) & (dep.muc_tieu == tgt)]
                ax.plot(g.gia_tri_tb, g.phan_gop_tb_mgl, "o-", ms=4, lw=1.5, color=tgt_color[tgt], label=f"dự báo {tgt}")
            ax.axhline(0, color=INK2, lw=0.7)
            ax.set_title(c)
            ax.set_xlabel("giá trị lúc dự báo")
        axes[0].set_ylabel("phần góp ròng TB (mg/L)")
        axes[0].legend(fontsize=7.5)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_phu_thuoc.png"), dpi=150)
        plt.close(fig)
    # 5. ví dụ một cửa sổ: phần góp có dấu theo nhóm và mốc
    order = np.argsort(-np.abs(ex).sum(1))[:8]
    lim = np.abs(ex[order]).max()
    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(ex[order], aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   extent=[-n_in * 0.25, 0, len(order) - 0.5, -0.5])
    ax.set_yticks(range(len(order)), [gnames[i] for i in order])
    ax.set_xlabel("giờ trước lúc dự báo")
    ax.set_title(f"{sets[-1]}, dự báo 6 giờ lúc {t0:%Y-%m-%d %H:%M}: phần góp theo nhóm và mốc "
                 f"(đỏ đẩy dự báo lên, xanh kéo xuống)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="mg/L")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_vi_du_anh_huong.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
