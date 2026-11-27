"""Phân tích ảnh hưởng của đầu vào lên dự báo bằng GradientSHAP (expected gradients) trên test, cho các mô hình đã
huấn luyện lại (scripts/final_train.py); dùng cho docs/04_MODEL_TRAINING_EVALUATION.md mục 8.

Phần góp của ô i (một cột tại một mốc của đầu vào) vào dự báo f ở một bước:
  phi_i = TB trên các cặp (x', a) của (x_i - x'_i) x df/dx_i tại x' + a (x - x'),
với x' là cửa sổ train lấy ngẫu nhiên (tham chiếu), a lấy đều trên (0, 1). Tổng phi trên mọi ô xấp xỉ
f(x) - TB f(x') (tính đầy đủ), được kiểm tra và ghi lại. Phần góp đổi sang mg/L.
Cửa sổ được giải thích: cửa sổ test có thời điểm dự báo tròn giờ, 32 tham chiếu mỗi cửa sổ, cùng tham chiếu cho mọi
bộ đặc trưng; bước dự báo 1 (15 phút) và 24 (6 giờ); ba seed của mỗi bộ, số liệu là trung bình trên ba seed.
Nhóm đầu vào: mỗi biến đo một nhóm (khóa là tên cột), 24 đặc trưng miền theo common.DOMAIN_GROUPS, sáu cờ một nhóm (co_dien).
Ghi vào artifacts/analysis/:
  attr_completeness.csv: tương quan giữa tổng phần góp và f(x) - TB f(x'), sai lệch tuyệt đối trung bình.
  attr_by_column.csv, attr_by_group.csv: mức ảnh hưởng = TB trên cửa sổ của |phần góp ròng| (cộng trên 96 mốc,
    cộng trên các cột của nhóm), kèm độ lệch chuẩn giữa seed và tỉ trọng (%).
  attr_by_lag.csv: tỉ trọng ảnh hưởng của từng giờ trong 24 giờ trước thời điểm dự báo (gio_truoc = 1 là giờ gần nhất).
  attr_by_lag_block.csv: tỉ trọng của bốn khoảng 6 giờ (bằng độ dài khoảng dự báo) của đầu vào: 1 đến 6, 7 đến 12,
    13 đến 18, 19 đến 24 giờ trước thời điểm dự báo.
  attr_by_condition.csv: tỉ trọng nhóm theo giai đoạn mùa của thời điểm dự báo (docs/01 mục 3.3).
  attr_refs_check.csv: với seed 0, chênh lệch lớn nhất (điểm phần trăm) giữa tỉ trọng nhóm tính với 32 và với 64
    tham chiếu, và sai lệch tính đầy đủ ở hai số tham chiếu: kiểm tra 32 tham chiếu là đủ.
  fig_anh_huong_nhom.png, fig_anh_huong_theo_tre.png, fig_anh_huong_theo_mua.png: vẽ từ attr_by_group.csv,
    attr_by_lag.csv, attr_by_condition.csv.
Chạy từ gốc repo (sau final_train.py): uv run python scripts/attribution.py
Chỉ vẽ lại hình từ bảng đã ghi:         uv run python scripts/attribution.py --plot
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import CONFIGS, DOMAIN_GROUPS, FEATURES, FILLED, FLAGS, VAR_NAMES, load_base_all, window_starts
from error_analysis import OUT, SETS, window_conditions
from plot_style import INK2, SEASON_NAME, panel, save
from train import FEATURE_SETS, RNNForecaster, final_dir, iter_runs

N_REF = 32                                        # số tham chiếu mỗi cửa sổ
CHUNK = 64                                        # số cửa sổ mỗi lô
TARGETS = {1: 0, 24: 23}                          # bước dự báo: chỉ số đầu ra (bước 1 = 15 phút, bước 24 = 6 giờ)
STEP_NAME = {1: "15 phút", 24: "6 giờ"}
LAG_BINS = [(4 * h, 4 * h + 4, h + 1) for h in range(24)]   # (mốc đầu, mốc cuối tính từ thời điểm dự báo, giờ thứ mấy)
GROUP_NAME = {**VAR_NAMES, **{k: n for k, n, _ in DOMAIN_GROUPS}, "co_dien": "Cờ điền"}   # khóa nhóm: tên chuẩn
LAG_BLOCKS = [(1, 6), (7, 12), (13, 18), (19, 24)]   # khoảng giờ trước thời điểm dự báo, mỗi khoảng 6 giờ
N_REF_CHECK = 64                                      # số tham chiếu của lần kiểm tra độ đủ (seed 0)
STATE_NAME = {"toan_bo": "Toàn bộ test", **{k: v.replace(" (", "\n(") for k, v in SEASON_NAME.items()}}


def columns(fs):
    """Tên các cột đầu vào của bộ fs theo đúng thứ tự trong mô hình (cột số rồi cờ)."""
    return list(FEATURE_SETS[fs][1]) + FLAGS


def groups(fs):
    """{khóa nhóm: chỉ số cột}: mỗi biến đo một nhóm, đặc trưng miền theo DOMAIN_GROUPS, sáu cờ một nhóm."""
    cols = columns(fs)
    g = {c: [cols.index(c)] for c in FEATURES}
    if fs != "base":
        g.update({k: [cols.index(c) for c in v] for k, _, v in DOMAIN_GROUPS})
    g["co_dien"] = [cols.index(c) for c in FLAGS]
    return g


def final_models(fs, model="lstm", cfg="24h_6h"):
    """Đường dẫn model.pt của ba seed đã chấm test của bộ fs, theo thứ tự seed."""
    found = {c["seed"]: os.path.join(d, "model.pt") for d, c in iter_runs(final_dir(model, cfg)) if c["features"] == fs}
    return [found[s] for s in sorted(found)]


def load_model(path, device):
    """Mô hình và tham số chuẩn hóa lưu cùng mô hình (trung bình, độ lệch chuẩn của các cột số)."""
    ck = torch.load(path, map_location=device)
    c = ck["config"]
    model = RNNForecaster(c["model"], c["n_feat"], c["n_out"], c["hidden"], c["layers"], c["dropout"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return model, np.array(ck["scaler_mean"]), np.array(ck["scaler_scale"])


def explain(model, X, t_explain, refs, alphas, n_in, sd, device, gidx):
    """Phần góp ròng theo cột (W, C, mục tiêu) và theo nhóm, mốc (W, G, n_in, mục tiêu), kèm kiểm tra tính đầy đủ;
    số tham chiếu mỗi cửa sổ là số cột của refs."""
    # cuDNN chỉ lan truyền ngược RNN ở chế độ train; tắt dropout để chế độ train cho đúng kết quả như chế độ eval
    model.train()
    model.rnn.dropout = 0.0
    ai = torch.arange(-n_in, 0, device=device)
    win = lambda t: X[torch.as_tensor(t, device=device)[:, None] + ai]   # noqa: E731
    W, C, K, G = len(t_explain), X.shape[1], len(TARGETS), len(gidx)
    n_ref = refs.shape[1]
    chunk = CHUNK * N_REF // n_ref                          # giữ số chuỗi mỗi lô như khi dùng N_REF tham chiếu
    col = np.zeros((W, C, K), np.float32)
    lag = np.zeros((W, G, n_in, K), np.float32)
    gap = np.zeros((W, K, 2), np.float32)
    M = torch.zeros(C, G, device=device)
    for j, idx in enumerate(gidx):
        M[idx, j] = 1
    for i in range(0, W, chunk):
        x = win(t_explain[i:i + chunk])
        B = len(x)
        xr = x.repeat_interleave(n_ref, 0)
        xb = win(refs[i:i + B].ravel())
        a = torch.as_tensor(alphas[i:i + B].ravel(), device=device)[:, None, None]
        xi = (xb + a * (xr - xb)).requires_grad_(True)
        out = model(xi)
        with torch.no_grad():
            fx, fb = model(x), model(xb).view(B, n_ref, -1).mean(1)
        for k, j in enumerate(TARGETS.values()):
            gr, = torch.autograd.grad(out[:, j].sum(), xi, retain_graph=k < K - 1)
            p = ((xr - xb) * gr).view(B, n_ref, n_in, C).mean(1) * sd                 # (B, n_in, C), mg/L
            col[i:i + B, :, k] = p.sum(1).detach().cpu().numpy()
            lag[i:i + B, :, :, k] = (p @ M).permute(0, 2, 1).detach().cpu().numpy()
            gap[i:i + B, k, 0] = p.sum((1, 2)).detach().cpu().numpy()
            gap[i:i + B, k, 1] = ((fx[:, j] - fb[:, j]) * sd).cpu().numpy()
    model.eval()
    return col, lag, gap


def main():
    """Tính phần góp cho mọi bộ đặc trưng đã chấm test, ghi bảng và hình; --plot chỉ vẽ lại từ bảng đã ghi."""
    p = argparse.ArgumentParser()
    p.add_argument("--plot", action="store_true", help="chỉ vẽ lại hình từ attr_by_group.csv, attr_by_lag.csv")
    if p.parse_args().plot:
        figures()
        return
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
    res = {}
    for fs in sets:
        d = d_all[fs]
        gi = groups(fs)
        out = dict(col=[], lag=[], gap=[], check=None)
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
            if s == 0:                                                           # kiểm tra độ đủ của N_REF
                rc = np.random.default_rng(1000)
                refs2 = rc.choice(st["train"], (len(t_explain), N_REF_CHECK))
                alphas2 = rc.random((len(t_explain), N_REF_CHECK)).astype(np.float32)
                _, lag2, gap2 = explain(model, X, t_explain, refs2, alphas2, n_in, scale[0], device, list(gi.values()))
                out["check"] = (lag, gap, lag2, gap2)
            print(f"{fs} seed {s}: xong", flush=True)
        res[fs] = dict(col=np.stack(out["col"]), lag=np.stack(out["lag"]), gap=np.stack(out["gap"]), cols=columns(fs),
                       gnames=list(gi), gidx=gi, check=out["check"])
    write_tables(res, cond, n_in)
    figures()


def share_rows(imp, names, fs, extra):
    """Dòng bảng: mức ảnh hưởng TB, độ lệch chuẩn giữa seed, tỉ trọng (%) cho từng tên và bước. imp: (seed, tên, bước)."""
    rows = []
    for k, step in enumerate(TARGETS):
        m, sd = imp[:, :, k].mean(0), imp[:, :, k].std(0, ddof=1)
        share = 100 * imp[:, :, k] / imp[:, :, k].sum(1, keepdims=True)
        for j, n in enumerate(names):
            rows.append(dict(bo=SETS[fs][0], buoc=step, **extra(j), anh_huong_mgl=m[j], sd_giua_seed=sd[j],
                             ti_trong_pct=share[:, j].mean()))
    return rows


def write_tables(res, cond, n_in):
    """Gộp phần góp thành các bảng attr_*.csv của docstring."""
    comp, bycol, bygrp, bylag, byblock, bycond, check = [], [], [], [], [], [], []
    for fs, r in res.items():
        col, lag, gap = r["col"], r["lag"], r["gap"]                 # (S, W, C, K), (S, W, G, L, K), (S, W, K, 2)
        for k, step in enumerate(TARGETS):
            for s in range(len(gap)):
                g = gap[s, :, k]
                comp.append(dict(bo=SETS[fs][0], buoc=step, seed=s, tuong_quan=np.corrcoef(g[:, 0], g[:, 1])[0, 1],
                                 sai_lech_tb_mgl=np.abs(g[:, 0] - g[:, 1]).mean(), do_lon_tb_mgl=np.abs(g[:, 1]).mean()))
        group_of = {c: n for n, idx in r["gidx"].items() for c in [r["cols"][i] for i in idx]}
        imp_col = np.abs(col).mean(1)                                # (S, C, K)
        bycol += share_rows(imp_col, r["cols"], fs, lambda j: dict(cot=r["cols"][j], nhom=group_of[r["cols"][j]]))
        gnet = lag.sum(3)                                            # phần góp ròng theo nhóm (S, W, G, K)
        bygrp += share_rows(np.abs(gnet).mean(1), r["gnames"], fs, lambda j: dict(nhom=r["gnames"][j]))
        # theo giờ trước thời điểm dự báo: |phần góp ròng của nhóm trong giờ đó|, tỉ trọng trên tổng các giờ và nhóm
        for k, step in enumerate(TARGETS):
            tot = np.stack([np.abs(lag[:, :, :, n_in - b:n_in - a, k].sum(3)).mean(1) for a, b, _ in LAG_BINS], 1)
            share = 100 * tot / tot.sum((1, 2), keepdims=True)                       # (S, giờ, G)
            for bi, (_, _, h) in enumerate(LAG_BINS):
                bylag.append(dict(bo=SETS[fs][0], buoc=step, gio_truoc=h, nhom="tat_ca",
                                  ti_trong_pct=share[:, bi].sum(1).mean()))
                for j, n in enumerate(r["gnames"]):
                    bylag.append(dict(bo=SETS[fs][0], buoc=step, gio_truoc=h, nhom=n,
                                      ti_trong_pct=share[:, bi, j].mean()))
            for a, b in LAG_BLOCKS:                                                  # khoảng 6 giờ: cộng các giờ a..b
                byblock.append(dict(bo=SETS[fs][0], buoc=step, khoang=f"{a}_den_{b}",
                                    ti_trong_pct=share[:, a - 1:b].sum((1, 2)).mean()))
        # theo giai đoạn mùa của thời điểm dự báo
        sel = [("toan_bo", np.ones(len(cond), bool))] + [(k, (cond.season == k).to_numpy()) for k in SEASON_NAME]
        for name, m in sel:
            v = np.abs(gnet[:, m]).mean(1)                                           # (S, G, K)
            share = 100 * v / v.sum(1, keepdims=True)
            for k, step in enumerate(TARGETS):
                for j, n in enumerate(r["gnames"]):
                    bycond.append(dict(bo=SETS[fs][0], buoc=step, trang_thai=name, so_cua_so=int(m.sum()), nhom=n,
                                       ti_trong_pct=share[:, j, k].mean()))
        # độ đủ của số tham chiếu: tỉ trọng nhóm với N_REF và N_REF_CHECK tham chiếu (seed 0)
        lag1, gap1, lag2, gap2 = r["check"]
        sh = [100 * v / v.sum(0, keepdims=True) for v in (np.abs(x.sum(2)).mean(0) for x in (lag1, lag2))]  # (G, K)
        for k, step in enumerate(TARGETS):
            check.append(dict(bo=SETS[fs][0], buoc=step, so_tham_chieu=N_REF, so_tham_chieu_kiem_tra=N_REF_CHECK,
                              chenh_ti_trong_max_diem_pct=np.abs(sh[0][:, k] - sh[1][:, k]).max(),
                              sai_lech_tb_mgl=np.abs(gap1[:, k, 0] - gap1[:, k, 1]).mean(),
                              sai_lech_tb_mgl_kiem_tra=np.abs(gap2[:, k, 0] - gap2[:, k, 1]).mean()))
    for name, rows in [("completeness", comp), ("by_column", bycol), ("by_group", bygrp), ("by_lag", bylag),
                       ("by_lag_block", byblock), ("by_condition", bycond), ("refs_check", check)]:
        pd.DataFrame(rows).round(6).to_csv(os.path.join(OUT, f"attr_{name}.csv"), index=False)
    print(pd.DataFrame(comp).groupby(["bo", "buoc"])[["tuong_quan", "sai_lech_tb_mgl", "do_lon_tb_mgl"]].mean().round(3))


def figures():
    """Ba hình, vẽ từ bảng đã ghi: tỉ trọng theo nhóm (mỗi bộ một khung, hai bước), theo giờ trước thời điểm dự
    báo (mỗi bước một khung, các bộ cùng khung), theo giai đoạn mùa (mỗi bộ một hàng, mỗi giai đoạn một khung)."""
    bygrp = pd.read_csv(os.path.join(OUT, "attr_by_group.csv"))
    bylag = pd.read_csv(os.path.join(OUT, "attr_by_lag.csv"))
    color = {tag: c for tag, c in SETS.values()}
    sets = list(dict.fromkeys(bygrp.bo))
    sizes = [bygrp[bygrp.bo == b].nhom.nunique() for b in sets]
    fig, axes = plt.subplots(len(sets), 1, figsize=(5.5, 0.22 * sum(sizes) + 1.2),
                             gridspec_kw={"height_ratios": sizes})
    axes = np.atleast_1d(axes)
    h = 0.38
    for ax, b, key in zip(axes, sets, "abc"):
        g = bygrp[bygrp.bo == b]
        order = g[g.buoc == 24].sort_values("ti_trong_pct").nhom.tolist()
        y = np.arange(len(order))
        for off, step, alpha in [(h / 2, 1, 0.45), (-h / 2, 24, 1.0)]:
            v = g[g.buoc == step].set_index("nhom").loc[order, "ti_trong_pct"]
            ax.barh(y + off, v, h * 0.95, color=color[b], alpha=alpha, label=f"Bước {STEP_NAME[step]}")
        ax.set_yticks(y, [GROUP_NAME[n] for n in order])
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right")
        panel(ax, key, b)
    axes[-1].set_xlabel("Tỉ trọng ảnh hưởng (%)")
    fig.tight_layout()
    save(fig, os.path.join(OUT, "fig_anh_huong_nhom.png"))
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(7, 2.7), sharey=True)
    for ax, step, key in zip(axes, TARGETS, "ab"):
        for b in sets:
            g = bylag[(bylag.bo == b) & (bylag.buoc == step) & (bylag.nhom == "tat_ca")].sort_values("gio_truoc")
            ax.plot(g.gio_truoc, g.ti_trong_pct, "o-", ms=2.5, color=color[b], label=b)
        for x in (6.5, 12.5, 18.5):                                                # ranh giới các khoảng 6 giờ
            ax.axvline(x, color=INK2, lw=0.5, ls=":")
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.set_xlabel("Số giờ trước thời điểm dự báo")
        panel(ax, key, f"Bước dự báo {STEP_NAME[step]}")
    axes[0].set_ylabel("Tỉ trọng ảnh hưởng (%)")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    save(fig, os.path.join(OUT, "fig_anh_huong_theo_tre.png"))
    fig_states(pd.read_csv(os.path.join(OUT, "attr_by_condition.csv")), sets, sizes, color)


def fig_states(bycond, sets, sizes, color):
    """Tỉ trọng của mọi nhóm ở bước 6 giờ theo giai đoạn mùa của thời điểm dự báo: mỗi bộ một hàng khung, toàn bộ test
    và mỗi giai đoạn mùa một khung; nhóm xếp theo tỉ trọng trên toàn bộ test, cùng thứ tự trong cả hàng."""
    g24 = bycond[bycond.buoc == 24]
    states = list(STATE_NAME)
    fig = plt.figure(figsize=(7.2, 0.24 * sum(sizes) + 2.2), layout="constrained")
    subs = fig.subfigures(len(sets), 1, height_ratios=[n + 3 for n in sizes])
    for sub, b, key in zip(np.atleast_1d(subs), sets, "abc"):
        g = g24[g24.bo == b]
        order = g[g.trang_thai == "toan_bo"].sort_values("ti_trong_pct").nhom.tolist()
        top = g.ti_trong_pct.max()
        axes = sub.subplots(1, len(states), sharey=True)
        for ax, st in zip(axes, states):
            v = g[g.trang_thai == st].set_index("nhom").loc[order, "ti_trong_pct"]
            ax.barh(np.arange(len(order)), v, 0.7, color=color[b])
            ax.set_xlim(0, top * 1.08)
            ax.set_title(STATE_NAME[st], loc="center", fontsize=8)
            ax.grid(axis="x")
            ax.grid(axis="y", visible=False)
            ax.tick_params(axis="y", length=0)
        axes[0].set_yticks(np.arange(len(order)), [GROUP_NAME[n] for n in order])
        sub.supxlabel("Tỉ trọng ảnh hưởng (%)", fontsize=8)
        sub.suptitle(f"({key}) {b}", x=0.0, ha="left", fontsize=9)
    save(fig, os.path.join(OUT, "fig_anh_huong_theo_mua.png"))


if __name__ == "__main__":
    main()
