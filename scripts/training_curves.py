"""Quá trình huấn luyện của từng thí nghiệm: sai số theo epoch trên train và val, tổng và theo từng bước dự báo
(docs/04_MODEL_TRAINING_EVALUATION.md mục 6.3).

Với mỗi bộ đặc trưng và seed 0, 1, 2: huấn luyện y hệt scripts/train.py (cùng seed, cùng thứ tự mẫu, cùng siêu tham
số) nhưng không dừng sớm, đủ N_EPOCHS epoch, để thấy cả phần sau điểm dừng. Sau mỗi epoch ghi MSE huấn luyện (thang
chuẩn hóa) và MAE, RMSE, R2, MAE từng bước trên val. Các epoch trùng với lần chạy có dừng sớm (artifacts/runs/) được
so với history.csv của lần đó và phải khớp tuyệt đối. Chỉ dùng train và val, không đụng test.
Ghi vào artifacts/analysis/:
  training_curves.csv: từng bộ, seed, epoch.
  fig_huan_luyen.png: MSE huấn luyện, MAE val, MAE val ở bước 15 phút và 6 giờ theo epoch.
  fig_huan_luyen_theo_buoc.png: MAE val của bộ thứ hai trừ bộ base, theo bước dự báo và epoch.
Chạy từ gốc repo (sau train.py hoặc tune.py): uv run python scripts/training_curves.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from common import CONFIGS, REPO
from figures import BLUE, ORANGE
from train import FEATURE_SETS, RNNForecaster, build_parser, evaluate, iter_runs, make_data, masked_mse, run_root, seed_all

OUT = os.path.join(REPO, "artifacts", "analysis")
N_EPOCHS = 15
SEEDS = (0, 1, 2)
SETS = {"base": ("E1", BLUE), "domain": ("E2", ORANGE)}


def curve(args, features, seed, device):
    """Huấn luyện một seed không dừng sớm; trả về danh sách chỉ số theo epoch."""
    seed_all(seed)
    loaders, scaler = make_data(args.cfg, args.batch, seed=seed, features=features)
    n_in, n_out = CONFIGS[args.cfg]
    model = RNNForecaster(args.model, loaders["train"].dataset.X.shape[1], n_out, args.hidden, args.layers,
                          args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rows = []
    for epoch in range(1, N_EPOCHS + 1):
        tot, n = 0.0, 0
        model.train()
        for x, y, m in loaders["train"]:
            x, y, m = (v.to(device, non_blocking=True) for v in (x, y, m))
            loss = masked_mse(model(x), y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            tot, n = tot + loss.item() * len(x), n + len(x)
        r = evaluate(model, loaders["val"], scaler, device)
        rows.append(dict(bo=features, seed=seed, epoch=epoch, train_mse=tot / n, val_MAE=r["MAE"], val_RMSE=r["RMSE"],
                         val_R2=r["R2"], **{f"MAE_b{k + 1}": v for k, v in enumerate(r["MAE_theo_buoc"])}))
        print(f"{features} seed {seed} epoch {epoch:2d}: val MAE {r['MAE']:.4f}", flush=True)
    return rows


def check_against_runs(tab, args):
    """So MSE huấn luyện và MAE val theo epoch với history.csv của các lần chạy có dừng sớm cùng siêu tham số."""
    hyper = ("model", "cfg", "hidden", "layers", "dropout", "lr", "weight_decay", "batch", "clip")
    n = 0
    for d, c in iter_runs(run_root(args.model, args.cfg)):
        f = os.path.join(d, "config.json")
        fs = c.get("features", "base")
        if c.get("final") or fs not in SETS or c["seed"] not in SEEDS or any(c[h] != getattr(args, h) for h in hyper):
            continue
        h = pd.read_csv(f.replace("config.json", "history.csv")).set_index("epoch")
        t = tab[(tab.bo == fs) & (tab.seed == c["seed"])].set_index("epoch")
        e = [x for x in h.index if x in t.index]
        assert np.array_equal(h.loc[e, "val_MAE"].to_numpy(), t.loc[e, "val_MAE"].to_numpy()), f
        assert np.array_equal(h.loc[e, "train_mse"].to_numpy(), t.loc[e, "train_mse"].to_numpy()), f
        n += 1
    print(f"khớp tuyệt đối với {n} lần chạy có dừng sớm")


def figures(tab):
    """Hai hình của docstring."""
    mean = tab.groupby(["bo", "epoch"]).mean(numeric_only=True).reset_index()
    best = tab.loc[tab.groupby(["bo", "seed"]).val_MAE.idxmin()]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [("train_mse", "MSE trên train (thang chuẩn hóa)", "Sai số trên dữ liệu học"),
              ("val_MAE", "MAE trên val (mg/L)", "Sai số trên val, trung bình 24 bước")]
    for ax, (col, ylab, title) in zip(axes, panels):
        for fs, (tag, c) in SETS.items():
            for s in SEEDS:
                g = tab[(tab.bo == fs) & (tab.seed == s)]
                ax.plot(g.epoch, g[col], color=c, lw=0.6, alpha=0.35)
            g = mean[mean.bo == fs]
            ax.plot(g.epoch, g[col], color=c, lw=2, label=f"{tag} (trung bình 3 seed)")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylab)
        ax.set_title(title)
    for fs, (tag, c) in SETS.items():
        b = best[best.bo == fs]
        axes[1].plot(b.epoch, b.val_MAE, "o", color=c, ms=6, mec="white", label=f"{tag}: epoch tốt nhất từng seed")
    axes[1].legend(fontsize=7.5)
    axes[0].legend(fontsize=7.5)
    ax = axes[2]
    for fs, (tag, c) in SETS.items():
        g = mean[mean.bo == fs]
        ax.plot(g.epoch, g.MAE_b1, color=c, lw=2, ls="--", label=f"{tag}, bước 15 phút")
        ax.plot(g.epoch, g.MAE_b24, color=c, lw=2, label=f"{tag}, bước 6 giờ")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MAE trên val (mg/L)")
    ax.set_title("Bước gần và bước xa")
    ax.legend(fontsize=7.5)
    for ax in axes:
        ax.set_xticks([1, 5, 10, 15])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_huan_luyen.png"), dpi=150)
    plt.close(fig)
    # hiệu MAE val theo bước và epoch
    steps = [f"MAE_b{k}" for k in range(1, 25)]
    a = mean[mean.bo == "domain"].set_index("epoch")[steps]
    b = mean[mean.bo == "base"].set_index("epoch")[steps]
    D = (a - b).to_numpy().T                                                     # (bước, epoch)
    lim = np.abs(D).max()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    im = ax.imshow(D, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower",
                   extent=[0.5, N_EPOCHS + 0.5, 0.125, 6.125])
    ax.set_xlabel("epoch")
    ax.set_ylabel("bước dự báo (giờ tới)")
    ax.set_title("MAE val của E2 trừ E1 (trung bình 3 seed): xanh là E2 tốt hơn, đỏ là E2 kém hơn")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="mg/L")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_huan_luyen_theo_buoc.png"), dpi=150)
    plt.close(fig)
    print(mean[["bo", "epoch", "train_mse", "val_MAE", "MAE_b1", "MAE_b4", "MAE_b24"]].round(4).to_string())


def main():
    """Ghi bảng và hình quá trình huấn luyện cho các bộ đặc trưng có trong SETS."""
    os.makedirs(OUT, exist_ok=True)
    args = build_parser().parse_args([])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for fs in SETS:
        if fs in FEATURE_SETS:
            for s in SEEDS:
                rows += curve(args, fs, s, device)
    tab = pd.DataFrame(rows)
    tab.round(6).to_csv(os.path.join(OUT, "training_curves.csv"), index=False)
    check_against_runs(tab, args)
    figures(tab)


if __name__ == "__main__":
    main()
