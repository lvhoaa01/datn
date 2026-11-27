"""Quá trình huấn luyện của từng thí nghiệm: sai số theo epoch trên train và val, tổng và theo từng bước dự báo
(docs/04_MODEL_TRAINING_EVALUATION.md mục 6.3).

Với mỗi bộ đặc trưng và seed 0, 1, 2: huấn luyện y hệt scripts/train.py (cùng seed, cùng thứ tự mẫu, cùng siêu tham
số) nhưng không dừng sớm, đủ N_EPOCHS epoch, để thấy cả phần sau điểm dừng. Sau mỗi epoch ghi MSE huấn luyện (thang
chuẩn hóa) và MAE, RMSE, R2, MAE từng bước trên val. Các epoch trùng với lần chạy có dừng sớm (artifacts/runs/) được
so với history.csv của lần đó và phải khớp tuyệt đối. Chỉ dùng train và val, không đụng test.
Ghi vào artifacts/analysis/:
  training_curves.csv: từng bộ, seed, epoch.
  fig_huan_luyen.png: MSE huấn luyện, MAE val, MAE val ở bước 15 phút và 6 giờ theo epoch.
Chạy từ gốc repo (sau train.py hoặc tune.py): uv run python scripts/training_curves.py
Chỉ vẽ lại hình từ bảng đã ghi:               uv run python scripts/training_curves.py --plot
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from common import CONFIGS, REPO
from plot_style import EXP_COLOR, INK, panel, save
from train import FEATURE_SETS, RNNForecaster, build_parser, evaluate, iter_runs, make_data, masked_mse, run_root, seed_all

OUT = os.path.join(REPO, "artifacts", "analysis")
N_EPOCHS = 15
SEEDS = (0, 1, 2)
SETS = {"base": ("E1", EXP_COLOR["base"]), "domain": ("E2", EXP_COLOR["domain"])}


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
    """Ba khung theo epoch, trung bình 3 seed: (a) MSE trên train; (b) MAE trên val, 24 bước; (c) MAE trên val ở
    bước 15 phút (nét đứt) và 6 giờ (nét liền)."""
    mean = tab.groupby(["bo", "epoch"]).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
    for fs, (tag, c) in SETS.items():
        g = mean[mean.bo == fs]
        axes[0].plot(g.epoch, g.train_mse, color=c)
        axes[1].plot(g.epoch, g.val_MAE, color=c)
        axes[2].plot(g.epoch, g.MAE_b1, color=c, ls="--")
        axes[2].plot(g.epoch, g.MAE_b24, color=c)
    axes[0].set_ylabel("MSE trên train (thang chuẩn hóa)")
    axes[1].set_ylabel("MAE trên val (mg/L)")
    axes[2].set_ylabel("MAE trên val (mg/L)")
    handles = [plt.Line2D([], [], color=c) for _, c in SETS.values()]
    handles += [plt.Line2D([], [], color=INK, ls="--"), plt.Line2D([], [], color=INK)]
    fig.legend(handles, [t for t, _ in SETS.values()] + ["(c) bước 15 phút", "(c) bước 6 giờ"], loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, 1.0))
    for ax, k in zip(axes, "abc"):
        ax.set_xlabel("Epoch")
        ax.set_xticks([1, 5, 10, 15])
        panel(ax, k)
    fig.tight_layout()
    save(fig, os.path.join(OUT, "fig_huan_luyen.png"))
    print(mean[["bo", "epoch", "train_mse", "val_MAE", "MAE_b1", "MAE_b4", "MAE_b24"]].round(4).to_string())


def main():
    """Ghi bảng và hình quá trình huấn luyện cho các bộ đặc trưng có trong SETS; --plot chỉ vẽ lại từ bảng đã ghi."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "training_curves.csv")
    p = argparse.ArgumentParser()
    p.add_argument("--plot", action="store_true", help="chỉ vẽ lại hình từ training_curves.csv, không huấn luyện")
    if p.parse_args().plot:
        figures(pd.read_csv(path))
        return
    args = build_parser().parse_args([])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for fs in SETS:
        if fs in FEATURE_SETS:
            for s in SEEDS:
                rows += curve(args, fs, s, device)
    tab = pd.DataFrame(rows)
    tab.round(6).to_csv(path, index=False)
    check_against_runs(tab, args)
    figures(pd.read_csv(path))


if __name__ == "__main__":
    main()
