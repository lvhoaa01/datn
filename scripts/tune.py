"""Dò siêu tham số trên val: chạy lưới cấu hình, mỗi cấu hình nhiều seed, rồi tổng hợp trung bình và độ lệch chuẩn.

Mỗi lần chạy là một lần train.py bình thường (một thư mục trong artifacts/runs/). Run nào đã có thư mục với đúng
các tham số và cùng dữ liệu (mã băm thư mục dữ liệu trong config.json trùng common.data_fingerprint()) thì được dùng lại,
không chạy lại (cùng seed cho kết quả trùng bit trên cùng máy); run học trên dữ liệu có mã băm khác không được dùng lại.
Test không được chấm.
Kết quả: artifacts/runs/<mô hình>_<cấu hình>/tuning/tuning.csv (thư mục domain/ với --features domain), mỗi dòng một
cấu hình, xếp theo MAE trung bình tăng dần; cột runs là tên thư mục các lần chạy của cấu hình;
MAE_buoc_dau_tb, MAE_buoc_cuoi_tb là MAE ở bước nhãn đầu (15 phút) và bước cuối (6 giờ hoặc 24 giờ tùy cấu hình).
Chạy từ gốc repo: uv run python scripts/tune.py --model lstm --cfg 24h_6h
"""
import argparse
import itertools
import os

import pandas as pd

from common import data_fingerprint, read_json
from train import FEATURE_SETS, build_parser, iter_runs, train, val_dir

# lưới dò: tốc độ học, số nơ-ron ẩn, dropout giữa hai lớp hồi quy
GRID = {"lr": [1e-3, 3e-4], "hidden": [64, 128], "dropout": [0.1, 0.3]}
SEEDS = [0, 1, 2]


def find_run(args):
    """Thư mục run có config khớp mọi tham số của args và học trên dữ liệu hiện tại, hoặc None nếu chưa có."""
    want = dict(vars(args), du_lieu_sha256=data_fingerprint(FEATURE_SETS[args.features][0]))
    for path, cfg in iter_runs(val_dir(args.model, args.cfg, args.features)):
        cfg.setdefault("features", "base")      # run không ghi bộ đặc trưng là dữ liệu gốc
        if all(cfg.get(k) == v for k, v in want.items()) and os.path.exists(os.path.join(path, "metrics.json")):
            return path
    return None


def main():
    """Chạy các run còn thiếu của lưới, ghi bảng tổng hợp theo cấu hình."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--cfg", default="24h_6h")
    ap.add_argument("--features", default="base", choices=list(FEATURE_SETS))
    opt = ap.parse_args()
    rows = []
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    for i, combo in enumerate(combos, 1):
        for seed in SEEDS:
            args = build_parser().parse_args(["--model", opt.model, "--cfg", opt.cfg, "--features", opt.features,
                                              "--seed", str(seed)])
            del args.check
            for k, v in combo.items():
                setattr(args, k, v)
            out = find_run(args)
            if out is None:
                print(f"[{i}/{len(combos)}] {combo} seed {seed}: chạy mới")
                out = train(args)
            else:
                print(f"[{i}/{len(combos)}] {combo} seed {seed}: dùng lại {os.path.basename(out)}")
            m = read_json(os.path.join(out, "metrics.json"))
            n_params = read_json(os.path.join(out, "config.json"))["n_params"]
            rows.append(dict(combo, seed=seed, run=os.path.basename(out), n_params=n_params,
                             best_epoch=m["best_epoch"], MAE=m["val"]["MAE"], RMSE=m["val"]["RMSE"], R2=m["val"]["R2"],
                             MAE_buoc_dau=m["val"]["MAE_theo_buoc"][0], MAE_buoc_cuoi=m["val"]["MAE_theo_buoc"][-1]))
    df = pd.DataFrame(rows)
    keys = list(GRID)
    g = df.groupby(keys)
    summary = pd.concat([g["n_params"].first(),
                         g["MAE"].mean().rename("MAE_tb"), g["MAE"].std().rename("MAE_sd"),
                         g["RMSE"].mean().rename("RMSE_tb"), g["R2"].mean().rename("R2_tb"),
                         g["MAE_buoc_dau"].mean().rename("MAE_buoc_dau_tb"),
                         g["MAE_buoc_cuoi"].mean().rename("MAE_buoc_cuoi_tb"),
                         g["best_epoch"].apply(lambda s: "/".join(map(str, s))).rename("epoch_tot_nhat"),
                         g["run"].apply(";".join).rename("runs")], axis=1).reset_index()
    summary = summary.sort_values("MAE_tb").round(4)
    path = os.path.join(val_dir(opt.model, opt.cfg, opt.features), "tuning.csv")
    summary.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.drop(columns="runs").to_string(index=False))
    print(f"đã ghi {path}")


if __name__ == "__main__":
    main()
