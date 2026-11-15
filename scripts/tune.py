"""Dò siêu tham số trên val: chạy lưới cấu hình, mỗi cấu hình nhiều seed, rồi tổng hợp trung bình và độ lệch chuẩn.

Mỗi lần chạy là một lần train.py bình thường (một thư mục trong artifacts/runs/). Run nào đã có thư mục với đúng
các tham số thì được dùng lại, không chạy lại (cùng seed cho kết quả trùng bit trên cùng máy). Test không được chấm.
Kết quả: artifacts/runs/tuning_<mô hình>_<cấu hình>.csv, mỗi dòng một cấu hình, xếp theo MAE trung bình tăng dần.
Chạy từ gốc repo: uv run python scripts/tune.py --model lstm --cfg 24h_6h
"""
import argparse
import glob
import itertools
import json
import os

import pandas as pd

from train import RUNS, build_parser, train

# lưới dò: tốc độ học, số nơ-ron ẩn, dropout giữa hai lớp hồi quy
GRID = {"lr": [1e-3, 3e-4], "hidden": [64, 128], "dropout": [0.1, 0.3]}
SEEDS = [0, 1, 2]


def find_run(args):
    """Thư mục run có config khớp mọi tham số của args, hoặc None nếu chưa có."""
    want = vars(args)
    for path in sorted(glob.glob(os.path.join(RUNS, "*", "config.json"))):
        cfg = json.load(open(path))
        if all(cfg.get(k) == v for k, v in want.items()) and os.path.exists(os.path.join(os.path.dirname(path),
                                                                                           "metrics.json")):
            return os.path.dirname(path)
    return None


def main():
    """Chạy các run còn thiếu của lưới, ghi bảng tổng hợp theo cấu hình."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--cfg", default="24h_6h")
    opt = ap.parse_args()
    rows = []
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    for i, combo in enumerate(combos, 1):
        for seed in SEEDS:
            args = build_parser().parse_args(["--model", opt.model, "--cfg", opt.cfg, "--seed", str(seed)])
            del args.check
            for k, v in combo.items():
                setattr(args, k, v)
            out = find_run(args)
            if out is None:
                print(f"[{i}/{len(combos)}] {combo} seed {seed}: chạy mới")
                out = train(args)
            else:
                print(f"[{i}/{len(combos)}] {combo} seed {seed}: dùng lại {os.path.basename(out)}")
            m = json.load(open(os.path.join(out, "metrics.json")))
            n_params = json.load(open(os.path.join(out, "config.json")))["n_params"]
            rows.append(dict(combo, seed=seed, run=os.path.basename(out), n_params=n_params,
                             best_epoch=m["best_epoch"], MAE=m["val"]["MAE"], RMSE=m["val"]["RMSE"], R2=m["val"]["R2"],
                             MAE_15min=m["val"]["MAE_theo_buoc"][0], MAE_6h=m["val"]["MAE_theo_buoc"][-1]))
    df = pd.DataFrame(rows)
    keys = list(GRID)
    g = df.groupby(keys)
    summary = pd.concat([g["n_params"].first(),
                         g["MAE"].mean().rename("MAE_tb"), g["MAE"].std().rename("MAE_sd"),
                         g["RMSE"].mean().rename("RMSE_tb"), g["R2"].mean().rename("R2_tb"),
                         g["MAE_15min"].mean().rename("MAE_15min_tb"), g["MAE_6h"].mean().rename("MAE_6h_tb"),
                         g["best_epoch"].apply(lambda s: "/".join(map(str, s))).rename("epoch_tot_nhat"),
                         g["run"].apply(";".join).rename("runs")], axis=1).reset_index()
    summary = summary.sort_values("MAE_tb").round(4)
    path = os.path.join(RUNS, f"tuning_{opt.model}_{opt.cfg}.csv")
    summary.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.drop(columns="runs").to_string(index=False))
    print(f"đã ghi {path}")


if __name__ == "__main__":
    main()
