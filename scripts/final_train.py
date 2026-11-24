"""Huấn luyện lại trên train cộng val với số epoch cố định, rồi chấm test đúng một lần
(docs/04_MODEL_TRAINING_EVALUATION.md mục 6).

Với mỗi bộ đặc trưng trong --sets:
  1. đọc ba lần chạy dừng sớm trên val (scripts/train.py, siêu tham số mặc định, seed 0, 1, 2);
     số epoch = trung bình epoch tốt nhất của ba lần đó, làm tròn;
  2. với từng seed: chuẩn hóa khớp trên train và val, huấn luyện trên mọi cửa sổ train và val đúng số epoch đó,
     không dừng sớm, giữ trọng số cuối; dự báo test.
Mỗi lần ghi artifacts/runs/<mô hình>_<cấu hình>/final/<bộ>_s<seed>/, ví dụ artifacts/runs/lstm_24h_6h/final/domain_s0/:
config.json, history.csv, metrics.json (chỉ số trên test), pred_test.npz (dự báo và DO thật, mg/L; mặt nạ DO cờ 0;
chỉ số dòng t) và model.pt. Bảng tổng hợp trong cùng thư mục final/: test.csv (từng seed) và test_tb.csv (trung bình,
độ lệch chuẩn giữa seed).
Test chỉ chấm một lần: script bỏ qua seed đã có lần chấm cho cùng mô hình, cấu hình, bộ đặc trưng (tránh chấm lại
sau khi đã thấy kết quả test). Tái lập: --verify huấn luyện lại đúng các lần đó vào thư mục tạm và so từng bit trọng số,
dự báo test với lần đã lưu (cùng máy, cùng phiên bản thư viện), không ghi lần chấm mới.
Chạy từ gốc repo: uv run python scripts/final_train.py --sets base domain
                  uv run python scripts/final_train.py --sets base domain --verify
"""
import os
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from common import CONFIGS, inverse_do, metrics, read_json, write_json
from train import (FEATURE_SETS, RNNForecaster, build_parser, data_fingerprint, final_dir, git_state, iter_runs,
                   make_data, masked_mse, predict, run_root, seed_all, val_dir)

SEEDS = (0, 1, 2)
HYPER = ("model", "cfg", "hidden", "layers", "dropout", "lr", "weight_decay", "batch", "clip")   # phải trùng mặc định


def val_runs(args, features):
    """Ba lần chạy dừng sớm trên val (seed 0, 1, 2) có đúng siêu tham số args và bộ đặc trưng features."""
    found = {}
    for d, c in iter_runs(val_dir(args.model, args.cfg, features)):
        if c.get("final") or c.get("features", "base") != features or any(c[h] != getattr(args, h) for h in HYPER):
            continue
        if c["seed"] in SEEDS and c.get("patience") == args.patience and c.get("epochs") == args.epochs:
            found[c["seed"]] = d
    missing = [s for s in SEEDS if s not in found]
    assert not missing, f"thiếu lần chạy val của bộ {features}, seed {missing}: chạy scripts/train.py trước"
    return [found[s] for s in SEEDS]


def scored_run(args, features, seed):
    """Thư mục lần chấm test đã có cho cùng mô hình, cấu hình, bộ đặc trưng và seed; None nếu chưa có."""
    for d, c in iter_runs(final_dir(args.model, args.cfg)):
        if (c["model"], c["cfg"], c["features"], c["seed"]) == (args.model, args.cfg, features, seed):
            return d
    return None


def verify_run(recorded, fresh):
    """So lần huấn luyện lại (fresh) với lần chấm đã lưu (recorded): trọng số và dự báo test có trùng từng bit."""
    a, b = (torch.load(os.path.join(x, "model.pt"), map_location="cpu") for x in (recorded, fresh))
    same_w = all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
    pa, pb = np.load(os.path.join(recorded, "pred_test.npz")), np.load(os.path.join(fresh, "pred_test.npz"))
    same_p = all(np.array_equal(pa[k], pb[k]) for k in pa.files)
    return same_w, same_p


def final_run(args, features, seed, n_epochs, source_runs, root=None):
    """Huấn luyện lại một seed trên train cộng val trong n_epochs epoch, chấm test, ghi thư mục <bộ>_s<seed> vào root
    (mặc định final_dir)."""
    root = root or final_dir(args.model, args.cfg)
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, scaler = make_data(args.cfg, args.batch, fit_on=("train", "val"), seed=seed, features=features)
    g = torch.Generator().manual_seed(seed)
    fit = DataLoader(ConcatDataset([loaders["train"].dataset, loaders["val"].dataset]), batch_size=args.batch,
                     shuffle=True, generator=g, pin_memory=True)
    n_in, n_out = CONFIGS[args.cfg]
    n_feat = loaders["train"].dataset.X.shape[1]
    model = RNNForecaster(args.model, n_feat, n_out, args.hidden, args.layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out = os.path.join(root, f"{features}_s{seed}")
    os.makedirs(out)
    config = {h: getattr(args, h) for h in HYPER}
    config.update(features=features, seed=seed, final=True, epochs=n_epochs, fit_on=["train", "val"],
                  val_runs=[os.path.relpath(r, run_root(args.model, args.cfg)) for r in source_runs], n_in=n_in, n_out=n_out, n_feat=n_feat,
                  n_params=sum(p.numel() for p in model.parameters()),
                  device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                  du_lieu_sha256=data_fingerprint(FEATURE_SETS[features][0]), so_cua_so_huan_luyen=len(fit.dataset),
                  so_cua_so_test=len(loaders["test"].dataset), **git_state())
    write_json(config, os.path.join(out, "config.json"), indent=1, ensure_ascii=False)
    hist = []
    for epoch in range(1, n_epochs + 1):
        t0, tot, n = time.time(), 0.0, 0
        model.train()
        for x, y, m in fit:
            x, y, m = (v.to(device, non_blocking=True) for v in (x, y, m))
            loss = masked_mse(model(x), y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            tot, n = tot + loss.item() * len(x), n + len(x)
        hist.append(dict(epoch=epoch, train_mse=tot / n, giay=round(time.time() - t0, 1)))
        print(f"  epoch {epoch:2d}  MSE huấn luyện {tot / n:.4f}  {hist[-1]['giay']} giây", flush=True)
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
    p, y, m = predict(model, loaders["test"], device)
    p, y = inverse_do(scaler, p), inverse_do(scaler, y)
    r = metrics(p, y, m)
    r["MAE_theo_buoc"] = [float(np.abs(p[:, k] - y[:, k])[m[:, k]].mean()) for k in range(n_out)]
    write_json(dict(test=r), os.path.join(out, "metrics.json"), indent=1)
    np.savez_compressed(os.path.join(out, "pred_test.npz"), pred=p.astype(np.float32), true=y.astype(np.float32),
                        mask=m, t=loaders["test"].dataset.t)
    torch.save(dict(state_dict=model.state_dict(), config=config, scaler_mean=scaler.mean_.tolist(),
                    scaler_scale=scaler.scale_.tolist()), os.path.join(out, "model.pt"))
    print(f"  {os.path.basename(out)}: test MAE {r['MAE']:.4f}  RMSE {r['RMSE']:.4f}  R2 {r['R2']:.4f}", flush=True)
    return out


def summary(args):
    """Ghi bảng tổng hợp mọi lần chấm test của mô hình và cấu hình: từng seed, rồi trung bình và độ lệch chuẩn."""
    rows = []
    for d, c in iter_runs(final_dir(args.model, args.cfg)):
        r = read_json(os.path.join(d, "metrics.json"))["test"]
        rows.append(dict(features=c["features"], seed=c["seed"], epochs=c["epochs"], MAE=r["MAE"], RMSE=r["RMSE"],
                         R2=r["R2"], MAE_15p=r["MAE_theo_buoc"][0], MAE_6h=r["MAE_theo_buoc"][-1],
                         run=os.path.basename(d)))
    t = pd.DataFrame(rows)
    num = ["MAE", "RMSE", "R2", "MAE_15p", "MAE_6h"]
    agg = t.groupby("features", sort=False)[num].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    path = os.path.join(final_dir(args.model, args.cfg), "test.csv")
    t.round(6).to_csv(path, index=False)
    agg.round(6).to_csv(path.replace(".csv", "_tb.csv"))
    print(agg.round(4).to_string())


def main():
    """Điểm vào: huấn luyện lại và chấm test cho từng bộ đặc trưng, rồi ghi bảng tổng hợp."""
    ap = build_parser()
    ap.add_argument("--sets", nargs="+", default=["base", "domain"], choices=list(FEATURE_SETS),
                    help="các bộ đặc trưng cần chấm test")
    ap.add_argument("--verify", action="store_true",
                    help="tái lập: huấn luyện lại các lần đã chấm vào thư mục tạm, so từng bit, không ghi lần chấm mới")
    args = ap.parse_args()
    verify = args.verify
    del args.verify
    failed = []
    for features in args.sets:
        src = val_runs(args, features)
        best = [read_json(os.path.join(r, "metrics.json"))["best_epoch"] for r in src]
        n_epochs = int(round(np.mean(best)))
        print(f"bộ {features}: epoch tốt nhất trên val {best}, huấn luyện lại {n_epochs} epoch")
        for seed in SEEDS:
            rec = scored_run(args, features, seed)
            if verify:
                if rec is None:
                    print(f"  seed {seed}: chưa có lần chấm để so")
                    continue
                with tempfile.TemporaryDirectory() as tmp:
                    fresh = final_run(args, features, seed, n_epochs, src, root=tmp)
                    same_w, same_p = verify_run(rec, fresh)
                print(f"  seed {seed}: trọng số {'trùng' if same_w else 'KHÁC'}, dự báo test {'trùng' if same_p else 'KHÁC'}"
                      f" với {os.path.basename(rec)}")
                if not (same_w and same_p):
                    failed.append((features, seed))
                continue
            if rec is not None:
                print(f"  seed {seed}: đã chấm test ({os.path.basename(rec)}), bỏ qua; test chỉ chấm một lần, "
                      "dùng --verify để kiểm tra tái lập")
                continue
            final_run(args, features, seed, n_epochs, src)
    if verify:
        print("REPRODUCED" if not failed else f"KHÔNG TRÙNG: {failed}")
        return
    summary(args)


if __name__ == "__main__":
    main()
