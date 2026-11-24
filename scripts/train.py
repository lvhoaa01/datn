"""Huấn luyện và đánh giá mô hình dự báo DO nhiều bước (LSTM, GRU).

Dữ liệu: --features base (thí nghiệm dữ liệu gốc) đọc base_splits; --features domain (thêm đặc trưng miền) đọc
base_domain_splits. Các cột số chuẩn hóa bằng StandardScaler khớp trên train (common.fit_scaler), cửa sổ theo
common.window_starts, cắt tại chỗ khi nạp, không lưu cửa sổ ra đĩa. Mỗi mẫu gồm
  x: (n_in, n_feat) đầu vào [t - n_in, t): các cột số đã chuẩn hóa (sáu biến, rồi 24 đặc trưng miền nếu domain),
     rồi sáu cờ điền (1 ô được điền, 0 số đo thật); n_feat là 12 (base) hoặc 36 (domain);
  y: (n_out,) DO đã chuẩn hóa của nhãn [t, t + n_out);
  m: (n_out,) True ở mốc nhãn có DO cờ 0; chỉ các mốc này được tính sai số và hàm mất mát.
Mô hình: mạng hồi quy (LSTM hoặc GRU) đọc đầu vào, trạng thái ẩn ở mốc cuối qua một lớp tuyến tính ra thẳng n_out
bước DO (dự báo trực tiếp). Huấn luyện trên train với AdamW, hàm mất mát MSE có mặt nạ, dừng sớm theo MAE (mg/L)
trên val; test không được chấm ở đây mà ở scripts/final_train.py, đúng một lần sau khi chọn xong cấu hình
(docs/04_MODEL_TRAINING_EVALUATION.md mục 6).
Mỗi lần chạy ghi artifacts/runs/<mô hình>_<cấu hình>/<tuning hoặc domain>/<tên theo siêu tham số và seed>/, ví dụ
artifacts/runs/lstm_24h_6h/tuning/lr0.001_h128_do0.3_s0/: config.json, history.csv, metrics.json và model.pt (trọng số
kèm trung bình, độ lệch chuẩn của scaler; không đưa lên git). Chạy lại cùng tham số thì ghi đè thư mục đó.
Chạy từ gốc repo: uv run python scripts/train.py --model lstm --cfg 24h_6h
                  uv run python scripts/train.py --check   (kiểm tra phần dữ liệu của cả hai cấu hình)
"""
import argparse
import glob
import os
import random
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import (BASE, CONFIGS, DOMAIN, DOMAIN_FEATURES, FEATURES, FILLED, FLAGS, NAMES, OBSERVED, REPO,
                    data_fingerprint, fit_scaler, inverse_do, load_base_all, metrics, read_json, window_starts, write_json)

RUNS = os.path.join(REPO, "artifacts", "runs")
# thư mục con theo vai trò: mọi lần chạy E1 có dừng sớm trên val thuộc lưới dò siêu tham số (tune.py)
VAL_GROUP = {"base": "tuning", "domain": "domain"}
NAMED_DEFAULTS = ("layers", "weight_decay", "batch", "epochs", "patience", "clip")   # chỉ ghi vào tên khi khác mặc định
# số cửa sổ dùng được của từng tập, khớp docs/04_MODEL_TRAINING_EVALUATION.md mục 2
EXPECTED = {"24h_6h": {"train": 189345, "val": 26185, "test": 26377},
            "7d_24h": {"train": 188896, "val": 26113, "test": 26305}}
# bộ đặc trưng: (thư mục dữ liệu, các cột số được chuẩn hóa); cờ luôn nối sau các cột số
FEATURE_SETS = {"base": (BASE, FEATURES), "domain": (DOMAIN, FEATURES + DOMAIN_FEATURES)}


class WindowDataset(Dataset):
    """Các cửa sổ (x, y, m) của một tập, cắt từ mảng chung X đã chuẩn hóa tại các thời điểm dự báo starts."""

    def __init__(self, X, obs, starts, n_in, n_out):
        """X: (n, n_feat) tensor chung; obs: (n,) mặt nạ DO cờ 0; starts: chỉ số t của các cửa sổ."""
        self.X, self.obs, self.t = X, obs, starts
        self.n_in, self.n_out = n_in, n_out

    def __len__(self):
        """Số cửa sổ."""
        return len(self.t)

    def __getitem__(self, i):
        """Cửa sổ thứ i: (x, y, m), là view của X và obs, không sao chép."""
        t = int(self.t[i])
        return self.X[t - self.n_in:t], self.X[t:t + self.n_out, 0], self.obs[t:t + self.n_out]


def make_data(cfg, batch_size=256, fit_on=("train",), d=None, seed=0, features="base"):
    """DataLoader của ba tập cho cấu hình cfg và bộ đặc trưng features, kèm scaler đã khớp trên fit_on; chỉ train được
    xáo trộn (theo seed)."""
    n_in, n_out = CONFIGS[cfg]
    folder, cols = FEATURE_SETS[features]
    d = load_base_all(folder) if d is None else d
    scaler = fit_scaler(d, fit_on, cols)
    Z = scaler.transform(d[cols].to_numpy())
    X = torch.as_tensor(np.hstack([Z, d[FLAGS].to_numpy() == FILLED]), dtype=torch.float32)
    obs = torch.as_tensor(d["flag_do_mg_l"].to_numpy() == OBSERVED)
    starts = window_starts(d, n_in, n_out)
    g = torch.Generator().manual_seed(seed)
    loaders = {n: DataLoader(WindowDataset(X, obs, starts[n], n_in, n_out), batch_size=batch_size,
                             shuffle=n == "train", generator=g if n == "train" else None, pin_memory=True)
               for n in NAMES}
    return loaders, scaler


class RNNForecaster(nn.Module):
    """Mạng hồi quy nhiều lớp đọc đầu vào; trạng thái ẩn ở mốc cuối qua lớp tuyến tính ra n_out bước DO."""

    def __init__(self, cell, n_feat, n_out, hidden, layers, dropout):
        """cell: "lstm" hoặc "gru"; dropout áp giữa các lớp hồi quy."""
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(n_feat, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):
        """x: (batch, n_in, n_feat) -> (batch, n_out) DO đã chuẩn hóa."""
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


def run_root(model, cfg):
    """Thư mục chứa mọi lần chạy của một mô hình và một cấu hình dự báo: artifacts/runs/<mô hình>_<cấu hình>/."""
    return os.path.join(RUNS, f"{model}_{cfg}")


def val_dir(model, cfg, features):
    """Thư mục các lần chạy có dừng sớm trên val của một bộ đặc trưng (tuning/ với E1, domain/ với E2)."""
    return os.path.join(run_root(model, cfg), VAL_GROUP.get(features, features))


def final_dir(model, cfg):
    """Thư mục các lần huấn luyện lại trên train cộng val và chấm test (scripts/final_train.py)."""
    return os.path.join(run_root(model, cfg), "final")


def run_name(args):
    """Tên thư mục của một lần chạy có dừng sớm theo siêu tham số và seed, ví dụ lr0.001_h128_do0.3_s0; siêu tham số
    khác giá trị mặc định của build_parser được nối thêm vào tên."""
    ap = build_parser()
    name = f"lr{args.lr:g}_h{args.hidden}_do{args.dropout:g}"
    name += "".join(f"_{k}{getattr(args, k):g}" for k in NAMED_DEFAULTS if getattr(args, k) != ap.get_default(k))
    return f"{name}_s{args.seed}"


def iter_runs(root=RUNS):
    """Mọi lần chạy dưới root, theo thứ tự đường dẫn: các cặp (thư mục, config)."""
    for f in sorted(glob.glob(os.path.join(root, "**", "config.json"), recursive=True)):
        yield os.path.dirname(f), read_json(f)


def masked_mse(pred, y, m):
    """MSE chỉ trên các mốc m (DO cờ 0)."""
    m = m.float()
    return ((pred - y) ** 2 * m).sum() / m.sum()


@torch.no_grad()
def predict(model, loader, device):
    """Dự báo cả tập: (pred, y, m) dạng numpy, DO còn ở thang chuẩn hóa."""
    model.eval()
    P, Y, M = [], [], []
    for x, y, m in loader:
        P.append(model(x.to(device, non_blocking=True)).cpu())
        Y.append(y)
        M.append(m)
    return torch.cat(P).numpy(), torch.cat(Y).numpy(), torch.cat(M).numpy()


def evaluate(model, loader, scaler, device):
    """MAE, RMSE, R2 (mg/L) trên các mốc DO cờ 0, kèm MAE theo từng bước dự báo."""
    p, y, m = predict(model, loader, device)
    p, y = inverse_do(scaler, p), inverse_do(scaler, y)
    r = metrics(p, y, m)
    r["MAE_theo_buoc"] = [float(np.abs(p[:, k] - y[:, k])[m[:, k]].mean()) for k in range(y.shape[1])]
    return r


def git_state():
    """Commit hiện tại và cờ git_dirty: True nếu lúc chạy có thay đổi chưa commit trong các file git theo dõi, tức kết
    quả không gắn trọn với commit đó."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip())
    return dict(git_commit=commit, git_dirty=dirty)


def seed_all(seed):
    """Cố định seed của random, numpy và torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args):
    """Huấn luyện trên train, dừng sớm theo MAE val, ghi thư mục kết quả; trả về đường dẫn thư mục."""
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, scaler = make_data(args.cfg, args.batch, seed=args.seed, features=args.features)
    n_in, n_out = CONFIGS[args.cfg]
    n_feat = loaders["train"].dataset.X.shape[1]
    model = RNNForecaster(args.model, n_feat, n_out, args.hidden, args.layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out = os.path.join(val_dir(args.model, args.cfg, args.features), run_name(args))
    name = os.path.relpath(out, RUNS)
    if os.path.isdir(out):
        print(f"{name}: đã có, ghi đè (cùng seed và dữ liệu cho kết quả trùng từng bit)")
        shutil.rmtree(out)
    os.makedirs(out)
    config = dict(vars(args), n_in=n_in, n_out=n_out, n_params=sum(p.numel() for p in model.parameters()),
                  device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", **git_state(),
                  du_lieu_sha256=data_fingerprint(FEATURE_SETS[args.features][0]), n_feat=n_feat,
                  so_cua_so={n: len(dl.dataset) for n, dl in loaders.items()})
    write_json(config, os.path.join(out, "config.json"), indent=1, ensure_ascii=False)
    print(f"{name}: {config['n_params']} tham số, {config['device']}")
    hist, best, best_state, best_epoch, wait = [], np.inf, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        t0, tot, n = time.time(), 0.0, 0
        model.train()
        for x, y, m in loaders["train"]:
            x, y, m = (v.to(device, non_blocking=True) for v in (x, y, m))
            loss = masked_mse(model(x), y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            tot, n = tot + loss.item() * len(x), n + len(x)
        val = evaluate(model, loaders["val"], scaler, device)
        hist.append(dict(epoch=epoch, train_mse=tot / n, val_MAE=val["MAE"], val_RMSE=val["RMSE"], val_R2=val["R2"],
                         giay=round(time.time() - t0, 1)))
        print(f"epoch {epoch:3d}  train MSE {tot / n:.4f}  val MAE {val['MAE']:.4f}  RMSE {val['RMSE']:.4f}  "
              f"R2 {val['R2']:.4f}  {hist[-1]['giay']} giây")
        if val["MAE"] < best - 1e-4:
            best, best_epoch, wait = val["MAE"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
    model.load_state_dict(best_state)
    val = evaluate(model, loaders["val"], scaler, device)
    write_json(dict(best_epoch=best_epoch, val=val), os.path.join(out, "metrics.json"), indent=1)
    torch.save(dict(state_dict=best_state, config=config, scaler_mean=scaler.mean_.tolist(),
                    scaler_scale=scaler.scale_.tolist()), os.path.join(out, "model.pt"))
    print(f"tốt nhất ở epoch {best_epoch}: val MAE {val['MAE']:.4f}  RMSE {val['RMSE']:.4f}  R2 {val['R2']:.4f}")
    return out


def check():
    """Kiểm tra: train sau chuẩn hóa có trung bình 0, độ lệch chuẩn 1; phép ngược trả đúng mg/L; số cửa sổ khớp
    docs/04_MODEL_TRAINING_EVALUATION.md mục 2; không cửa sổ nào chứa NaN; cột cờ chỉ gồm 0 và 1, khớp base_splits; nhãn của mọi cửa sổ có ít nhất một
    mốc DO cờ 0. Với bộ domain: các cột gốc và cờ trùng base_splits, đầu vào có 36 cột."""
    d = load_base_all()
    scaler = fit_scaler(d)
    Z = scaler.transform(d.loc[d["split"] == "train", FEATURES].to_numpy())
    assert np.allclose(Z.mean(0), 0, atol=1e-9) and np.allclose(Z.std(0), 1, atol=1e-9)
    do = d["do_mg_l"].to_numpy()
    assert np.allclose(inverse_do(scaler, scaler.transform(d[FEATURES].to_numpy())[:, 0]), do)
    for c, mu, sd in zip(FEATURES, scaler.mean_, scaler.scale_):
        print(f"{c:15s} trung bình {mu:9.3f}  độ lệch chuẩn {sd:8.3f}")
    dd = load_base_all(DOMAIN)
    assert dd[["datetime_pst"] + FEATURES + FLAGS].equals(d[["datetime_pst"] + FEATURES + FLAGS])
    for features, cfg in [(f, c) for f in FEATURE_SETS for c in CONFIGS]:
        loaders, _ = make_data(cfg, batch_size=4096, d=dd if features == "domain" else d, features=features)
        n_feat = len(FEATURE_SETS[features][1]) + len(FLAGS)
        for n, dl in loaders.items():
            count = bad = filled = 0
            for x, y, m in dl:
                count += len(x)
                assert x.shape[2] == n_feat
                f = x[:, :, -len(FLAGS):]
                bad += int((torch.isnan(x).any((1, 2)) | torch.isnan(y).any(1) | ~m.any(1)
                            | ((f != 0) & (f != 1)).any((1, 2))).sum())
                filled += int(f[:, -1].sum())
            t = dl.dataset.t
            ref = int((d[FLAGS].to_numpy()[t - 1] == FILLED).sum())   # cờ ở mốc cuối đầu vào, đọc thẳng từ d
            assert filled == ref, (cfg, n, filled, ref)
            assert count == EXPECTED[cfg][n] and bad == 0, (cfg, n, count, bad)
            print(f"{features:6s} {cfg} {n:5s}: {count} cửa sổ, x {tuple(x.shape[1:])}, y {tuple(y.shape[1:])}, đạt")
    print("DATA CHECKS PASSED")


def build_parser():
    """Bộ đọc tham số dòng lệnh, giữ giá trị mặc định của mọi siêu tham số (tune.py dùng lại)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="kiểm tra phần dữ liệu")
    ap.add_argument("--model", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--cfg", default="24h_6h", choices=list(CONFIGS))
    ap.add_argument("--features", default="base", choices=list(FEATURE_SETS),
                    help="base: dữ liệu gốc; domain: thêm 24 đặc trưng miền")
    # mặc định là cấu hình chốt sau dò siêu tham số trên val (artifacts/runs/lstm_24h_6h/tuning/tuning.csv)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5, help="số epoch không cải thiện val MAE trước khi dừng")
    ap.add_argument("--clip", type=float, default=1.0, help="ngưỡng chuẩn gradient")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main():
    """Điểm vào: --check hoặc huấn luyện một mô hình."""
    args = build_parser().parse_args()
    if args.check:
        check()
    else:
        del args.check
        train(args)


if __name__ == "__main__":
    main()
