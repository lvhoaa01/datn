"""Huấn luyện mô hình dự báo DO. Hiện có phần dữ liệu; phần mô hình và vòng huấn luyện bổ sung sau.

Dữ liệu: base_splits chuẩn hóa bằng StandardScaler khớp trên train (common.fit_scaler), cửa sổ theo
common.window_starts, cắt tại chỗ khi nạp, không lưu cửa sổ ra đĩa. Mỗi mẫu gồm
  x: (n_in, 12) đầu vào [t - n_in, t): sáu biến đã chuẩn hóa, rồi sáu cờ điền (1 ô được điền, 0 số đo thật;
     cửa sổ chứa ô cờ -1 đã bị loại nên cờ chỉ còn 0 hoặc 1);
  y: (n_out,) DO đã chuẩn hóa của nhãn [t, t + n_out);
  m: (n_out,) True ở mốc nhãn có DO cờ 0; chỉ các mốc này được tính sai số và hàm mất mát.
Chạy từ gốc repo: uv run python scripts/train.py --check   (kiểm tra phần dữ liệu của cả hai cấu hình)
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import CONFIGS, FEATURES, FILLED, FLAGS, NAMES, OBSERVED, fit_scaler, inverse_do, load_base_all, window_starts

# số cửa sổ dùng được của từng tập, khớp README mục 5.2
EXPECTED = {"24h_6h": {"train": 189769, "val": 25520, "test": 26183},
            "7d_24h": {"train": 189121, "val": 24161, "test": 25472}}


class WindowDataset(Dataset):
    """Các cửa sổ (x, y, m) của một tập, cắt từ mảng chung X đã chuẩn hóa tại các thời điểm dự báo starts."""

    def __init__(self, X, obs, starts, n_in, n_out):
        """X: (n, 12) tensor chung; obs: (n,) mặt nạ DO cờ 0; starts: chỉ số t của các cửa sổ."""
        self.X, self.obs, self.t = X, obs, starts
        self.n_in, self.n_out = n_in, n_out

    def __len__(self):
        """Số cửa sổ."""
        return len(self.t)

    def __getitem__(self, i):
        """Cửa sổ thứ i: (x, y, m), là view của X và obs, không sao chép."""
        t = int(self.t[i])
        return self.X[t - self.n_in:t], self.X[t:t + self.n_out, 0], self.obs[t:t + self.n_out]


def make_data(cfg, batch_size=256, fit_on=("train",), d=None):
    """DataLoader của ba tập cho cấu hình cfg, kèm scaler đã khớp trên fit_on; chỉ train được xáo trộn."""
    n_in, n_out = CONFIGS[cfg]
    d = load_base_all() if d is None else d
    scaler = fit_scaler(d, fit_on)
    Z = scaler.transform(d[FEATURES].to_numpy())
    X = torch.as_tensor(np.hstack([Z, d[FLAGS].to_numpy() == FILLED]), dtype=torch.float32)
    obs = torch.as_tensor(d["flag_do_mg_l"].to_numpy() == OBSERVED)
    starts = window_starts(d, n_in, n_out)
    loaders = {n: DataLoader(WindowDataset(X, obs, starts[n], n_in, n_out), batch_size=batch_size,
                             shuffle=n == "train") for n in NAMES}
    return loaders, scaler


def check():
    """Kiểm tra: train sau chuẩn hóa có trung bình 0, độ lệch chuẩn 1; phép ngược trả đúng mg/L; số cửa sổ khớp
    README; không cửa sổ nào chứa NaN; cột cờ chỉ gồm 0 và 1, khớp base_splits; nhãn của mọi cửa sổ có ít nhất một
    mốc DO cờ 0."""
    d = load_base_all()
    scaler = fit_scaler(d)
    Z = scaler.transform(d.loc[d["split"] == "train", FEATURES].to_numpy())
    assert np.allclose(Z.mean(0), 0, atol=1e-9) and np.allclose(Z.std(0), 1, atol=1e-9)
    do = d["do_mg_l"].to_numpy()
    assert np.allclose(inverse_do(scaler, scaler.transform(d[FEATURES].to_numpy())[:, 0]), do, equal_nan=True)
    for c, mu, sd in zip(FEATURES, scaler.mean_, scaler.scale_):
        print(f"{c:15s} trung bình {mu:9.3f}  độ lệch chuẩn {sd:8.3f}")
    for cfg in CONFIGS:
        loaders, _ = make_data(cfg, batch_size=4096, d=d)
        for n, dl in loaders.items():
            count = bad = filled = 0
            for x, y, m in dl:
                count += len(x)
                f = x[:, :, 6:]
                bad += int((torch.isnan(x).any((1, 2)) | torch.isnan(y).any(1) | ~m.any(1)
                            | ((f != 0) & (f != 1)).any((1, 2))).sum())
                filled += int(f[:, -1].sum())
            t = dl.dataset.t
            ref = int((d[FLAGS].to_numpy()[t - 1] == FILLED).sum())   # cờ ở mốc cuối đầu vào, đọc thẳng từ d
            assert filled == ref, (cfg, n, filled, ref)
            assert count == EXPECTED[cfg][n] and bad == 0, (cfg, n, count, bad)
            print(f"{cfg} {n:5s}: {count} cửa sổ, x {tuple(x.shape[1:])}, y {tuple(y.shape[1:])}, đạt")
    print("DATA CHECKS PASSED")


def main():
    """Điểm vào; hiện chỉ có --check."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="kiểm tra phần dữ liệu")
    if ap.parse_args().check:
        check()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
