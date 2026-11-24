"""Tạo dataset/base_domain_splits/: base_splits cộng 24 đặc trưng miền (common.DOMAIN_FEATURES,
docs/03_DOMAIN_FEATURES_ANALYSIS.md).

1. Ước lượng hệ số trao đổi khí k trên các đêm của train (common.night_table, common.fit_reaeration), ghi
   artifacts/dataset/reaeration_k.csv: k (mỗi giờ), ER (hô hấp TB, mg/L mỗi giờ), r, số đêm.
2. Tính đặc trưng trên chuỗi nối ba tập đã làm sạch, liên tục qua ranh giới tập, nhân quả (common.domain_features).
   Mỗi dòng giữ nguyên văn dòng tương ứng của base_splits rồi nối thêm 24 cột, ghi 6 chữ số thập phân.
Chạy từ gốc repo (sau build_base_splits.py): uv run python scripts/build_domain_splits.py
"""
import csv
import os

from common import (BASE, DOMAIN, DOMAIN_FEATURES, NAMES, REAERATION, STATS, domain_features, fit_reaeration,
                    load_base_all, night_table, read_k)


def main():
    """Ước lượng k trên train, tính đặc trưng trên chuỗi nối ba tập, ghi reaeration_k.csv và ba file base_domain_splits/."""
    os.makedirs(DOMAIN, exist_ok=True)
    os.makedirs(STATS, exist_ok=True)
    d = load_base_all()
    fit = fit_reaeration(night_table(d))
    with open(REAERATION, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "er", "r", "n_nights"])
        w.writerow([f"{fit['k']:.6f}", f"{fit['er']:.6f}", f"{fit['r']:.4f}", fit["n_nights"]])
    k = read_k()                                   # dùng đúng giá trị đã ghi, như hệ thống web
    print(f"k = {k} mỗi giờ, ER = {fit['er']:.4f} mg/L mỗi giờ, r = {fit['r']:.3f}, {fit['n_nights']} đêm train")
    f = domain_features(d, k).to_numpy()
    a = 0
    for name in NAMES:
        with open(os.path.join(BASE, f"{name}.csv")) as src:
            lines = src.read().splitlines()
        b = a + len(lines) - 1
        with open(os.path.join(DOMAIN, f"{name}.csv"), "w") as out:
            out.write(lines[0] + "," + ",".join(DOMAIN_FEATURES) + "\n")
            for line, row in zip(lines[1:], f[a:b]):
                out.write(line + "," + ",".join(f"{round(x, 6) + 0.0:.6f}" for x in row) + "\n")   # + 0.0: không ghi -0
        print(f"{name}: {b - a} dòng, {len(DOMAIN_FEATURES)} đặc trưng miền")
        a = b


if __name__ == "__main__":
    main()
