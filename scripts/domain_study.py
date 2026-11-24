"""Căn cứ cho hệ số trao đổi khí k và ví dụ tính đặc trưng miền (docs/03_DOMAIN_FEATURES_ANALYSIS.md).

Ghi vào artifacts/dataset/:
  reaeration_k_ci.csv: khoảng tin cậy 95% của k bằng bootstrap (bốc lại các đêm train có lặp, 2000 lần, seed 0).
  reaeration_k_by_period.csv: k ước lượng riêng theo từng năm và từng giai đoạn mùa của train.
  reaeration_k_sensitivity.csv: tương quan của er_now, gpp_now tính với k khác so với bản dùng k đã ước lượng.
  er_temp_offset.csv: chênh lệch giữa nhiệt độ hiện tại và nhiệt độ TB các bước đêm trong 24 giờ qua.
  domain_example.csv: các đại lượng trung gian và 24 đặc trưng của một dòng train (ví dụ tính tay trong tài liệu).
  fig_k_ban_dem.png: các đêm train, trung bình theo nhóm thiếu hụt và đường hồi quy.
Chạy từ gốc repo (sau build_domain_splits.py): uv run python scripts/domain_study.py
"""
import os

import matplotlib
import numpy as np
import pandas as pd

from common import (DAY, NIGHT_MIN_STEPS, STATS, do_saturation, domain_features, fit_reaeration, load_base_all,
                    night_table, oxygen_steps, read_k, solar_elevation_sin)
from describe_dataset import season

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXAMPLE = "2021-07-15 15:00:00"        # dòng ví dụ: chiều mùa nóng trong train
K_TEST = [0.01, 0.03, 0.04, 0.08]      # k thử cho độ nhạy, ngoài hai đầu khoảng tin cậy


def main():
    """Tính và ghi các bảng, hình của domain_study."""
    d = load_base_all()
    k = read_k()
    se = solar_elevation_sin(d["datetime_pst"])
    nights = night_table(d, se)
    # 1. khoảng tin cậy bootstrap
    rng = np.random.default_rng(0)
    D, dd = nights.D.to_numpy(), nights.dd.to_numpy()
    boot = []
    for _ in range(2000):
        i = rng.integers(0, len(D), len(D))
        boot.append(np.polyfit(D[i], dd[i], 1)[0])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    pd.DataFrame([dict(k=k, k_p2_5=lo, k_p97_5=hi, n_resamples=2000, seed=0)]).round(6).to_csv(
        os.path.join(STATS, "reaeration_k_ci.csv"), index=False)
    # 2. k theo năm và theo mùa
    idx = pd.to_datetime(nights.index)
    rows = [dict(nhom="toan_bo", ky="train", **fit_reaeration(nights))]
    rows += [dict(nhom="nam", ky=str(y), **fit_reaeration(g)) for y, g in nights.groupby(idx.year)]
    rows += [dict(nhom="mua", ky=s, **fit_reaeration(g)) for s, g in nights.groupby(idx.month.map(season))]
    pd.DataFrame(rows).round(4).to_csv(os.path.join(STATS, "reaeration_k_by_period.csv"), index=False)
    # 3. độ nhạy của er_now, gpp_now với k
    base = domain_features(d, k)
    sens = []
    for kk in sorted([lo, hi] + K_TEST):
        f = domain_features(d, kk)
        sens.append(dict(k=kk, corr_er_now=np.corrcoef(base.er_now, f.er_now)[0, 1],
                         corr_gpp_now=np.corrcoef(base.gpp_now, f.gpp_now)[0, 1]))
    pd.DataFrame(sens).round(4).to_csv(os.path.join(STATS, "reaeration_k_sensitivity.csv"), index=False)
    # 4. nhiệt độ hiện tại so với nhiệt độ đêm gần nhất (căn cứ không quy đổi hô hấp theo nhiệt độ)
    _, ddv, Dm, night = oxygen_steps(d, se)
    S = lambda v: pd.Series(v).rolling(DAY, min_periods=1).sum().to_numpy()   # noqa: E731
    n_night = S(night.astype(float))
    T = d["water_temp_c"].to_numpy()
    Tn = S(np.where(night, T, 0)) / np.maximum(n_night, 1)
    off = np.abs(T - Tn)[n_night >= NIGHT_MIN_STEPS]
    pd.DataFrame([dict(median=np.median(off), p95=np.percentile(off, 95), n=len(off))]).round(3).to_csv(
        os.path.join(STATS, "er_temp_offset.csv"), index=False)
    # 5. dòng ví dụ
    i = int(np.flatnonzero(d["datetime_pst"] == pd.Timestamp(EXAMPLE))[0])
    w = slice(i - DAY + 1, i + 1)
    sat = do_saturation(d["water_temp_c"], d["spc_us_cm"]).to_numpy()
    ex = dict(datetime_pst=EXAMPLE, do=d.do_mg_l[i], temp=T[i], spc=d.spc_us_cm[i], chl=d.chl_fluor_ug_l[i],
              do_sat=sat[i], light=max(se[i], 0), n_night_steps=int(night[w].sum()),
              night_mean_D=Dm[w][night[w]].mean(), night_mean_dd=ddv[w][night[w]].mean(),
              do_24h_before=d.do_mg_l[i - DAY], sum_D_x_0_25=Dm[w].sum() * 0.25,
              sum_light_x_0_25=np.clip(se[w], 0, None).sum() * 0.25, k=k)
    ex.update({c: base[c][i] for c in base.columns})
    pd.DataFrame([ex]).round(6).to_csv(os.path.join(STATS, "domain_example.csv"), index=False)
    # 6. hình
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    ax.scatter(D, dd, s=5, alpha=0.25, color="#2a78d6", label="một đêm train")
    bins = pd.cut(nights.D, [-6, -2, -1, 0, 0.5, 1, 1.5, 3])
    g = nights.groupby(bins, observed=True)[["D", "dd"]].mean()
    ax.scatter(g.D, g.dd, s=45, color="#eb6834", zorder=3, label="trung bình theo nhóm thiếu hụt")
    x = np.linspace(D.min(), D.max(), 2)
    er = rows[0]["er"]
    ax.plot(x, k * x - er, color="black", lw=1.5, label=f"hồi quy: k = {k:.4f} mỗi giờ")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_ylim(np.percentile(dd, 0.5), np.percentile(dd, 99.5))
    ax.set_xlabel("thiếu hụt TB đêm (mg/L): âm là thừa oxy, dương là thiếu oxy")
    ax.set_ylabel("DO thay đổi TB đêm (mg/L mỗi giờ)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(STATS, "fig_k_ban_dem.png"))
    print(f"k = {k}, 95%: {lo:.4f} đến {hi:.4f}; ví dụ {EXAMPLE}; |T - T đêm| trung vị {np.median(off):.2f}")


if __name__ == "__main__":
    main()
