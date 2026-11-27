"""Căn cứ cho hệ số trao đổi khí k và ví dụ tính đặc trưng miền (docs/03_DOMAIN_FEATURES_ANALYSIS.md).

Ghi vào artifacts/dataset/:
  reaeration_k_ci.csv: khoảng tin cậy 95% của k bằng bootstrap (bốc lại các đêm train có lặp, 2000 lần, seed 0).
  night_regression.csv: mỗi đêm train một dòng: thiếu hụt trung bình, tốc độ thay đổi DO trung bình, số bước đêm,
    nhiệt độ trung bình; dữ liệu của hồi quy ban đêm và của fig_k_ban_dem.png.
  reaeration_k_by_period.csv: k ước lượng riêng theo từng năm có đủ 12 tháng trong train (2024 chỉ có tháng 1 đến 5
    nên không tính riêng) và theo từng giai đoạn mùa của train.
  reaeration_k_sensitivity.csv: tương quan của er_now, gpp_now tính với k khác so với bản dùng k đã ước lượng; k thử
    là hai đầu khoảng tin cậy và k nhỏ nhất, lớn nhất trong các k ước lượng riêng theo giai đoạn.
  er_temp_offset.csv: chênh lệch giữa nhiệt độ hiện tại và nhiệt độ TB các bước đêm trong 24 giờ qua.
  night_steps.csv: số bước đêm có DO đo thật của mỗi đêm train (nhỏ nhất, trung vị, lớn nhất, số đêm dưới ngưỡng
    NIGHT_MIN_STEPS) và số bước của đêm thiên văn ngắn nhất, dài nhất.
  domain_example.csv: dạng dài (ten, gia_tri): năm biến đo tại dòng ví dụ và 1 giờ, 3 giờ, 24 giờ trước, số ô điền
    trong 24 giờ trước, mọi đại lượng trung gian và 24 đặc trưng của một dòng train (ví dụ tính tay, mục 4).
  salinity_check.csv: độ mặn lớn nhất và chênh lệch DO bão hòa (mg/L) khi thay công thức độ mặn Schemel (2001)
    bằng phương trình 4 của USGS (2011), trên mọi dòng 2019 đến 2025.
  fig_k_ban_dem.png: các đêm train, trung bình theo thập phân vị của thiếu hụt và đường hồi quy.
  fig_dac_trung_vi_du.png: DO và một số đặc trưng miền trong ba ngày quanh dòng ví dụ.
Chạy từ gốc repo (sau build_domain_splits.py): uv run python scripts/domain_study.py
"""
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (DAY, FLAGS, LON, M2_EPOCH, NIGHT_MIN_STEPS, STATS, do_saturation, domain_features,
                    fit_reaeration, load_base_all, night_table, oxygen_steps, read_k, salinity, solar_elevation_sin)
from describe_dataset import season
from plot_style import BLUE, GREEN, INK, INK2, ORANGE, PURPLE, VERMILION, panel, save

EXAMPLE = "2021-07-15 15:00:00"        # dòng ví dụ: chiều mùa nóng trong train, 24 giờ trước không có ô điền


def example_rows(d, k, feats, se, dd, Dm, night):
    """Các đại lượng của dòng EXAMPLE theo thứ tự tính tay trong docs/03: (tên, giá trị)."""
    i = int(np.flatnonzero(d["datetime_pst"] == pd.Timestamp(EXAMPLE))[0])
    w = slice(i - DAY + 1, i + 1)                                      # 96 mốc gần nhất, kể cả mốc hiện tại
    t = d["datetime_pst"][i]
    do, T, ph, spc, chl = (d[c].to_numpy() for c in ("do_mg_l", "water_temp_c", "ph", "spc_us_cm", "chl_fluor_ug_l"))
    hr = t.hour + t.minute / 60
    g = 2 * np.pi / 365 * (t.day_of_year - 1 + (hr - 12) / 24)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g)
                    - 0.040849 * np.sin(2 * g))
    tst = hr * 60 + eqt + 4 * LON + 480
    R = spc[i] / 53087.0
    TK = T[i] + 273.15
    sat = do_saturation(d["water_temp_c"], d["spc_us_cm"]).to_numpy()
    light = np.clip(se, 0, None)
    nw = night[w]
    gpp24 = do[i] - do[i - DAY] - k * Dm[w].sum() * 0.25 + 24 * feats.er_now[i]
    rows = [(f"{name}{suf}", v[i - lag]) for suf, lag in [("", 0), ("_1h_truoc", 4), ("_3h_truoc", 12), ("_24h_truoc", DAY)]
            for name, v in [("do", do), ("temp", T), ("ph", ph), ("spc", spc), ("chl", chl)]]
    rows += [("so_o_dien_24h", int(d[FLAGS].to_numpy()[w].sum())),
            ("T_kelvin", TK), ("R", R), ("do_man_S", salinity(spc[i])), ("ln_do_sat", np.log(sat[i])),
            ("thu_tu_ngay_N", t.day_of_year), ("gio_thap_phan", hr), ("goc_ngay_rad", g), ("xich_vi_rad", decl),
            ("phuong_trinh_thoi_gian_phut", eqt), ("gio_mat_troi_phut", tst), ("goc_gio_do", tst / 4 - 180),
            ("sin_do_cao_mat_troi", se[i]),
            ("k", k), ("so_buoc_dem", int(nw.sum())), ("thieu_hut_dem_tb", Dm[w][nw].mean()),
            ("toc_do_do_dem_tb", dd[w][nw].mean()), ("tong_thieu_hut_x_0_25", Dm[w].sum() * 0.25),
            ("gpp_24h", gpp24), ("tong_anh_sang_x_0_25", light[w].sum() * 0.25),
            ("do_tb_24h", do[w].mean()),
            ("gio_tu_moc_m2", (t - pd.Timestamp(M2_EPOCH)).total_seconds() / 3600),
            ("phan_ngay_u", hr / 24), ("phan_nam_v", (t.day_of_year - 1 + hr / 24) / 365.25)]
    return rows + [(c, feats[c][i]) for c in feats.columns]


def fig_night(k, er):
    """Hồi quy ban đêm: mỗi đêm train một chấm, trung bình theo thập phân vị của thiếu hụt, đường hồi quy."""
    nights = pd.read_csv(os.path.join(STATS, "night_regression.csv")).rename(
        columns={"thieu_hut_tb": "D", "toc_do_tb": "dd"})
    D, dd = nights.D.to_numpy(), nights.dd.to_numpy()
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.scatter(D, dd, s=4, alpha=0.3, color=BLUE, linewidths=0, label="Một đêm train")
    g = nights.groupby(pd.qcut(nights.D, 10), observed=True)[["D", "dd"]].mean()   # 10 nhóm, mỗi nhóm 10% số đêm
    ax.plot(g.D, g.dd, "o", ms=5, color=VERMILION, label="Trung bình theo thập phân vị thiếu hụt")
    x = np.linspace(D.min(), D.max(), 2)
    ax.plot(x, k * x - er, color=INK, lw=1.2, label="Đường hồi quy")
    ax.axhline(0, color=INK2, lw=0.5)
    ax.axvline(0, color=INK2, lw=0.5)
    ax.set_ylim(np.percentile(dd, 0.5), np.percentile(dd, 99.5))
    ax.set_xlabel("Thiếu hụt oxy trung bình của đêm (mg/L)")
    ax.set_ylabel("Tốc độ thay đổi DO\ntrung bình của đêm (mg/L mỗi giờ)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, os.path.join(STATS, "fig_k_ban_dem.png"))


def fig_example(d, feats):
    """Bốn khung, ba ngày quanh dòng EXAMPLE: DO và DO bão hòa; light; gpp_now và er_now; do_anom_24h và do_std_24h."""
    t0 = pd.Timestamp(EXAMPLE)
    m = ((d["datetime_pst"] >= t0 - pd.Timedelta("2D15h")) & (d["datetime_pst"] < t0 + pd.Timedelta("9h"))).to_numpy()
    t, f = d["datetime_pst"][m], feats[m]
    panels = [([("DO", d["do_mg_l"][m], BLUE, "-"), ("do_sat_mgl", f.do_sat_mgl, INK, "--")], "mg/L"),
              ([("light", f.light, ORANGE, "-")], "Ánh sáng"),
              ([("gpp_now", f.gpp_now, GREEN, "-"), ("er_now", f.er_now, VERMILION, "-")], "mg/L mỗi giờ"),
              ([("do_anom_24h", f.do_anom_24h, BLUE, "-"), ("do_std_24h", f.do_std_24h, PURPLE, "-")], "mg/L")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7, 5.4), sharex=True)
    for ax, (lines, ylab), key in zip(axes, panels, "abcd"):
        for name, v, c, ls in lines:
            ax.plot(t, v, color=c, ls=ls, lw=1.1, label=name)
        ax.axvline(t0, color=INK2, lw=0.7, ls=":")
        ax.set_ylabel(ylab)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        panel(ax, key)
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
    fig.tight_layout()
    save(fig, os.path.join(STATS, "fig_dac_trung_vi_du.png"))


def main():
    """Tính và ghi các bảng, hình của domain_study."""
    d = load_base_all()
    k = read_k()
    se = solar_elevation_sin(d["datetime_pst"])
    nights = night_table(d, se)
    nights.rename(columns={"D": "thieu_hut_tb", "dd": "toc_do_tb", "n": "so_buoc", "T": "nhiet_do_tb"})[
        ["thieu_hut_tb", "toc_do_tb", "so_buoc", "nhiet_do_tb"]].round(6).rename_axis("dem").to_csv(
        os.path.join(STATS, "night_regression.csv"))
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
    full = [y for y, g in nights.groupby(idx.year) if pd.to_datetime(g.index).month.nunique() == 12]  # năm đủ 12 tháng
    rows += [dict(nhom="nam", ky=str(y), **fit_reaeration(g)) for y, g in nights.groupby(idx.year) if y in full]
    rows += [dict(nhom="mua", ky=s, **fit_reaeration(g)) for s, g in nights.groupby(idx.month.map(season))]
    pd.DataFrame(rows).round(4).to_csv(os.path.join(STATS, "reaeration_k_by_period.csv"), index=False)
    # 3. độ nhạy của er_now, gpp_now với k
    base = domain_features(d, k)
    sens = []
    ks = [r["k"] for r in rows[1:]]                                  # k ước lượng riêng theo năm, theo mùa
    for kk in sorted([lo, hi, min(ks), max(ks)]):
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
    pd.DataFrame([dict(median=np.median(off), p95=np.percentile(off, 95), n=len(off))]).round(6).to_csv(
        os.path.join(STATS, "er_temp_offset.csv"), index=False)
    # 4b. số bước đêm mỗi đêm train: có số đo thật (night) và theo thiên văn (mặt trời dưới chân trời ở hai đầu bước)
    tr = (d["split"] == "train").to_numpy()
    lab = (d["datetime_pst"] + pd.Timedelta(hours=12)).dt.date.to_numpy()
    astro = (se < 0) & (np.roll(se, 1) < 0)
    n_obs = pd.Series(lab[night & tr]).value_counts()
    n_astro = pd.Series(lab[astro & tr]).value_counts().sort_index().iloc[1:-1]   # bỏ hai đêm bị cắt ở hai đầu train
    pd.DataFrame([dict(so_dem=len(n_obs), so_dem_duoi_nguong=int((n_obs < NIGHT_MIN_STEPS).sum()),
                       nguong=NIGHT_MIN_STEPS, buoc_min=n_obs.min(), buoc_trung_vi=n_obs.median(),
                       buoc_max=n_obs.max(), dem_thien_van_ngan_nhat=n_astro.min(),
                       dem_thien_van_dai_nhat=n_astro.max())]).to_csv(os.path.join(STATS, "night_steps.csv"), index=False)
    # 5. dòng ví dụ: mọi đại lượng đầu vào, trung gian và 24 đặc trưng (docs/03 mục 4)
    pd.DataFrame(example_rows(d, k, base, se, ddv, Dm, night), columns=["ten", "gia_tri"]).round(6).to_csv(
        os.path.join(STATS, "domain_example.csv"), index=False)
    # 6. độ mặn: công thức Schemel (2001) dùng trong đặc trưng so với phương trình 4 của USGS (2011)
    spc = d["spc_us_cm"].to_numpy()
    sat = base.do_sat_mgl.to_numpy()
    s_usgs = 5.572e-4 * spc + 2.02e-9 * spc**2
    gap = np.abs(do_saturation(d["water_temp_c"], spc, s_usgs).to_numpy() - sat)
    pd.DataFrame([dict(spc_max=spc.max(), sal_schemel_max=salinity(spc).max(), sal_usgs_max=s_usgs.max(),
                       do_sat_diff_max=gap.max(), do_sat_diff_p99=np.percentile(gap, 99), n=len(gap))]).round(6).to_csv(
        os.path.join(STATS, "salinity_check.csv"), index=False)
    # 7. hình
    fig_night(k, rows[0]["er"])
    fig_example(d, base)
    print(f"k = {k}, 95%: {lo:.4f} đến {hi:.4f}; ví dụ {EXAMPLE}; |T - T đêm| trung vị {np.median(off):.2f}")


if __name__ == "__main__":
    main()
