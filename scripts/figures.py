"""Hình của docs/01_DATASET_ANALYSIS.md và docs/02_DATA_CLEANING.md.

Ghi vào artifacts/dataset/:
  fig_dien_bien.png: năm biến theo ngày 2019 đến 2025, ranh giới ba tập, tháng tảo nở hoa, đợt DO thấp (01 mục 3.5);
    vẽ từ daily_series.csv, monthly_by_year.csv (tháng nở hoa) và low_do_episodes.csv (DO thấp nhất của từng đợt).
  fig_ngay_dem_mua.png: DO theo giờ trong ngày cho từng mùa (diel_profile.csv); biên độ DO trong ngày, nhiệt độ nước,
    chlorophyll theo tháng (monthly_dynamics.csv) (01 mục 3.3).
  fig_su_kien.png: bảy ngày quanh ba cực trị của cả chuỗi 2019 đến 2025, theo mốc 15 phút: DO cao nhất, DO thấp nhất,
    độ đục cao nhất (01 mục 3.5).
  event_windows.csv: thống kê của bảy ngày trong từng khung của fig_su_kien.png: thời điểm, giá trị cực trị, khoảng
    giá trị của các biến, phần trăm bão hòa oxy, biên độ DO trong ngày (trung vị trên các ngày đủ số liệu).
  fig_ffill.png: điền khoảng trống ở lần cả đầu đo ngừng ghi dài nhất của test và của cả chuỗi (gap_stats.csv),
    vẽ từ base_splits/ (02 mục 3.2).
  fig_dot_bien.png: điểm DO đột biến một mốc sáng 2025-08-14, ngày có DO thấp nhất của test (02 mục 5).
Đọc raw/, base_splits/ và các bảng của describe_dataset.py, cleaning_study.py; chạy sau các script đó.
Chạy từ gốc repo: uv run python scripts/figures.py
"""
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd

from common import (CONFIGS, FEATURES as F, NAMES, SPLITS, STATS as OUT, UNITS, VAR_NAMES, do_saturation, load_base_all,
                    load_raw_split, window_starts)
from describe_dataset import season
from plot_style import BLUE, INK, INK2, ORANGE, SEASON_COLOR, SEASON_NAME, VERMILION, panel, save

LOW = 6.0       # ngưỡng DO thấp (mg/L), docs/01_DATASET_ANALYSIS.md mục 3.4
BLOOM = 20.0    # tháng tảo nở hoa: chlorophyll trung bình tháng từ 20 ug/L


def label(c):
    """Nhãn trục của một biến: tên chuẩn kèm đơn vị."""
    return f"{VAR_NAMES[c]} ({UNITS[c]})" if UNITS[c] else VAR_NAMES[c]


def raw_all():
    """Ba tập chưa làm sạch nối tiếp, chỉ số datetime_pst, thêm cột split."""
    return pd.concat([load_raw_split(n).assign(split=n) for n in NAMES]).set_index("datetime_pst")


def periods(months):
    """Gộp các tháng liền nhau (chuỗi YYYY-MM) thành các đoạn (thời điểm đầu, thời điểm cuối)."""
    out = []
    for m in sorted(pd.Period(x, "M") for x in months):
        if out and m == out[-1][1] + 1:
            out[-1][1] = m
        else:
            out.append([m, m])
    return [(a.start_time, b.end_time) for a, b in out]


def fig_dien_bien():
    """Năm khung chung trục thời gian: trung bình ngày của DO, nhiệt độ nước, chlorophyll, độ dẫn điện, độ đục."""
    mb = pd.read_csv(os.path.join(OUT, "monthly_by_year.csv"))
    bloom = periods(mb.loc[mb.chl_mean >= BLOOM, "thang"])
    ds = pd.read_csv(os.path.join(OUT, "daily_series.csv"), parse_dates=["ngay"]).set_index("ngay")
    day = ds[[f"{c}_tb" for c in F]].set_axis(F, axis=1)
    ep = pd.read_csv(os.path.join(OUT, "low_do_episodes.csv"), parse_dates=["thoi_diem_thap_nhat"])
    ep = ep[ep["nguong"] == LOW]
    rows = ["do_mg_l", "water_temp_c", "chl_fluor_ug_l", "spc_us_cm", "turbidity_fnu"]
    fig, axes = plt.subplots(len(rows), 1, figsize=(7, 8), sharex=True)
    for i, (ax, c) in enumerate(zip(axes, rows)):
        for a, b in bloom:
            ax.axvspan(a, b, color=ORANGE, alpha=0.25, linewidth=0)
        for n in NAMES[1:]:
            ax.axvline(pd.Timestamp(SPLITS[n][0]), color=INK, lw=0.7, ls=":")
        ax.plot(day.index, day[c], color=BLUE, lw=0.8)
        ax.set_ylabel(label(c))
        ax.margins(x=0.005)
        panel(ax, "abcde"[i])
    ax = axes[0]
    ax.fill_between(day.index, ds["do_mg_l_min"], ds["do_mg_l_max"], color=BLUE, alpha=0.2, linewidth=0)
    ax.axhline(LOW, color=INK, lw=0.7, ls="--")
    ax.plot(ep["thoi_diem_thap_nhat"], ep["do_thap_nhat"], "v", color=VERMILION, ms=4.5, mec="white", mew=0.5)
    ax.set_ylim(2, 22)
    trans = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    for n in NAMES:
        a, b = (pd.Timestamp(x) for x in SPLITS[n])
        ax.text(a + (b - a) / 2, 1.04, n, transform=trans, ha="center", va="bottom", color=INK)
    handles = [plt.Line2D([], [], color=BLUE, lw=0.8), plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.2),
               plt.Line2D([], [], ls="", marker="v", color=VERMILION, ms=4.5),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE, alpha=0.25)]
    fig.legend(handles, ["Trung bình ngày", "Thấp nhất đến cao nhất trong ngày", "Đợt DO dưới 6 mg/L",
                         "Tháng tảo nở hoa"], loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    save(fig, os.path.join(OUT, "fig_dien_bien.png"))


def bars_by_month(ax, values, ylabel):
    """Cột theo tháng 1 đến 12, tô màu theo giai đoạn mùa."""
    months = np.arange(1, 13)
    ax.bar(months, values, width=0.7, color=[SEASON_COLOR[season(m)] for m in months])
    ax.set_xticks(months)
    ax.set_xlabel("Tháng")
    ax.set_ylabel(ylabel)


def fig_ngay_dem_mua():
    """Bốn khung: DO trung bình theo giờ cho từng mùa; biên độ DO trong ngày, nhiệt độ nước, chlorophyll theo tháng."""
    md = pd.read_csv(os.path.join(OUT, "monthly_dynamics.csv"))
    prof = pd.read_csv(os.path.join(OUT, "diel_profile.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    ax = axes[0, 0]
    for k in ("lanh", "xuan", "nong", "thu"):
        p = prof[prof["mua"] == k]
        ax.plot(p["gio"], p["do_tb"], color=SEASON_COLOR[k], lw=1.6)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlim(0, 24)
    ax.set_xlabel("Giờ trong ngày")
    ax.set_ylabel("DO trung bình (mg/L)")
    bars_by_month(axes[0, 1], md["diel_range_median"], "Biên độ DO trong ngày (mg/L)")
    bars_by_month(axes[1, 0], md["temp_mean"], label("water_temp_c"))
    bars_by_month(axes[1, 1], md["chl_mean"], label("chl_fluor_ug_l"))
    for ax, k in zip(axes.ravel(), "abcd"):
        panel(ax, k)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SEASON_COLOR[k]) for k in ("lanh", "xuan", "nong", "thu")]
    fig.legend(handles, [SEASON_NAME[k] for k in ("lanh", "xuan", "nong", "thu")], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save(fig, os.path.join(OUT, "fig_ngay_dem_mua.png"))


def event_window(r, t):
    """Bảy ngày quanh thời điểm t: từ 4 ngày trước đến 3 ngày sau."""
    return r.loc[t - pd.Timedelta("4D"):t + pd.Timedelta("3D")]


def event_table(r, sat, events):
    """Ghi event_windows.csv: thống kê bảy ngày quanh từng cực trị của fig_su_kien.png."""
    keys = {"DO cao nhất": "do_cao_nhat", "DO thấp nhất": "do_thap_nhat", "Độ đục cao nhất": "do_duc_cao_nhat"}
    rows = []
    for name, t in events:
        x = event_window(r, t)
        s = 100 * x["do_mg_l"] / sat.loc[x.index]
        day = x["do_mg_l"].resample("D")
        dr = (day.max() - day.min())[day.count() >= 86]
        rows.append(dict(su_kien=keys[name], thoi_diem=str(t), do_tai_su_kien=x.loc[t, "do_mg_l"],
                         bao_hoa_tai_su_kien=s.loc[t], do_min=x["do_mg_l"].min(), do_max=x["do_mg_l"].max(),
                         bien_do_do_trung_vi=dr.median(), bao_hoa_min=s.min(), bao_hoa_max=s.max(),
                         ph_min=x["ph"].min(), ph_max=x["ph"].max(), chl_min=x["chl_fluor_ug_l"].min(),
                         chl_max=x["chl_fluor_ug_l"].max(), duc_min=x["turbidity_fnu"].min(),
                         duc_max=x["turbidity_fnu"].max(), nhiet_do_max=x["water_temp_c"].max()))
    pd.DataFrame(rows).round(6).to_csv(os.path.join(OUT, "event_windows.csv"), index=False)


def fig_su_kien(r):
    """Ba cột cực trị của cả chuỗi (7 ngày quanh mỗi cực trị, theo mốc 15 phút), bốn hàng: DO, pH, chlorophyll, độ đục."""
    sat = do_saturation(r["water_temp_c"], r["spc_us_cm"].ffill())
    events = [("DO cao nhất", r["do_mg_l"].idxmax()),
              ("DO thấp nhất", r["do_mg_l"].idxmin()),
              ("Độ đục cao nhất", r["turbidity_fnu"].idxmax())]
    event_table(r, sat, events)
    rows = ["do_mg_l", "ph", "chl_fluor_ug_l", "turbidity_fnu"]
    fig, axes = plt.subplots(len(rows), 3, figsize=(7, 6.4), sharex="col")
    for j, (name, t) in enumerate(events):
        x = event_window(r, t)
        a, b = x.index[0], x.index[-1]
        for i, c in enumerate(rows):
            ax = axes[i, j]
            ax.plot(x.index, x[c], color=BLUE, lw=0.8)
            if c == "do_mg_l":
                ax.plot(x.index, sat.loc[a:b], color=INK, lw=0.9, ls="--")
            ax.axvline(t, color=VERMILION, lw=0.8)
            if j == 0:
                ax.set_ylabel(label(c))
        panel(axes[0, j], "abc"[j], name)
        axes[-1, j].xaxis.set_major_locator(mdates.DayLocator(interval=2))
        axes[-1, j].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    fig.legend([plt.Line2D([], [], color=BLUE, lw=0.8), plt.Line2D([], [], color=INK, lw=0.9, ls="--"),
                plt.Line2D([], [], color=VERMILION, lw=0.8)], ["Số đo", "DO bão hòa", "Thời điểm sự kiện"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save(fig, os.path.join(OUT, "fig_su_kien.png"))


def fig_ffill():
    """Hai khung: DO sau khi điền ở lần cả đầu đo ngừng ghi dài nhất của test (a) và của cả chuỗi (b)."""
    gs = pd.read_csv(os.path.join(OUT, "gap_stats.csv")).set_index("split")
    assert gs.loc["test", "longest_all_na_start"] == "2025-04-08 10:30:00", "khung (a) phải là lần ngừng ghi dài nhất của test"
    assert gs.loc["all", "longest_all_na_start"] == "2020-06-12 13:45:00", "khung (b) phải là lần ngừng ghi dài nhất"
    d = load_base_all().set_index("datetime_pst")
    n_in, n_out = CONFIGS["24h_6h"]
    kept = set(window_starts(d.reset_index(), n_in, n_out)["train"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 2.8), gridspec_kw={"width_ratios": [1, 1.8]})
    for ax, (a, b), ms in [(a1, ("2025-04-08 09:00", "2025-04-08 12:30"), 4), (a2, ("2020-06-11 12:00", "2020-06-16 00:00"), 1.5)]:
        s = d.loc[a:b]
        fill = s["flag_do_mg_l"] == 1
        ax.plot(s.index, s["do_mg_l"], color=INK2, lw=0.6, zorder=1)
        ax.plot(s.index[~fill], s["do_mg_l"][~fill], "o", color=BLUE, ms=ms, mew=0)
        ax.plot(s.index[fill], s["do_mg_l"][fill], "o", color=VERMILION, ms=ms, mew=0)
        ax.set_ylabel("DO (mg/L)")
    a1.xaxis.set_major_locator(mdates.HourLocator())
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    idx = np.flatnonzero((d.index >= "2020-06-11 12:00") & (d.index <= "2020-06-16 00:00"))
    drop = [i for i in idx if i >= n_in and i not in kept]
    a2.axvspan(d.index[drop[0]], d.index[drop[-1]], color=INK2, alpha=0.15, linewidth=0)
    a2.xaxis.set_major_locator(mdates.DayLocator())
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    panel(a1, "a", "2025-04-08")
    panel(a2, "b", "2020-06-11 đến 2020-06-15")
    fig.legend([plt.Line2D([], [], ls="", marker="o", color=BLUE, ms=4),
                plt.Line2D([], [], ls="", marker="o", color=VERMILION, ms=4),
                plt.Rectangle((0, 0), 1, 1, color=INK2, alpha=0.15)],
               ["Số đo thật (cờ 0)", "Giá trị điền (cờ 1)", "Thời điểm dự báo bị bỏ"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save(fig, os.path.join(OUT, "fig_ffill.png"))


def fig_dot_bien(r):
    """Hai khung chung trục thời gian: DO và nhiệt độ nước sáng 2025-08-14, khoanh các điểm đột biến một mốc."""
    test = r[r["split"] == "test"]
    assert str(test["do_mg_l"].idxmin().date()) == "2025-08-14", "ngày vẽ phải là ngày có DO thấp nhất của test"
    rev = pd.read_csv(os.path.join(OUT, "do_reversal_points.csv"), parse_dates=["time"])
    s = r.loc["2025-08-14 08:00":"2025-08-14 13:00"]
    p = rev[(rev["time"] >= s.index[0]) & (rev["time"] <= s.index[-1])]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 3.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(s.index, s["do_mg_l"], color=BLUE, lw=1.1, marker="o", ms=2.5)
    a1.plot(p["time"], p["value"], "o", ms=8, mfc="none", mec=VERMILION, mew=1.2, label="Điểm đột biến một mốc")
    a1.axhline(LOW, color=INK, lw=0.7, ls="--")
    a1.set_ylabel("DO (mg/L)")
    a1.legend(loc="upper left")
    a2.plot(s.index, s["water_temp_c"], color=INK, lw=1.1)
    a2.set_ylabel(label("water_temp_c"))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout()
    save(fig, os.path.join(OUT, "fig_dot_bien.png"))


def main():
    """Vẽ và ghi năm hình."""
    r = raw_all()
    fig_dien_bien()
    fig_ngay_dem_mua()
    fig_su_kien(r)
    fig_ffill()
    fig_dot_bien(r)
    print("written to", OUT)


if __name__ == "__main__":
    main()
