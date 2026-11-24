"""Hình minh họa cho docs/01_DATASET_ANALYSIS.md và docs/02_DATA_CLEANING.md.

Ghi vào artifacts/dataset/:
  fig_toan_chuoi.png: DO trung bình ngày 2019 đến 2025 theo ba tập, đợt DO dưới 6 mg/L, lần đầu đo ngừng ghi (01_DATASET_ANALYSIS 4).
  fig_ngay_dem_mua.png: nhịp ngày đêm theo mùa và các đại lượng theo tháng (01_DATASET_ANALYSIS 3.3, 3.4).
  fig_o_thieu.png: tỉ lệ ô trống theo biến và tập, vị trí các khoảng trống từ 2 giờ (02_DATA_CLEANING 4.1).
  fig_ffill.png: ffill trên một khoảng trống ngắn và một lần ngừng ghi 64.25 giờ, cửa sổ bị bỏ (02_DATA_CLEANING 4.2).
  fig_dot_bien.png: điểm DO đột biến một mốc ngày 2025-08-14 (02_DATA_CLEANING 6).
  fig_dien_bien.png: năm biến theo ngày 2019 đến 2025 cùng các sự kiện (01_DATASET_ANALYSIS 3.5).
  fig_su_kien.png: ba sự kiện điển hình theo mốc 15 phút: tảo nở hoa, DO thấp nhất, độ đục cao nhất (01_DATASET_ANALYSIS 3.5).
Đọc raw/, base_splits/ và các bảng của describe_dataset.py, cleaning_study.py, gap_study.py; chạy sau các script đó.
Chạy từ gốc repo: uv run python scripts/figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import (CONFIGS, FEATURES as F, NAMES, SPLITS, STATS as OUT, do_saturation, load_base_all,  # noqa: E402
                    load_raw_split, window_starts)
from describe_dataset import episode_spans, season  # noqa: E402

# màu: mực chữ, mực phụ, lưới, nền; màu phân loại theo thứ tự cố định của bảng màu tham chiếu
INK, INK2, GRID, SURF, AXIS = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb", "#c9c8c3"
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
SPLIT_COLOR = {"train": BLUE, "val": ORANGE, "test": AQUA}
SEASONS = {"lanh": ("Lạnh (11 đến 3)", BLUE), "nong": ("Nóng (6 đến 9)", ORANGE),
           "xuan": ("Xuân (4, 5)", AQUA), "thu": ("Thu (10)", YELLOW)}
LABEL = {"do_mg_l": "DO", "water_temp_c": "Nhiệt độ", "ph": "pH", "spc_us_cm": "Độ dẫn",
         "turbidity_fnu": "Độ đục", "chl_fluor_ug_l": "Chlorophyll"}
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 10, "axes.titlelocation": "left", "axes.titlepad": 8,
                     "axes.titlecolor": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8.5})


def raw_all():
    """Ba tập chưa làm sạch nối tiếp, chỉ số datetime_pst, thêm cột split."""
    return pd.concat([load_raw_split(n).assign(split=n) for n in NAMES]).set_index("datetime_pst")


def save(fig, name):
    """Lưu hình PNG vào artifacts/dataset/."""
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)


def fig_toan_chuoi(r):
    """DO trung bình ngày (đường) và khoảng nhỏ nhất đến lớn nhất trong ngày (dải), tô màu theo tập."""
    fig, ax = plt.subplots(figsize=(12, 4))
    day = r["do_mg_l"].resample("D")
    mean, lo, hi = day.mean(), day.min(), day.max()
    for n in NAMES:
        a, b = (pd.Timestamp(x) for x in SPLITS[n])
        sel = (mean.index >= a) & (mean.index < b)
        c = SPLIT_COLOR[n]
        ax.fill_between(mean.index[sel], lo[sel], hi[sel], color=c, alpha=0.18, linewidth=0)
        ax.plot(mean.index[sel], mean[sel], color=c, linewidth=1.2, label=f"{n}: {a:%Y-%m} đến {b - pd.Timedelta('1D'):%Y-%m}")
        ax.text(a + (b - a) / 2, 19.3, n, color=INK, ha="center", fontsize=9.5)
    ax.axhline(6, color=INK2, linewidth=0.9, linestyle="--")
    ax.text(mean.index[0], 6.15, "6 mg/L", color=INK2, fontsize=8)
    do = r["do_mg_l"].to_numpy()
    for a, l in episode_spans(np.nan_to_num(do, nan=99) < 6):
        ax.plot(r.index[a + int(np.nanargmin(do[a:a + l]))], np.nanmin(do[a:a + l]), "v", color=RED, ms=6,
                mec=SURF, mew=1)
    ax.plot([], [], "v", color=RED, ms=6, label="Đợt DO dưới 6 mg/L (giá trị thấp nhất)")
    g = pd.read_csv(os.path.join(OUT, "gap_events.csv"), parse_dates=["bat_dau"])
    g = g[(g["so_bien_cung_trong"] == 6) & (g["gio"] >= 6)].drop_duplicates("bat_dau")
    for t, h in zip(g["bat_dau"], g["gio"]):
        ax.axvspan(t, t + pd.Timedelta(hours=h), ymin=0, ymax=0.06, color=INK2)
    ax.plot([], [], "s", color=INK2, ms=6, label="Lần đầu đo ngừng ghi từ 6 giờ")
    ax.set_ylim(2, 20)
    ax.set_ylabel("DO (mg/L)")
    ax.set_title("DO theo ngày, 2019-01 đến 2025-11: trung bình ngày (đường), thấp nhất đến cao nhất trong ngày (dải)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0.005)
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, -0.08))
    fig.tight_layout()
    save(fig, "fig_toan_chuoi.png")


def bars_by_month(ax, values, title, unit):
    """Cột theo tháng 1 đến 12, tô màu theo mùa, ghi giá trị trên cột."""
    months = np.arange(1, 13)
    ax.bar(months, values, width=0.7, color=[SEASONS[season(m)][1] for m in months], edgecolor=SURF, linewidth=1.5)
    for m, v in zip(months, values):
        ax.text(m, v, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(months)
    ax.set_xlabel("Tháng")
    ax.set_ylabel(unit)
    ax.set_ylim(0, max(values) * 1.15)
    ax.set_title(title)


def fig_ngay_dem_mua(r):
    """Sáu khung: DO theo giờ trong ngày cho từng mùa, DO theo tháng, và bốn đại lượng theo tháng (monthly_dynamics)."""
    md = pd.read_csv(os.path.join(OUT, "monthly_dynamics.csv"))
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8))
    ax = axes[0, 0]
    do = r["do_mg_l"]
    hour = do.index.hour + do.index.minute / 60
    reg = pd.Series(do.index.month.map(season), index=do.index)
    for k, (name, c) in SEASONS.items():
        p = do[reg == k].groupby(hour[reg == k]).mean()
        ax.plot(p.index, p.values, color=c, linewidth=2, label=name)
        ax.text(23.9, p.values[-1], name.split(" ")[0], color=INK, fontsize=7.5, va="center")
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlim(0, 27)
    ax.set_xlabel("Giờ trong ngày")
    ax.set_ylabel("mg/L")
    ax.set_title("DO trung bình theo giờ, theo mùa")
    ax = axes[0, 1]
    g = do.groupby(do.index.month)
    q05, q50, q95 = g.quantile(0.05), g.mean(), g.quantile(0.95)
    ax.fill_between(q50.index, q05, q95, color=BLUE, alpha=0.18, linewidth=0, label="5% đến 95% số đo")
    ax.plot(q50.index, q50, color=BLUE, linewidth=2, marker="o", ms=4, label="Trung bình")
    ax.axhline(6, color=INK2, linewidth=0.9, linestyle="--")
    ax.text(1, 6.15, "6 mg/L", color=INK2, fontsize=7.5)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Tháng")
    ax.set_ylabel("mg/L")
    ax.set_title("DO theo tháng")
    ax.legend(loc="upper left")
    bars_by_month(axes[0, 2], md["temp_mean"], "Nhiệt độ nước trung bình", "độ C")
    bars_by_month(axes[1, 0], md["diel_range_median"], "Chênh lệch DO trong ngày (trung vị)", "mg/L")
    bars_by_month(axes[1, 1], md["do_change_6h_rms"], "DO thay đổi sau 6 giờ (căn trung bình bình phương)", "mg/L")
    bars_by_month(axes[1, 2], md["chl_mean"], "Chlorophyll trung bình", "ug/L")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in SEASONS.values()]
    fig.legend(handles, [n for n, _ in SEASONS.values()], loc="upper right", ncol=4, bbox_to_anchor=(0.995, 0.995))
    fig.suptitle("Nhịp ngày đêm và theo mùa của DO, 2019 đến 2025", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig_ngay_dem_mua.png")


def fig_o_thieu():
    """Hai khung: phần trăm ô trống theo biến và tập (feature_stats); khoảng trống từ 2 giờ theo thời gian (gap_events)."""
    fs = pd.read_csv(os.path.join(OUT, "feature_stats.csv"))
    g = pd.read_csv(os.path.join(OUT, "gap_events.csv"), parse_dates=["bat_dau"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
    x = np.arange(len(F))
    w = 0.26
    for i, n in enumerate(NAMES):
        v = fs.set_index(["split", "feature"]).loc[n].loc[F, "missing_pct"].to_numpy()
        a1.bar(x + (i - 1) * w, v, width=w, color=SPLIT_COLOR[n], edgecolor=SURF, linewidth=1.5, label=n)
        for xi, vi in zip(x + (i - 1) * w, v):
            a1.text(xi, vi, f"{vi:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK2, rotation=90)
    a1.set_xticks(x, [LABEL[c] for c in F])
    a1.set_ylim(0, fs.loc[fs["split"] != "all", "missing_pct"].max() * 1.3)
    a1.set_ylabel("% số ô")
    a1.set_title("Tỉ lệ ô trống theo biến và tập")
    a1.legend(loc="upper left", ncol=3)
    rows = {c: len(F) - 1 - j for j, c in enumerate(F)}
    for whole, c, lab in [(True, ORANGE, "Cả sáu biến cùng trống"), (False, BLUE, "Chỉ một vài biến trống")]:
        e = g[(g["so_bien_cung_trong"] == 6) == whole]
        a2.scatter(e["bat_dau"], e["bien"].map(rows), s=12 + e["gio"] * 2.5, color=c, alpha=0.75,
                   edgecolor=SURF, linewidth=0.8, label=lab)
    for n in NAMES[1:]:
        a2.axvline(pd.Timestamp(SPLITS[n][0]), color=INK2, linewidth=0.9, linestyle="--")
    for n in NAMES:
        a, b = (pd.Timestamp(t) for t in SPLITS[n])
        a2.text(a + (b - a) / 2, len(F) - 0.35, n, ha="center", color=INK, fontsize=9)
    a2.set_yticks(list(rows.values()), [LABEL[c] for c in rows])
    a2.set_ylim(-0.6, len(F) - 0.1)
    a2.grid(axis="y", visible=False)
    a2.grid(axis="x", visible=True)
    a2.xaxis.set_major_locator(mdates.YearLocator())
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    a2.set_title("Các khoảng trống từ 2 giờ trở lên (diện tích chấm theo độ dài)")
    a2.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig_o_thieu.png")


def fig_ffill(r):
    """Hai khung: DO sau ffill quanh một khoảng trống ngắn (test) và lần ngừng ghi 64.25 giờ (train)."""
    d = load_base_all().set_index("datetime_pst")
    n_in, n_out = CONFIGS["24h_6h"]
    kept = set(window_starts(d.reset_index(), n_in, n_out)["train"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 1.8]})
    for ax, (a, b) in [(a1, ("2025-04-08 09:00", "2025-04-08 12:30")), (a2, ("2020-06-11 12:00", "2020-06-16 00:00"))]:
        s = d.loc[a:b]
        fill = s["flag_do_mg_l"] == 1
        ax.plot(s.index, s["do_mg_l"], color=INK2, linewidth=1, alpha=0.5, zorder=1)
        ms = 5 if ax is a1 else 2.5
        ax.plot(s.index[~fill], s["do_mg_l"][~fill], "o", color=BLUE, ms=ms, mec=SURF, mew=0.5, label="Đo thật (cờ 0)")
        ax.plot(s.index[fill], s["do_mg_l"][fill], "o", color=ORANGE, ms=ms, mec=SURF, mew=0.5, label="Điền (cờ 1)")
        ax.set_ylabel("DO (mg/L)")
    a1.set_title("Khoảng trống 30 phút, 2025-04-08 (test)")
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    a1.legend(loc="upper left")
    idx = np.flatnonzero((d.index >= "2020-06-11 12:00") & (d.index <= "2020-06-16 00:00"))
    drop = [i for i in idx if i >= n_in and i not in kept]
    t0, t1 = d.index[drop[0]], d.index[drop[-1]]
    a2.axvspan(t0, t1, color=INK2, alpha=0.12, linewidth=0)
    a2.text(t0 + (t1 - t0) / 2, a2.get_ylim()[1], f"{len(drop)} thời điểm dự báo bị bỏ:\nnhãn 6 giờ toàn DO điền",
            ha="center", va="top", fontsize=8, color=INK)
    a2.set_title("Lần ngừng ghi 64.25 giờ, 2020-06-12 13:45 đến 2020-06-15 05:45 (train)")
    a2.xaxis.set_major_locator(mdates.DayLocator())
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    a2.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "fig_ffill.png")


def fig_dot_bien(r):
    """Hai khung chung trục thời gian: DO và nhiệt độ sáng 2025-08-14, đánh dấu các điểm đột biến một mốc."""
    rev = pd.read_csv(os.path.join(OUT, "do_reversal_points.csv"), parse_dates=["time"])
    s = r.loc["2025-08-14 08:00":"2025-08-14 13:00"]
    p = rev[(rev["time"] >= s.index[0]) & (rev["time"] <= s.index[-1])]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(s.index, s["do_mg_l"], color=BLUE, linewidth=1.5, marker="o", ms=3.5, mec=SURF, mew=0.5)
    a1.plot(p["time"], p["value"], "o", ms=10, mfc="none", mec=RED, mew=1.8, label="Điểm đột biến một mốc")
    a1.axhline(6, color=INK2, linewidth=0.9, linestyle="--")
    low = p.loc[p["value"].idxmin()]
    a1.annotate(f"{low['time']:%H:%M}: {low['value']} mg/L", (low["time"], low["value"]), xytext=(12, 0),
                textcoords="offset points", va="center", fontsize=8, color=INK)
    a1.set_ylabel("DO (mg/L)")
    a1.set_title("DO sáng 2025-08-14 (test): lệch quá 0.8 mg/L trong một mốc rồi quay lại")
    a1.legend(loc="upper left")
    a2.plot(s.index, s["water_temp_c"], color=INK2, linewidth=1.5)
    a2.set_ylabel("Nhiệt độ (độ C)")
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout()
    save(fig, "fig_dot_bien.png")


def periods(months):
    """Gộp các tháng liền nhau (chuỗi YYYY-MM) thành các đoạn (ngày đầu, ngày cuối)."""
    out = []
    for m in sorted(pd.Period(x, "M") for x in months):
        if out and m == out[-1][1] + 1:
            out[-1][1] = m
        else:
            out.append([m, m])
    return [(a.start_time, b.end_time) for a, b in out]


def fig_dien_bien(r):
    """Năm biến theo ngày 2019 đến 2025, tô các giai đoạn tảo nở hoa, đánh dấu đợt DO thấp, lũ đục, độ dẫn thấp."""
    mb = pd.read_csv(os.path.join(OUT, "monthly_by_year.csv"))
    bloom = periods(mb.loc[mb.chl_mean >= 20, "thang"])
    lowspc = periods(mb.loc[mb.spc_mean < 120, "thang"])
    day = r[F].resample("D").mean()
    fig, axes = plt.subplots(5, 1, figsize=(12, 10.5), sharex=True)
    rows = [("do_mg_l", "DO (mg/L)"), ("water_temp_c", "Nhiệt độ (độ C)"), ("chl_fluor_ug_l", "Chlorophyll (ug/L)"),
            ("spc_us_cm", "Độ dẫn (uS/cm)"), ("turbidity_fnu", "Độ đục (FNU)")]
    for ax, (c, lab) in zip(axes, rows):
        for a, b in bloom:
            ax.axvspan(a, b, color=ORANGE, alpha=0.13, linewidth=0)
        for n in NAMES:
            ax.axvline(pd.Timestamp(SPLITS[n][0]), color=INK2, lw=0.8, ls=":")
        ax.plot(day.index, day[c], color=BLUE, lw=1)
        ax.set_ylabel(lab)
        ax.margins(x=0.005)
    ax = axes[0]
    dd = r["do_mg_l"].resample("D")
    ax.fill_between(day.index, dd.min(), dd.max(), color=BLUE, alpha=0.18, linewidth=0)
    ax.axhline(6, color=INK2, lw=0.9, ls="--")
    do = r["do_mg_l"].to_numpy()
    for a, l in episode_spans(np.nan_to_num(do, nan=99) < 6):
        ax.plot(r.index[a + int(np.nanargmin(do[a:a + l]))], np.nanmin(do[a:a + l]), "v", color=RED, ms=6, mec=SURF)
    ax.plot([], [], "v", color=RED, ms=6, label="đợt DO dưới 6 mg/L")
    ax.fill_between([], [], [], color=BLUE, alpha=0.18, label="thấp nhất đến cao nhất trong ngày")
    ax.fill_between([], [], [], color=ORANGE, alpha=0.13, label="tháng tảo nở hoa (chlorophyll TB tháng từ 20 ug/L)")
    ax.legend(loc="upper left", ncol=3)
    ax.set_ylim(2, 22)
    for n in NAMES:
        a, b = (pd.Timestamp(x) for x in SPLITS[n])
        axes[2].text(a + (b - a) / 2, 95, n, ha="center", fontsize=9.5, color=INK)
    for a, b in lowspc:
        axes[3].annotate("độ dẫn thấp kéo dài", (a + (b - a) / 2, 80), xytext=(0, 18), textcoords="offset points",
                         ha="center", fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    top = mb.nlargest(2, "turb_max").sort_values("thang")
    for (k, v), dx in zip(zip(top.thang, top.turb_max), (-8, 8)):
        t = pd.Timestamp(k + "-01")
        y = day.loc[t:t + pd.Timedelta("30D"), "turbidity_fnu"].max()
        axes[4].annotate(f"độ đục cao {k[5:]}/{k[:4]}", (day.loc[t:t + pd.Timedelta("30D"), "turbidity_fnu"].idxmax(), y),
                         xytext=(dx, -2), textcoords="offset points", fontsize=7.5, color=INK2, va="top",
                         ha="right" if dx < 0 else "left")
    axes[0].set_title("Diễn biến 2019 đến 2025 (trung bình ngày); đường chấm là ranh giới train, val, test")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    save(fig, "fig_dien_bien.png")


def fig_su_kien(r):
    """Ba sự kiện điển hình, mỗi sự kiện 7 ngày theo mốc 15 phút: tảo nở hoa (DO cao nhất của val),
    DO thấp nhất chuỗi, độ đục cao nhất chuỗi."""
    sat = do_saturation(r["water_temp_c"], r["spc_us_cm"].ffill())
    events = [("Tảo nở hoa", r.loc["2024-06":"2024-08", "do_mg_l"].idxmax()),
              ("DO thấp nhất chuỗi", r["do_mg_l"].idxmin()),
              ("Độ đục cao nhất chuỗi", r["turbidity_fnu"].idxmax())]
    rows = [("do_mg_l", "DO (mg/L)"), ("water_temp_c", "Nhiệt độ (độ C)"), ("chl_fluor_ug_l", "Chlorophyll (ug/L)"),
            ("ph", "pH"), ("turbidity_fnu", "Độ đục (FNU)")]
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 10), sharex="col")
    for j, (name, t) in enumerate(events):
        a, b = t - pd.Timedelta("4D"), t + pd.Timedelta("3D")
        x = r.loc[a:b]
        for i, (c, lab) in enumerate(rows):
            ax = axes[i, j]
            ax.plot(x.index, x[c], color=BLUE, lw=1)
            if c == "do_mg_l":
                ax.plot(x.index, sat.loc[a:b], color=INK2, lw=1, ls="--", label="DO bão hòa")
                ax.axhline(6, color=RED, lw=0.8, ls=":")
                ax.legend(loc="upper left")
            ax.axvline(t, color=ORANGE, lw=1)
            if j == 0:
                ax.set_ylabel(lab)
        axes[0, j].set_title(f"{name}: {t:%Y-%m-%d %H:%M}")
        axes[-1, j].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        axes[-1, j].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    fig.tight_layout()
    save(fig, "fig_su_kien.png")


def main():
    """Vẽ và ghi bảy hình."""
    r = raw_all()
    fig_toan_chuoi(r)
    fig_ngay_dem_mua(r)
    fig_o_thieu()
    fig_ffill(r)
    fig_dot_bien(r)
    fig_dien_bien(r)
    fig_su_kien(r)
    print("written to", OUT)


if __name__ == "__main__":
    main()
