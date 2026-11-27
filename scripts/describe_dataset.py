"""Thống kê mô tả ba tập train, val, test trước làm sạch, dùng cho docs/01_DATASET_ANALYSIS.md mục 3.

Ghi vào artifacts/dataset/: feature_stats (từng biến), gap_stats (khoảng trống theo dòng), do_dynamics (DO theo tập),
monthly_dynamics, split_by_regime (theo tháng, theo mùa), monthly_by_year (từng tháng của từng năm), low_do_episodes
(từng đợt DO dưới 6 và dưới 5 mg/L), diel_profile (DO trung bình theo giờ trong ngày của từng giai đoạn mùa),
daily_series (trung bình ngày của năm biến, DO thấp nhất và cao nhất trong ngày). Trong gap_stats, do_na_ca_dau_do là
số ô DO trống mà năm biến còn lại cùng trống; longest_all_na_h, longest_all_na_start là độ dài (giờ) và thời điểm bắt
đầu của lần cả đầu đo ngừng ghi dài nhất. Trong do_dynamics, month_var_explained_pct là phần trăm
phương sai DO giải thích bởi trung bình từng tháng, year_mean_min, year_mean_max là trung bình năm nhỏ nhất, lớn nhất; diel_min_hour, diel_max_hour là giờ
có DO trung bình thấp nhất, cao nhất trong ngày. Cột do_change_6h_rms: căn bậc hai của trung bình bình phương
hiệu DO 6 giờ sau trừ DO hiện tại (mg/L), đo mức biến động. Cột temp_mean_train của monthly_dynamics là
nhiệt độ nước trung bình tháng chỉ tính trên train: căn cứ chia giai đoạn mùa. Danh sách khoảng trống dài nằm ở gap_events.csv
(scripts/gap_study.py).
Chạy từ gốc repo: uv run python scripts/describe_dataset.py
"""
import os
import numpy as np
import pandas as pd

from common import FEATURES as F, NAMES, STATS as OUT, do_saturation, load_raw_split, runs

STEP_H = 0.25             # một mốc = 0.25 giờ
SEASONS = ["lanh", "xuan", "nong", "thu"]


def load(name):
    """Tập chưa làm sạch, chỉ số là datetime_pst."""
    d = load_raw_split(name)
    return d.set_index("datetime_pst")


def episode_spans(below, min_h=1.0, merge_h=1.0):
    """Các đợt DO thấp (vị trí bắt đầu, số mốc): gộp hai đoạn dưới ngưỡng cách nhau không quá merge_h giờ, rồi giữ
    đoạn dài ít nhất min_h giờ."""
    b = np.asarray(below, bool).copy()
    st, ln = runs(~b)
    for a, l in zip(st, ln):
        if 0 < a and a + l < len(b) and l * STEP_H <= merge_h:
            b[a:a + l] = True
    st, ln = runs(b)
    keep = ln * STEP_H >= min_h
    return list(zip(st[keep], ln[keep]))


def episodes(below, min_h=1.0, merge_h=1.0):
    """Số đợt DO thấp theo episode_spans."""
    return len(episode_spans(below, min_h, merge_h))


def month_r2(do):
    """Phần trăm phương sai DO giải thích bởi trung bình từng tháng (theo năm và tháng)."""
    v = do.dropna()
    m = v.groupby([v.index.year, v.index.month]).transform("mean")
    return 100 * (1 - ((v - m) ** 2).sum() / ((v - v.mean()) ** 2).sum())


def season(m):
    """Giai đoạn mùa của tháng m (docs/01_DATASET_ANALYSIS.md mục 3.3): lanh, xuan, nong, thu."""
    return {11: "lanh", 12: "lanh", 1: "lanh", 2: "lanh", 3: "lanh", 4: "xuan", 5: "xuan",
            6: "nong", 7: "nong", 8: "nong", 9: "nong", 10: "thu"}[m]


def main():
    """Tính và ghi sáu bảng thống kê."""
    os.makedirs(OUT, exist_ok=True)
    S = {n: load(n) for n in NAMES}
    S["all"] = pd.concat([S["train"], S["val"], S["test"]])
    # 1. thống kê từng biến
    rows = []
    for n, d in S.items():
        for c in F:
            v = d[c].dropna()
            q = v.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1]).to_numpy()
            rows.append(dict(split=n, feature=c, n=len(d), missing=int(d[c].isna().sum()),
                             missing_pct=round(d[c].isna().mean() * 100, 6), mean=v.mean(), sd=v.std(),
                             min=q[0], p01=q[1], p05=q[2], p25=q[3], median=q[4], p75=q[5], p95=q[6], p99=q[7], max=q[8]))
    fs = pd.DataFrame(rows).round(6)
    fs.to_csv(os.path.join(OUT, "feature_stats.csv"), index=False)
    # 2. cấu trúc dữ liệu khuyết
    rows = []
    for n, d in S.items():
        allna = d[F].isna().any(axis=1).to_numpy()
        st, ln = runs(allna)
        whole = d[F].isna().all(axis=1).to_numpy()                  # cả đầu đo ngừng ghi: sáu biến cùng trống
        ws, wl = runs(whole)
        r = dict(split=n, rows=len(d), start=str(d.index[0]), end=str(d.index[-1]), days=round(len(d) / 96, 1),
                 rows_any_na=int(allna.sum()), pct_any_na=round(allna.mean() * 100, 6), na_runs=len(ln),
                 na_runs_le_1h=int((ln <= 4).sum()),
                 longest_na_run_h=float(ln.max() * STEP_H) if len(ln) else 0.0,
                 do_na_ca_dau_do=int((d["do_mg_l"].isna() & d[F[1:]].isna().all(axis=1)).sum()),
                 longest_all_na_h=float(wl.max() * STEP_H) if len(wl) else 0.0,
                 longest_all_na_start=str(d.index[ws[int(np.argmax(wl))]]) if len(wl) else "")
        for c in F:
            s_, l_ = runs(d[c].isna().to_numpy())
            r[f"maxgap_h_{c}"] = float(l_.max() * STEP_H) if len(l_) else 0.0
        rows.append(r)
    gs = pd.DataFrame(rows)
    gs.to_csv(os.path.join(OUT, "gap_stats.csv"), index=False)
    # 3. DO: dao động ngày, độ bão hòa, đợt DO thấp, tương quan với các biến khác
    full = S["all"]
    rows = []
    for n, d in S.items():
        do = d["do_mg_l"]
        day = do.resample("D")
        dr = (day.max() - day.min())[day.count() >= 86]
        yr = do.groupby(do.index.year).mean()
        prof = do.groupby(do.index.hour + do.index.minute / 60).mean()   # DO trung bình theo giờ trong ngày
        sat = do / do_saturation(d["water_temp_c"], d["spc_us_cm"].fillna(0)) * 100
        r = dict(split=n, DO_mean=do.mean(), DO_sd=do.std(), diel_range_median=dr.median(), diel_range_p90=dr.quantile(.9),
                 sat_pct_median=sat.median(), sat_pct_p01=sat.quantile(.01), sat_pct_p99=sat.quantile(.99),
                 unchanged_adjacent_pct=(do.diff() == 0).mean() * 100,
                 episodes_lt6=episodes((do < 6).fillna(False)), episodes_lt5=episodes((do < 5).fillna(False)),
                 pct_time_lt6=(do < 6).mean() * 100, pct_time_lt5=(do < 5).mean() * 100,
                 corr_temp=do.corr(d["water_temp_c"]), corr_ph=do.corr(d["ph"]), corr_spc=do.corr(d["spc_us_cm"]),
                 corr_turb=do.corr(d["turbidity_fnu"]), corr_chl=do.corr(d["chl_fluor_ug_l"]),
                 month_var_explained_pct=month_r2(do), year_mean_min=yr.min(), year_mean_max=yr.max(),
                 diel_min_hour=prof.idxmin(), diel_max_hour=prof.idxmax())
        rows.append(r)
    dyn = pd.DataFrame(rows).round(6)
    dyn.to_csv(os.path.join(OUT, "do_dynamics.csv"), index=False)
    # 4. theo tháng và theo giai đoạn mùa; mức biến động là căn trung bình bình phương của DO(t + 6 giờ) - DO(t)
    rows = []
    do = full["do_mg_l"]
    day = do.resample("D")
    dr = (day.max() - day.min())[day.count() >= 86]
    e6 = (do.shift(-24) - do)
    tr = S["train"]
    months = sorted(set(full.index.strftime("%Y-%m")))       # các tháng có dữ liệu, dạng YYYY-MM
    for m in range(1, 13):
        sel = full.index.month == m
        ym = [k for k in months if int(k[5:]) == m]
        ep6 = sum(episodes((full.loc[k, "do_mg_l"] < 6).fillna(False)) for k in ym)
        ep5 = sum(episodes((full.loc[k, "do_mg_l"] < 5).fillna(False)) for k in ym)
        rows.append(dict(month=m, regime=season(m), DO_mean=do[sel].mean(), DO_sd=do[sel].std(), DO_p05=do[sel].quantile(.05),
                         diel_range_median=dr[dr.index.month == m].median(), temp_mean=full.loc[sel, "water_temp_c"].mean(),
                         chl_mean=full.loc[sel, "chl_fluor_ug_l"].mean(), spc_mean=full.loc[sel, "spc_us_cm"].mean(),
                         turb_mean=full.loc[sel, "turbidity_fnu"].mean(),
                         do_change_6h_rms=np.sqrt((e6[sel] ** 2).mean()), episodes_lt6=ep6, episodes_lt5=ep5,
                         temp_mean_train=tr.loc[tr.index.month == m, "water_temp_c"].mean()))
    pd.DataFrame(rows).round(6).to_csv(os.path.join(OUT, "monthly_dynamics.csv"), index=False)
    rows = []
    for n in NAMES:
        d = S[n]
        for reg in ["lanh", "xuan", "nong", "thu"]:
            sel = d.index.month.map(season) == reg
            if not sel.any():
                continue
            do_ = d.loc[sel, "do_mg_l"]
            e = (d["do_mg_l"].shift(-24) - d["do_mg_l"])[sel]
            rows.append(dict(split=n, regime=reg, months=int(pd.Series(d.index[sel].strftime("%Y-%m")).nunique()),
                             rows=int(sel.sum()), DO_mean=do_.mean(), DO_sd=do_.std(),
                             do_change_6h_rms=np.sqrt((e**2).mean())))
    pd.DataFrame(rows).round(6).to_csv(os.path.join(OUT, "split_by_regime.csv"), index=False)
    # 5. từng tháng của từng năm: diễn biến nhiều năm và các sự kiện (docs/01_DATASET_ANALYSIS.md mục 3.5)
    sat = full["do_mg_l"] / do_saturation(full["water_temp_c"], full["spc_us_cm"].fillna(0)) * 100
    rows = []
    for k in months:
        x = full.loc[k]
        drk = dr[dr.index.strftime("%Y-%m") == k]
        rows.append(dict(thang=k, DO_mean=x["do_mg_l"].mean(), DO_min=x["do_mg_l"].min(), DO_max=x["do_mg_l"].max(),
                         diel_range_median=drk.median(), sat_pct_max=sat.loc[k].max(), temp_mean=x["water_temp_c"].mean(),
                         temp_max=x["water_temp_c"].max(), chl_mean=x["chl_fluor_ug_l"].mean(),
                         chl_max=x["chl_fluor_ug_l"].max(), ph_mean=x["ph"].mean(), spc_mean=x["spc_us_cm"].mean(),
                         turb_mean=x["turbidity_fnu"].mean(), turb_max=x["turbidity_fnu"].max(),
                         episodes_lt6=episodes((x["do_mg_l"] < 6).fillna(False))))
    pd.DataFrame(rows).round(6).to_csv(os.path.join(OUT, "monthly_by_year.csv"), index=False)
    # 6. từng đợt DO thấp (docs/01_DATASET_ANALYSIS.md mục 3.4)
    rows = []
    split_of = pd.Series(np.concatenate([[n] * len(S[n]) for n in NAMES]), index=full.index)
    for thr in (6, 5):
        for a, ln in episode_spans((full["do_mg_l"] < thr).fillna(False).to_numpy()):
            x = full.iloc[a:a + ln]
            rows.append(dict(nguong=thr, tap=split_of.iloc[a], bat_dau=str(x.index[0]), ket_thuc=str(x.index[-1]),
                             so_gio=ln * STEP_H, do_thap_nhat=x["do_mg_l"].min(),
                             thoi_diem_thap_nhat=str(x["do_mg_l"].idxmin()), nhiet_do_min=x["water_temp_c"].min(),
                             nhiet_do_max=x["water_temp_c"].max()))
    pd.DataFrame(rows).round(6).to_csv(os.path.join(OUT, "low_do_episodes.csv"), index=False)
    # 7. DO trung bình theo giờ trong ngày của từng giai đoạn mùa (docs/01 Hình 1a)
    hour = full.index.hour + full.index.minute / 60
    reg = full.index.month.map(season)
    rows = [dict(mua=k, gio=h, do_tb=v) for k in SEASONS
            for h, v in full.loc[reg == k, "do_mg_l"].groupby(hour[reg == k]).mean().items()]
    pd.DataFrame(rows).round(6).to_csv(os.path.join(OUT, "diel_profile.csv"), index=False)
    # 8. chuỗi theo ngày (docs/01 Hình 2)
    day = full[F].resample("D").mean()
    day.columns = [f"{c}_tb" for c in F]
    dd = full["do_mg_l"].resample("D")
    day.insert(1, "do_mg_l_min", dd.min())
    day.insert(2, "do_mg_l_max", dd.max())
    day.index = day.index.strftime("%Y-%m-%d")
    day.round(6).rename_axis("ngay").to_csv(os.path.join(OUT, "daily_series.csv"))
    print("written to", OUT)


if __name__ == "__main__":
    main()
