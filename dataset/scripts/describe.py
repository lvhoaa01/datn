"""Thong ke mo ta dataset tu ba file split (can pandas, numpy).
Ket qua ghi vao thu muc stats/. Chay tu thu muc dataset:  python scripts/describe.py
"""
import json
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "stats")
F = ["do_mg_l", "water_temp_c", "ph", "spc_us_cm", "turbidity_fnu", "chl_fluor_ug_l"]
STEP_H = 0.25


def load(name):
    d = pd.read_csv(os.path.join(ROOT, "splits", f"{name}.csv"), na_values=["NA"], parse_dates=["datetime_pst"])
    return d.set_index("datetime_pst")


def runs(mask):
    m = np.asarray(mask, bool)
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return s, e - s


def fill_short(s, k=4):
    v = s.to_numpy(float).copy()
    st, ln = runs(np.isnan(v))
    for a, l in zip(st, ln):
        if l <= k and a > 0 and a + l < len(v):
            v[a:a + l] = np.linspace(v[a - 1], v[a + l], l + 2)[1:-1]
    return v


def episodes(below, min_h=1.0, merge_h=1.0):
    b = np.asarray(below, bool).copy()
    st, ln = runs(~b)
    for a, l in zip(st, ln):
        if 0 < a and a + l < len(b) and l * STEP_H <= merge_h:
            b[a:a + l] = True
    st, ln = runs(b)
    return int((ln * STEP_H >= min_h).sum())


def do_sat(t, spc):
    T = t + 273.15
    R = np.clip(spc, 0, None) / 53087.0
    sal = 0.0120 - 0.2174 * R**0.5 + 25.3283 * R + 13.7714 * R**1.5 - 6.4788 * R**2 + 2.5842 * R**2.5
    lnc = (-139.34411 + 1.575701e5 / T - 6.642308e7 / T**2 + 1.243800e10 / T**3 - 8.621949e11 / T**4
           - sal * (1.7674e-2 - 10.754 / T + 2140.7 / T**2))
    return np.exp(lnc)


def season(m):
    return {11: "lanh", 12: "lanh", 1: "lanh", 2: "lanh", 3: "lanh", 4: "xuan", 5: "xuan",
            6: "nong", 7: "nong", 8: "nong", 9: "nong", 10: "thu"}[m]


def main():
    os.makedirs(OUT, exist_ok=True)
    S = {n: load(n) for n in ["train", "val", "test"]}
    S["all"] = pd.concat([S["train"], S["val"], S["test"]])
    res = {}
    # 1. thong ke tung bien
    rows = []
    for n, d in S.items():
        for c in F:
            v = d[c].dropna()
            q = v.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1]).to_numpy()
            rows.append(dict(split=n, feature=c, n=len(d), missing=int(d[c].isna().sum()),
                             missing_pct=round(d[c].isna().mean() * 100, 3), mean=v.mean(), sd=v.std(),
                             min=q[0], p01=q[1], p05=q[2], p25=q[3], median=q[4], p75=q[5], p95=q[6], p99=q[7], max=q[8]))
    fs = pd.DataFrame(rows).round(3)
    fs.to_csv(os.path.join(OUT, "feature_stats.csv"), index=False)
    # 2. cau truc khuyet va cua so
    rows = []
    for n, d in S.items():
        allna = d[F].isna().any(axis=1).to_numpy()
        st, ln = runs(allna)
        filled = np.column_stack([fill_short(d[c]) for c in F])
        ok = ~np.isnan(filled).any(axis=1)
        st2, ln2 = runs(ok)
        r = dict(split=n, rows=len(d), start=str(d.index[0]), end=str(d.index[-1]), days=round(len(d) / 96, 1),
                 rows_any_na=int(allna.sum()), pct_any_na=round(allna.mean() * 100, 3), na_runs=len(ln),
                 na_runs_le_1h=int((ln <= 4).sum()), na_runs_gt_6h=int((ln > 24).sum()),
                 longest_na_run_h=float(ln.max() * STEP_H) if len(ln) else 0.0,
                 pct_complete_after_fill_1h=round(ok.mean() * 100, 3),
                 longest_complete_segment_days=round(ln2.max() * STEP_H / 24, 1))
        for c in F:
            s_, l_ = runs(d[c].isna().to_numpy())
            r[f"maxgap_h_{c}"] = float(l_.max() * STEP_H) if len(l_) else 0.0
        rows.append(r)
    gs = pd.DataFrame(rows)
    gs.to_csv(os.path.join(OUT, "gap_stats.csv"), index=False)
    # NA runs > 6h, liet ke
    long = []
    for n in ["train", "val", "test"]:
        d = S[n]
        for c in F:
            s_, l_ = runs(d[c].isna().to_numpy())
            for a, l in zip(s_, l_):
                if l * STEP_H > 6:
                    long.append(dict(split=n, feature=c, start=str(d.index[a]), hours=l * STEP_H))
    pd.DataFrame(long).to_csv(os.path.join(OUT, "long_gaps.csv"), index=False)
    # 3. DO: dong luc, su kien, baseline, cua so
    full = S["all"]
    rows = []
    for n, d in S.items():
        do = d["do_mg_l"]
        day = do.resample("D")
        dr = (day.max() - day.min())[day.count() >= 86]
        sat = do / do_sat(d["water_temp_c"], d["spc_us_cm"].fillna(0)) * 100
        r = dict(split=n, DO_mean=do.mean(), DO_sd=do.std(), diel_range_median=dr.median(), diel_range_p90=dr.quantile(.9),
                 sat_pct_median=sat.median(), sat_pct_p01=sat.quantile(.01), sat_pct_p99=sat.quantile(.99),
                 unchanged_adjacent_pct=(do.diff() == 0).mean() * 100,
                 episodes_lt6=episodes((do < 6).fillna(False)), episodes_lt5=episodes((do < 5).fillna(False)),
                 pct_time_lt6=(do < 6).mean() * 100, pct_time_lt5=(do < 5).mean() * 100,
                 corr_temp=do.corr(d["water_temp_c"]), corr_ph=do.corr(d["ph"]), corr_spc=do.corr(d["spc_us_cm"]),
                 corr_turb=do.corr(d["turbidity_fnu"]), corr_chl=do.corr(d["chl_fluor_ug_l"]))
        for h in (1, 6, 24):
            k = int(h / STEP_H)
            e = (do.shift(-k) - do).dropna()
            r[f"persistence_{h}h_RMSE"] = np.sqrt((e**2).mean())
            r[f"persistence_{h}h_R2"] = 1 - (e**2).mean() / do.var()
        e = (do - do.shift(96)).dropna()
        r["same_time_yesterday_RMSE"] = np.sqrt((e**2).mean())
        r["same_time_yesterday_R2"] = 1 - (e**2).mean() / do.var()
        rows.append(r)
    dyn = pd.DataFrame(rows).round(3)
    dyn.to_csv(os.path.join(OUT, "do_dynamics.csv"), index=False)
    # cua so: nhan nam tron trong split, dau vao duoc lui qua ranh gioi (du lieu qua khu)
    filled = np.column_stack([fill_short(full[c]) for c in F])
    ok = ~np.isnan(filled).any(axis=1)
    cs = np.concatenate([[0], np.cumsum(~ok)])
    win = []
    for n in ["train", "val", "test"]:
        i0 = full.index.get_loc(S[n].index[0]); i1 = full.index.get_loc(S[n].index[-1])
        for lin, lout, lab in [(24, 6, "24h_to_6h"), (168, 24, "7d_to_24h")]:
            a, b = int(lin / STEP_H), int(lout / STEP_H)
            t = np.arange(max(i0, a), i1 - b + 2)
            win.append(dict(split=n, window=lab, count=int(((cs[t + b] - cs[t - a]) == 0).sum())))
    pd.DataFrame(win).to_csv(os.path.join(OUT, "window_counts.csv"), index=False)
    # 4. theo thang va theo che do mua
    rows = []
    do = full["do_mg_l"]
    day = do.resample("D"); dr = (day.max() - day.min())[day.count() >= 86]
    e6 = (do.shift(-24) - do)
    for m in range(1, 13):
        sel = full.index.month == m
        ep6 = sum(episodes((full.loc[f"{y}-{m:02d}", "do_mg_l"] < 6).fillna(False)) for y in sorted(set(full.index.year))
                  if len(full.loc[f"{y}-{m:02d}"] if f"{y}-{m:02d}" in full.index.strftime("%Y-%m") else []))
        ep5 = sum(episodes((full.loc[f"{y}-{m:02d}", "do_mg_l"] < 5).fillna(False)) for y in sorted(set(full.index.year))
                  if f"{y}-{m:02d}" in set(full.index.strftime("%Y-%m")))
        rows.append(dict(month=m, regime=season(m), DO_mean=do[sel].mean(), DO_sd=do[sel].std(), DO_p05=do[sel].quantile(.05),
                         diel_range_median=dr[dr.index.month == m].median(), temp_mean=full.loc[sel, "water_temp_c"].mean(),
                         chl_mean=full.loc[sel, "chl_fluor_ug_l"].mean(), spc_mean=full.loc[sel, "spc_us_cm"].mean(),
                         turb_mean=full.loc[sel, "turbidity_fnu"].mean(),
                         persistence_6h_RMSE=np.sqrt((e6[sel] ** 2).mean()), episodes_lt6=ep6, episodes_lt5=ep5))
    pd.DataFrame(rows).round(3).to_csv(os.path.join(OUT, "monthly_dynamics.csv"), index=False)
    rows = []
    for n in ["train", "val", "test"]:
        d = S[n]
        for reg in ["lanh", "xuan", "nong", "thu"]:
            sel = d.index.month.map(season) == reg
            if not sel.any():
                continue
            do_ = d.loc[sel, "do_mg_l"]
            e = (d["do_mg_l"].shift(-24) - d["do_mg_l"])[sel]
            rows.append(dict(split=n, regime=reg, months=int(pd.Series(d.index[sel].strftime("%Y-%m")).nunique()),
                             rows=int(sel.sum()), DO_mean=do_.mean(), DO_sd=do_.std(),
                             persistence_6h_RMSE=np.sqrt((e**2).mean())))
    pd.DataFrame(rows).round(3).to_csv(os.path.join(OUT, "split_by_regime.csv"), index=False)
    print("written to", OUT)


if __name__ == "__main__":
    main()
