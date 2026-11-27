"""Phân tích sai số trên test của các mô hình đã huấn luyện lại (scripts/final_train.py),
dùng cho docs/04_MODEL_TRAINING_EVALUATION.md mục 7.

Mỗi điểm sai số là một cặp (cửa sổ test, bước dự báo) có DO đo thật (cờ 0). Điều kiện của một cửa sổ là điều kiện
ở mốc cuối của đầu vào (số đo mới nhất lúc dự báo). Mọi con số là trung bình trên ba seed.
Mọi cột thay_doi_pct_<bộ> là thay đổi MAE (%) của bộ đó so với E1 trên cùng các điểm: 100 (MAE bộ / MAE E1 - 1).
Ghi vào artifacts/analysis/:
  error_overall.csv: trên mọi điểm test: MAE, RMSE, R2 (trung bình và độ lệch chuẩn giữa seed) của baseline giữ
    nguyên DO (bo = giu_nguyen: dự báo mọi bước bằng DO ở mốc cuối của đầu vào) và của từng bộ; thay đổi MAE, RMSE.
  error_by_step.csv: theo bước dự báo (1 đến 24): MAE của từng bộ (và độ lệch chuẩn giữa seed), MAE của baseline.
  error_by_month.csv: theo tháng của thời điểm dự báo: biên độ DO trong ngày (trung vị), MAE từng bộ.
  error_by_condition.csv: tỉ lệ điểm, MAE và tỉ lệ tổng sai số tuyệt đối theo giai đoạn mùa (nhom = mua, docs/01
    mục 3.3) và theo phần trăm bão hòa oxy dưới hoặc từ 100% (nhom = bao_hoa, ranh giới đổi chiều trao đổi khí).
  error_low_do_episodes.csv: cửa sổ quanh từng đợt DO dưới 6 mg/L của test (cửa sổ có thời điểm dự báo trong đợt
    hoặc trong 6 giờ trước khi đợt bắt đầu) và quanh cả năm đợt (dot = tat_ca_dot): số điểm, MAE, sai số trung bình
    có dấu (dương là dự báo cao hơn thật).
  error_low_do_points.csv: các điểm có DO đo dưới 6 mg/L (chọn theo chính giá trị cần dự báo): số mốc, số điểm, MAE,
    sai số trung bình có dấu.
  fig_test_theo_buoc.png, fig_sai_so_theo_thang.png, fig_du_bao_dot_do_thap.png.
Chạy từ gốc repo (sau final_train.py): uv run python scripts/error_analysis.py
"""
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import REPO, do_saturation, load_base_all, metrics
from describe_dataset import episode_spans, season
from plot_style import BASELINE, EXP_COLOR, INK, panel, save
from train import final_dir, iter_runs

OUT = os.path.join(REPO, "artifacts", "analysis")
SETS = {fs: (tag, EXP_COLOR[fs]) for fs, tag in [("base", "E1"), ("domain", "E2"), ("semantic", "E3")]}  # tên, màu
LOW = 6.0                       # ngưỡng DO thấp (mg/L), docs/01_DATASET_ANALYSIS.md mục 3.4
LEAD = 24                       # cửa sổ "dự báo vào đợt": thời điểm dự báo trong 24 mốc (6 giờ) trước khi đợt bắt đầu
SEASONS = ["lanh", "xuan", "nong", "thu"]   # giai đoạn mùa, docs/01_DATASET_ANALYSIS.md mục 3.3
SAT_EDGE = 100.0                            # % bão hòa oxy: dưới 100% nước thiếu oxy, từ 100% nước thừa oxy


def load_preds(model="lstm", cfg="24h_6h"):
    """{bộ đặc trưng: (pred, true, mask, t)}: pred có dạng (seed, cửa sổ, bước), mg/L; các seed cùng thứ tự cửa sổ."""
    out = {}
    for d, c in iter_runs(final_dir(model, cfg)):
        out.setdefault(c["features"], {})[c["seed"]] = d
    res = {}
    for fs, by_seed in out.items():
        z = [np.load(os.path.join(by_seed[s], "pred_test.npz")) for s in sorted(by_seed)]
        assert all((v["t"] == z[0]["t"]).all() for v in z)
        res[fs] = (np.stack([v["pred"] for v in z]), z[0]["true"], z[0]["mask"], z[0]["t"])
    return dict(sorted(res.items(), key=lambda kv: list(SETS).index(kv[0])))


def window_conditions(d, t):
    """Điều kiện ở mốc cuối đầu vào (t - 1) của từng cửa sổ: % bão hòa, trạng thái bão hòa, giai đoạn mùa."""
    last = d.iloc[t - 1]
    sat = 100 * last["do_mg_l"].to_numpy() / do_saturation(last["water_temp_c"], last["spc_us_cm"]).to_numpy()
    time = last["datetime_pst"]
    return pd.DataFrame(dict(do_sat_pct=sat, sat_state=np.where(sat < SAT_EDGE, "duoi_100", "tu_100"),
                             season=time.dt.month.map(season).to_numpy()))


def group_table(preds, groups, labels, name):
    """Tỉ lệ điểm, MAE và tỉ lệ tổng sai số của mỗi nhóm cửa sổ; groups: nhãn nhóm của từng cửa sổ."""
    rows = []
    fs0 = next(iter(preds))
    m = preds[fs0][2]
    n_all = m.sum()
    for g in labels:
        sel = (groups == g)[:, None] & m
        row = dict(nhom=name, khoang=g, ti_le_diem_pct=100 * sel.sum() / n_all)
        for fs, (p, y, _, _) in preds.items():
            e = np.abs(p - y[None])                                   # (seed, cửa sổ, bước)
            mae = np.mean([e[s][sel].mean() for s in range(len(p))])
            share = np.mean([100 * e[s][sel].sum() / e[s][m].sum() for s in range(len(p))])
            row[f"MAE_{SETS[fs][0]}"], row[f"ti_le_sai_so_pct_{SETS[fs][0]}"] = mae, share
        for fs in list(preds)[1:]:
            row[f"thay_doi_pct_{SETS[fs][0]}"] = 100 * (row[f"MAE_{SETS[fs][0]}"] / row[f"MAE_{SETS[fs0][0]}"] - 1)
        rows.append(row)
    return rows


def add_change(row):
    """Thêm vào row các cột thay_doi_pct_<bộ> từ các cột MAE_<bộ> (so với E1)."""
    for tag in [v[0] for v in SETS.values()][1:]:
        if f"MAE_{tag}" in row:
            row[f"thay_doi_pct_{tag}"] = 100 * (row[f"MAE_{tag}"] / row["MAE_E1"] - 1)
    return row


def low_do_tables(preds, d):
    """Hai bảng: cửa sổ quanh từng đợt DO dưới LOW của test (và cả năm đợt); các điểm có DO đo dưới LOW."""
    fs0 = next(iter(preds))
    y, m, t = preds[fs0][1], preds[fs0][2], preds[fs0][3]

    def stats(sel):
        """Số điểm, MAE và sai số có dấu của từng bộ trên các điểm sel, kèm thay đổi MAE."""
        return add_change(dict(so_diem=int(sel.sum()), **{
            k: v for fs, (p, *_) in preds.items() for k, v in err_stats(p, y, sel, SETS[fs][0]).items()}))

    test = d[d["split"] == "test"]
    i0 = test.index[0]
    near = np.zeros(len(t), bool)
    rows = []
    for a, ln in episode_spans(test["do_mg_l"].to_numpy() < LOW):
        start, end = i0 + a, i0 + a + ln - 1
        w = (t >= start - LEAD) & (t <= end)
        near |= w
        rows.append(dict(dot=f"{d.datetime_pst[start]:%Y-%m-%d %H:%M}", so_gio=ln * 0.25,
                         do_thap_nhat=d.do_mg_l[start:end + 1].min(), **stats(w[:, None] & m)))
    rows.append(dict(dot="tat_ca_dot", **stats(near[:, None] & m)))
    low = (y < LOW) & m
    obs = test["do_mg_l"].where(test["flag_do_mg_l"] == 0)
    points = [dict(nguong=LOW, so_moc=int((obs < LOW).sum()), **stats(low))]
    return pd.DataFrame(rows), pd.DataFrame(points)


def err_stats(p, y, sel, tag):
    """MAE và sai số trung bình có dấu (dự báo trừ thật) trên các điểm sel, trung bình trên các seed."""
    return {f"MAE_{tag}": np.mean([np.abs(p[s] - y)[sel].mean() for s in range(len(p))]),
            f"sai_so_co_dau_{tag}": np.mean([(p[s] - y)[sel].mean() for s in range(len(p))])}


def fig_steps(steps, sets):
    """MAE theo bước dự báo của từng bộ đặc trưng và của baseline giữ nguyên DO (đọc từ error_by_step.csv)."""
    fig, ax = plt.subplots(figsize=(4.8, 3))
    for fs in sets:
        ax.plot(steps.phut / 60, steps[f"MAE_{SETS[fs][0]}"], color=SETS[fs][1], lw=1.6, label=SETS[fs][0])
    ax.plot(steps.phut / 60, steps["MAE_giu_nguyen"], color=BASELINE, ls="--", label="Baseline giữ nguyên DO")
    ax.set_xticks(range(7))
    ax.set_xlim(0, 6.1)
    ax.set_xlabel("Bước dự báo (giờ)")
    ax.set_ylabel("MAE trên test (mg/L)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, os.path.join(OUT, "fig_test_theo_buoc.png"))


def fig_months(mon, preds):
    """Hai khung chung trục tháng: (a) biên độ DO trong ngày; (b) MAE từng bộ."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    x = np.arange(len(mon))
    a1.bar(x, mon.bien_do_ngay_mgl, 0.6, color=BASELINE)
    a1.set_ylabel("Biên độ DO\ntrong ngày (mg/L)")
    w = 0.8 / len(preds)
    for j, fs in enumerate(preds):
        tag = SETS[fs][0]
        a2.bar(x + (j - (len(preds) - 1) / 2) * w, mon[f"MAE_{tag}"], w * 0.92, color=SETS[fs][1], label=tag)
    a2.set_ylabel("MAE trên test (mg/L)")
    a2.set_xticks(x, [k[5:] for k in mon.thang])
    a2.set_xlabel("Tháng (năm 2025)")
    a2.legend(loc="upper left")
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout()
    save(fig, os.path.join(OUT, "fig_sai_so_theo_thang.png"))


def fig_low_do(preds, d):
    """DO đo và dự báo ở bước 15 phút, 6 giờ của từng bộ (trung bình ba seed) trong tuần có hai đợt DO thấp đầu tiên
    của test."""
    fs0 = next(iter(preds))
    t = preds[fs0][3]
    time = d.datetime_pst
    lo, hi = pd.Timestamp("2025-08-10"), pd.Timestamp("2025-08-17")
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.2), sharex=True)
    for ax, (k, lead), key in zip(axes, [(0, "15 phút"), (23, "6 giờ")], "ab"):
        tgt = t + k                                   # dòng của mốc được dự báo ở bước k + 1
        sel = (time[tgt].to_numpy() >= lo) & (time[tgt].to_numpy() < hi)
        ax.plot(time[tgt[sel]], preds[fs0][1][sel, k], color=INK, lw=0.9)
        for fs, (p, *_) in preds.items():
            ax.plot(time[tgt[sel]], p[:, sel, k].mean(0), color=SETS[fs][1], lw=1)
        ax.axhline(LOW, color=INK, lw=0.7, ls="--")
        ax.set_ylim(3, 14.5)
        ax.set_ylabel("DO (mg/L)")
        panel(ax, key, f"Bước dự báo {lead}")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    fig.legend([plt.Line2D([], [], color=INK, lw=0.9)] + [plt.Line2D([], [], color=SETS[fs][1]) for fs in preds],
               ["DO đo"] + [SETS[fs][0] for fs in preds], loc="lower center", ncol=1 + len(preds),
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save(fig, os.path.join(OUT, "fig_du_bao_dot_do_thap.png"))


def main():
    """Tính và ghi các bảng, hình phân tích sai số trên test."""
    os.makedirs(OUT, exist_ok=True)
    d = load_base_all()
    preds = load_preds()
    assert "base" in preds, "chưa có lần chấm test của bộ base: chạy scripts/final_train.py"
    p0, y0, m0, t0 = preds["base"]
    naive = d["do_mg_l"].to_numpy()[t0 - 1]
    naive_all = np.repeat(naive[:, None], y0.shape[1], axis=1)
    # toàn bộ test: baseline và từng bộ; số của các bộ phải trùng bảng chấm test của final_train.py
    tb = pd.read_csv(os.path.join(final_dir("lstm", "24h_6h"), "test_tb.csv")).set_index("features")
    overall = [dict(bo="giu_nguyen", **metrics(naive_all, y0, m0))]
    for fs, (p, y, m, _) in preds.items():
        ms = pd.DataFrame([metrics(p[s], y, m) for s in range(len(p))])
        row = dict(bo=SETS[fs][0])
        for c in ("MAE", "RMSE", "R2"):
            row[c], row[f"{c}_sd"] = ms[c].mean(), ms[c].std(ddof=1)
            assert abs(row[c] - tb.loc[fs, f"{c}_mean"]) < 1e-6, f"{fs} {c} lệch test_tb.csv"
        overall.append(row)
    overall = pd.DataFrame(overall)
    e1 = overall.set_index("bo").loc["E1"]
    for c in ("MAE", "RMSE"):
        overall[f"thay_doi_{c}_pct"] = np.where(overall.bo.isin(["giu_nguyen", "E1"]), np.nan,
                                                100 * (overall[c] / e1[c] - 1))
    overall.round(6).to_csv(os.path.join(OUT, "error_overall.csv"), index=False)
    # theo bước
    steps = []
    for k in range(y0.shape[1]):
        row = dict(buoc=k + 1, phut=15 * (k + 1))
        for fs, (p, y, m, _) in preds.items():
            v = [np.abs(p[s][:, k] - y[:, k])[m[:, k]].mean() for s in range(len(p))]
            row[f"MAE_{SETS[fs][0]}"], row[f"MAE_sd_{SETS[fs][0]}"] = np.mean(v), np.std(v, ddof=1)
        row["MAE_giu_nguyen"] = np.abs(naive - y0[:, k])[m0[:, k]].mean()
        steps.append(add_change(row))
    steps = pd.DataFrame(steps)
    steps.round(6).to_csv(os.path.join(OUT, "error_by_step.csv"), index=False)
    # theo điều kiện
    cond = window_conditions(d, preds["base"][3])
    rows = (group_table(preds, cond.season.to_numpy(), SEASONS, "mua")
            + group_table(preds, cond.sat_state.to_numpy(), ["duoi_100", "tu_100"], "bao_hoa"))
    tab = pd.DataFrame(rows)
    tab.round(6).to_csv(os.path.join(OUT, "error_by_condition.csv"), index=False)
    episodes, points = low_do_tables(preds, d)
    episodes.round(6).to_csv(os.path.join(OUT, "error_low_do_episodes.csv"), index=False)
    points.round(6).to_csv(os.path.join(OUT, "error_low_do_points.csv"), index=False)
    # theo tháng: biên độ DO trong ngày của test (số đo thật) và MAE
    test = d[d["split"] == "test"].set_index("datetime_pst")
    do_obs = test["do_mg_l"].where(test["flag_do_mg_l"] == 0)
    day = do_obs.resample("D")
    dr = (day.max() - day.min())[day.count() >= 86]
    month = d.datetime_pst.dt.strftime("%Y-%m").to_numpy()[t0]               # tháng của thời điểm dự báo
    mon = []
    for k in sorted(set(month)):
        row = dict(thang=k, bien_do_ngay_mgl=dr[dr.index.strftime("%Y-%m") == k].median())
        sel = (month == k)[:, None] & m0
        for fs, (p, y, _, _) in preds.items():
            row[f"MAE_{SETS[fs][0]}"] = np.mean([np.abs(p[s] - y)[sel].mean() for s in range(len(p))])
        mon.append(add_change(row))
    mon = pd.DataFrame(mon)
    mon.round(6).to_csv(os.path.join(OUT, "error_by_month.csv"), index=False)
    # các hình đọc lại đúng file nguồn ghi trong tài liệu
    fig_months(pd.read_csv(os.path.join(OUT, "error_by_month.csv")), preds)
    fig_steps(pd.read_csv(os.path.join(OUT, "error_by_step.csv")), list(preds))
    fig_low_do(preds, d)
    print(overall.round(4).to_string())
    print(tab.round(3).to_string())
    print(episodes.round(3).to_string())
    print(points.round(3).to_string())


if __name__ == "__main__":
    main()
