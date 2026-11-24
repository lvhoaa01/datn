"""Phân tích sai số trên test của các mô hình đã huấn luyện lại (scripts/final_train.py),
dùng cho docs/04_MODEL_TRAINING_EVALUATION.md mục 7 và 9.

Mỗi điểm sai số là một cặp (cửa sổ test, bước dự báo) có DO đo thật (cờ 0). Điều kiện của một cửa sổ là điều kiện
ở mốc cuối của đầu vào (số đo mới nhất lúc dự báo). Mọi con số là trung bình trên ba seed.
Ghi vào artifacts/analysis/:
  error_by_step.csv: MAE theo bước dự báo, kèm mốc tham chiếu "giữ nguyên DO" (bo = giu_nguyen: dự báo mọi bước
    bằng DO ở mốc cuối của đầu vào), để biết một bước dự báo khó đến đâu.
  error_by_month.csv: theo tháng của thời điểm dự báo: chênh lệch DO trong ngày (trung vị), MAE từng bộ.
  error_by_season.csv, error_by_condition.csv: tỉ lệ điểm, MAE và tỉ lệ tổng sai số tuyệt đối theo giai đoạn mùa,
    nhiệt độ nước, phần trăm bão hòa oxy, giờ trong ngày; thay_doi_pct là MAE của bộ so với bộ base.
  error_low_do.csv: các điểm có DO thật dưới 6 mg/L và từng đợt DO dưới 6 mg/L của test: số điểm, MAE, sai số
    trung bình có dấu (dương là dự báo cao hơn thật).
  error_summary.csv: thay đổi MAE (%) của từng bộ so với bộ base trên mọi lát cắt ở trên.
  fig_test_theo_buoc.png, fig_sai_so_theo_dieu_kien.png, fig_du_bao_dot_do_thap.png, fig_sai_so_theo_thang.png,
  fig_e2_so_voi_e1.png.
Chạy từ gốc repo (sau final_train.py): uv run python scripts/error_analysis.py
"""
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import REPO, do_saturation, load_base_all
from describe_dataset import episode_spans, season
from figures import AQUA, BLUE, INK, INK2, ORANGE, RED
from train import final_dir, iter_runs

OUT = os.path.join(REPO, "artifacts", "analysis")
SETS = {"base": ("E1", BLUE), "domain": ("E2", ORANGE), "semantic": ("E3", AQUA)}   # bộ đặc trưng: tên, màu
LOW = 6.0                       # ngưỡng DO thấp (mg/L), docs/01_DATASET_ANALYSIS.md mục 3.4
LEAD = 24                       # cửa sổ "dự báo vào đợt": thời điểm dự báo trong 24 mốc (6 giờ) trước khi đợt bắt đầu
CONDITIONS = {
    "nhiet_do": ("water_temp_c", [-np.inf, 18, 24, np.inf], ["dưới 18", "18 đến 24", "từ 24"]),
    "bao_hoa_pct": ("do_sat_pct", [-np.inf, 90, 110, 130, np.inf], ["dưới 90", "90 đến 110", "110 đến 130", "từ 130"]),
    "gio": ("hour", [0, 6, 12, 18, 24], ["0 đến 6 giờ", "6 đến 12 giờ", "12 đến 18 giờ", "18 đến 24 giờ"]),
}
SEASON_NAME = {"lanh": "lạnh", "xuan": "xuân", "nong": "nóng", "thu": "thu"}


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
    """Điều kiện ở mốc cuối đầu vào (t - 1) của từng cửa sổ: nhiệt độ, % bão hòa, giờ, giai đoạn mùa."""
    last = d.iloc[t - 1]
    sat = 100 * last["do_mg_l"].to_numpy() / do_saturation(last["water_temp_c"], last["spc_us_cm"]).to_numpy()
    time = last["datetime_pst"]
    return pd.DataFrame(dict(water_temp_c=last["water_temp_c"].to_numpy(), do_sat_pct=sat,
                             hour=(time.dt.hour + time.dt.minute / 60).to_numpy(),
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


def low_do_table(preds, d):
    """Sai số ở các điểm DO thật dưới LOW và theo từng đợt DO thấp của test."""
    fs0 = next(iter(preds))
    y, m, t = preds[fs0][1], preds[fs0][2], preds[fs0][3]
    low = (y < LOW) & m
    rows = [dict(doi_tuong=f"mọi điểm DO thật dưới {LOW:g} mg/L", so_diem=int(low.sum()), **{
        k: v for fs, (p, *_) in preds.items() for k, v in err_stats(p, y, low, SETS[fs][0]).items()})]
    test = d[d["split"] == "test"]
    i0 = test.index[0]
    for a, ln in episode_spans(test["do_mg_l"].to_numpy() < LOW):
        start, end = i0 + a, i0 + a + ln - 1
        sel = ((t >= start - LEAD) & (t <= end))[:, None] & m
        lab = f"đợt {d.datetime_pst[start]:%Y-%m-%d %H:%M}, {ln * 0.25:g} giờ, DO thấp nhất {d.do_mg_l[start:end + 1].min():.2f}"
        rows.append(dict(doi_tuong=lab, so_diem=int(sel.sum()), **{
            k: v for fs, (p, *_) in preds.items() for k, v in err_stats(p, y, sel, SETS[fs][0]).items()}))
    return pd.DataFrame(rows)


def err_stats(p, y, sel, tag):
    """MAE và sai số trung bình có dấu (dự báo trừ thật) trên các điểm sel, trung bình trên các seed."""
    return {f"MAE_{tag}": np.mean([np.abs(p[s] - y)[sel].mean() for s in range(len(p))]),
            f"sai_so_co_dau_{tag}": np.mean([(p[s] - y)[sel].mean() for s in range(len(p))])}


def fig_steps(steps):
    """MAE theo bước dự báo của từng bộ đặc trưng và của mốc giữ nguyên DO."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for fs, g in steps.groupby("bo", sort=False):
        if fs == "giu_nguyen":
            ax.plot(g.phut / 60, g.MAE, color=INK2, lw=1.5, ls="--", label="giữ nguyên DO hiện tại")
        else:
            ax.plot(g.phut / 60, g.MAE, color=SETS[fs][1], lw=2, label=SETS[fs][0])
    ax.set_xlabel("bước dự báo (giờ tới)")
    ax.set_ylabel("MAE trên test (mg/L)")
    ax.set_title("Sai số tăng theo khoảng dự báo")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_test_theo_buoc.png"), dpi=150)
    plt.close(fig)


def fig_conditions(tab, preds):
    """MAE theo nhóm điều kiện (nhiệt độ, % bão hòa, giờ), kèm tỉ lệ điểm của mỗi nhóm."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    titles = {"nhiet_do": "Nhiệt độ nước (độ C)", "bao_hoa_pct": "Bão hòa oxy (%)", "gio": "Giờ dự báo"}
    w = 0.8 / len(preds)
    for ax, (name, g) in zip(axes, tab.groupby("nhom", sort=False)):
        x = np.arange(len(g))
        for j, fs in enumerate(preds):
            tag = SETS[fs][0]
            ax.bar(x + (j - (len(preds) - 1) / 2) * w, g[f"MAE_{tag}"], w * 0.92, color=SETS[fs][1], label=tag)
        ax.set_xticks(x, [f"{k}\n{v:.0f}% điểm" for k, v in zip(g.khoang, g.ti_le_diem_pct)])
        ax.set_title(titles[name])
    axes[0].set_ylabel("MAE trên test (mg/L)")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_sai_so_theo_dieu_kien.png"), dpi=150)
    plt.close(fig)


def fig_low_do(preds, d):
    """DO thật và dự báo 1 giờ, 6 giờ trước của từng bộ trong tuần có các đợt DO thấp của test."""
    fs0 = next(iter(preds))
    t = preds[fs0][3]
    time = d.datetime_pst
    lo, hi = pd.Timestamp("2025-08-10"), pd.Timestamp("2025-08-17")
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True)
    for ax, (k, lead) in zip(axes, [(3, "1 giờ"), (23, "6 giờ")]):
        tgt = t + k                                   # dòng của mốc được dự báo ở bước k
        sel = (time[tgt].to_numpy() >= lo) & (time[tgt].to_numpy() < hi)
        ax.plot(time[tgt[sel]], preds[fs0][1][sel, k], color=INK, lw=1.4, label="DO đo")
        for fs, (p, *_) in preds.items():
            ax.plot(time[tgt[sel]], p[:, sel, k].mean(0), color=SETS[fs][1], lw=1.2,
                    label=f"{SETS[fs][0]}, dự báo trước {lead}")
        ax.axhline(LOW, color=INK2, lw=0.8, ls="--")
        ax.set_ylim(3, 14.5)
        ax.set_ylabel("DO (mg/L)")
        ax.set_title(f"Dự báo trước {lead} (trung bình ba seed)")
        ax.legend(loc="upper right", ncol=3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_du_bao_dot_do_thap.png"), dpi=150)
    plt.close(fig)


def fig_months(mon, preds):
    """Hai khung chung trục tháng: chênh lệch DO trong ngày, và MAE từng bộ."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    x = np.arange(len(mon))
    a1.bar(x, mon.chenh_lech_ngay_mgl, 0.6, color=INK2)
    a1.set_ylabel("mg/L")
    a1.set_title("Chênh lệch DO trong ngày (trung vị theo tháng)")
    w = 0.8 / len(preds)
    for j, fs in enumerate(preds):
        tag = SETS[fs][0]
        a2.bar(x + (j - (len(preds) - 1) / 2) * w, mon[f"MAE_{tag}"], w * 0.92, color=SETS[fs][1], label=tag)
    a2.set_ylabel("MAE trên test (mg/L)")
    a2.set_title("Sai số dự báo theo tháng của thời điểm dự báo")
    a2.set_xticks(x, mon.thang)
    a2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_sai_so_theo_thang.png"), dpi=150)
    plt.close(fig)


def fig_summary(summ):
    """Thay đổi MAE (%) của bộ thứ hai so với bộ base trên mọi lát cắt: âm là tốt hơn."""
    tag = [c for c in summ.columns if c.startswith("thay_doi_pct_")][0]
    v = summ[tag].to_numpy()
    y = np.arange(len(summ))[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(summ) + 1.2))
    ax.barh(y, v, 0.7, color=[AQUA if a < 0 else RED for a in v])
    for yi, a in zip(y, v):
        ax.text(a + (0.4 if a >= 0 else -0.4), yi, f"{a:+.1f}%", va="center", ha="left" if a >= 0 else "right",
                fontsize=7.5, color=INK)
    ax.set_yticks(y, summ.lat_cat)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(min(v.min() * 1.3, -5), max(v.max() * 1.4, 5))
    ax.set_xlabel(f"MAE của {tag[-2:]} so với E1 (%): âm là tốt hơn E1, dương là kém hơn")
    ax.set_title(f"{tag[-2:]} so với E1 trên từng lát cắt của test")
    ax.grid(axis="x", color="#e6e5e0")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_e2_so_voi_e1.png"), dpi=150)
    plt.close(fig)


def main():
    """Tính và ghi các bảng, hình phân tích sai số trên test."""
    os.makedirs(OUT, exist_ok=True)
    d = load_base_all()
    preds = load_preds()
    assert "base" in preds, "chưa có lần chấm test của bộ base: chạy scripts/final_train.py"
    # theo bước
    steps = []
    for fs, (p, y, m, _) in preds.items():
        for k in range(y.shape[1]):
            v = [np.abs(p[s][:, k] - y[:, k])[m[:, k]].mean() for s in range(len(p))]
            steps.append(dict(bo=fs, buoc=k + 1, phut=15 * (k + 1), MAE=np.mean(v), MAE_sd=np.std(v, ddof=1)))
    p0, y0, m0, t0 = preds["base"]
    naive = d["do_mg_l"].to_numpy()[t0 - 1]
    for k in range(y0.shape[1]):
        steps.append(dict(bo="giu_nguyen", buoc=k + 1, phut=15 * (k + 1),
                          MAE=np.abs(naive - y0[:, k])[m0[:, k]].mean(), MAE_sd=0.0))
    steps = pd.DataFrame(steps)
    steps.round(6).to_csv(os.path.join(OUT, "error_by_step.csv"), index=False)
    # theo điều kiện
    cond = window_conditions(d, preds["base"][3])
    seas = group_table(preds, cond.season.to_numpy(), ["lanh", "xuan", "nong", "thu"], "mua")
    for r in seas:
        r["khoang"] = SEASON_NAME[r["khoang"]]
    pd.DataFrame(seas).round(6).to_csv(os.path.join(OUT, "error_by_season.csv"), index=False)
    rows = []
    for name, (col, bins, labels) in CONDITIONS.items():
        g = pd.cut(cond[col], bins, right=False, labels=labels).astype(str).to_numpy()
        rows += group_table(preds, g, labels, name)
    tab = pd.DataFrame(rows)
    tab.round(6).to_csv(os.path.join(OUT, "error_by_condition.csv"), index=False)
    low = low_do_table(preds, d)
    low.round(6).to_csv(os.path.join(OUT, "error_low_do.csv"), index=False)
    # theo tháng: chênh lệch DO trong ngày của test (số đo thật) và MAE
    test = d[d["split"] == "test"].set_index("datetime_pst")
    do_obs = test["do_mg_l"].where(test["flag_do_mg_l"] == 0)
    day = do_obs.resample("D")
    dr = (day.max() - day.min())[day.count() >= 86]
    month = d.datetime_pst.dt.strftime("%Y-%m").to_numpy()[t0]               # tháng của thời điểm dự báo
    mon = []
    for k in sorted(set(month)):
        row = dict(thang=k, chenh_lech_ngay_mgl=dr[dr.index.strftime("%Y-%m") == k].median())
        sel = (month == k)[:, None] & m0
        for fs, (p, y, _, _) in preds.items():
            row[f"MAE_{SETS[fs][0]}"] = np.mean([np.abs(p[s] - y)[sel].mean() for s in range(len(p))])
        mon.append(row)
    mon = pd.DataFrame(mon)
    mon.round(6).to_csv(os.path.join(OUT, "error_by_month.csv"), index=False)
    # tổng hợp thay đổi so với base
    if len(preds) > 1:
        fs1 = list(preds)[1]
        t1, tb = SETS[fs1][0], SETS["base"][0]
        tot = {fs: np.mean([np.abs(p[s] - y)[m].mean() for s in range(len(p))]) for fs, (p, y, m, _) in preds.items()}
        summ = [dict(lat_cat="toàn bộ test", **{f"thay_doi_pct_{t1}": 100 * (tot[fs1] / tot["base"] - 1)})]
        st = steps.set_index(["bo", "buoc"]).MAE
        for k, lab in [(1, "bước 15 phút"), (2, "bước 30 phút"), (4, "bước 1 giờ"), (12, "bước 3 giờ"), (24, "bước 6 giờ")]:
            summ.append(dict(lat_cat=lab, **{f"thay_doi_pct_{t1}": 100 * (st[fs1, k] / st["base", k] - 1)}))
        names = {"mua": "mùa", "nhiet_do": "nhiệt độ", "bao_hoa_pct": "bão hòa %", "gio": "giờ"}
        for r in seas + rows:
            summ.append(dict(lat_cat=f"{names[r['nhom']]} {r['khoang']}", **{f"thay_doi_pct_{t1}": r[f"thay_doi_pct_{t1}"]}))
        for _, r in low.iterrows():
            lab = "điểm DO dưới 6 mg/L" if r.doi_tuong.startswith("mọi") else "quanh " + r.doi_tuong.split(",")[0]
            summ.append(dict(lat_cat=lab, **{f"thay_doi_pct_{t1}": 100 * (r[f"MAE_{t1}"] / r[f"MAE_{tb}"] - 1)}))
        summ = pd.DataFrame(summ)
        summ.round(6).to_csv(os.path.join(OUT, "error_summary.csv"), index=False)
        fig_summary(summ)
    fig_months(mon, preds)
    fig_steps(steps)
    fig_conditions(tab, preds)
    fig_low_do(preds, d)
    print(tab.round(3).to_string())
    print(low.round(3).to_string())


if __name__ == "__main__":
    main()
