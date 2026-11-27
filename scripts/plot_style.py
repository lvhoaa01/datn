"""Kiểu trình bày chung cho mọi hình của tài liệu (artifacts/dataset/, artifacts/analysis/).

Quy ước: font Roboto (DejaVu Sans nếu máy không có Roboto), chữ 9 pt, nền trắng; không đặt tiêu đề trong hình (chú
thích hình nằm trong tài liệu), các khung đánh dấu (a), (b), ...; màu theo bảng Okabe và Ito (phân biệt được với
người mù màu) và cố định theo vai trò: E1 xanh dương, E2 đỏ cam, E3 xanh lục, baseline xám.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, GRID = "#222222", "#6b6b6b", "#e8e8e8"
BLUE, VERMILION, GREEN, ORANGE, SKY, PURPLE = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7"
EXP_COLOR = {"base": BLUE, "domain": VERMILION, "semantic": GREEN}     # màu của E1, E2, E3
BASELINE = INK2                                                          # baseline giữ nguyên DO
SEASON_COLOR = {"lanh": SKY, "xuan": GREEN, "nong": VERMILION, "thu": ORANGE}
SEASON_NAME = {"lanh": "Lạnh (tháng 11 đến 3)", "xuan": "Xuân (tháng 4, 5)", "nong": "Nóng (tháng 6 đến 9)",
               "thu": "Thu (tháng 10)"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Roboto", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlelocation": "left", "axes.titlepad": 4,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.handlelength": 1.8,
    "lines.linewidth": 1.3, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})


def panel(ax, letter, text=""):
    """Nhãn khung ở góc trên trái: (a), hoặc (a) kèm một cụm từ ngắn."""
    ax.set_title(f"({letter}) {text}".rstrip(), loc="left")


def save(fig, path):
    """Lưu hình PNG (200 dpi) và đóng hình."""
    fig.savefig(path)
    plt.close(fig)
