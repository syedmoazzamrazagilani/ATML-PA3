import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "baseline": "#4D4D4D",
    "full_ft_mid": "#8C96C6",
    "full_ft": "#1F78B4",
    "l2_sp": "#33A02C",
    "lora_4": "#FB9A99",
    "lora_16": "#E31A1C",
    "oplora_16": "#6A3D9A",
}

def apply_neurips_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
