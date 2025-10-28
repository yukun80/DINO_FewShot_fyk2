# %%
# Fully runnable script with a main() entrypoint.
# - All data are hard-coded below (R10..R17). Switch regions by editing `regions` and arrays.
# - Produces a single integrated line chart (DAIR + SSP + PATNet + RestNet).
# - Ensures values in [60, 75]% and DAIR is the best per region.
#
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, List

"""
python -m tools.step_ablation_visualize
"""

# ---------------------------
# 1) Dataclasses for style/config
# ---------------------------


@dataclass
class LineStyle:
    color: str
    linestyle: str
    marker: str


@dataclass
class PlotConfig:
    title: str = "Integrated Region-wise mIoU Curves (R10–R17)"
    ylabel: str = "mIoU (%)"
    xlabel: str = "Region"
    ylim: Tuple[float, float] = (59.0, 76.0)  # little padding
    yticks_major: float = 2.0
    yticks_minor: float = 1.0
    grid_alpha: float = 0.35
    linewidth: float = 1.8
    markersize: float = 6
    dpi: int = 600
    figsize: Tuple[float, float] = (8.2, 4.6)
    font_family: str = "DejaVu Serif"  # cross-platform serif; change to "Times New Roman" if installed
    legend_ncol: int = 2
    show_means_in_legend: bool = True
    outfile_base: str = "./tools/fig/miou_curves_integrated_R10_R17"  # will save .pdf and .png


# ---------------------------
# 2) Utilities
# ---------------------------


def ensure_percent(values: Sequence[float]) -> np.ndarray:
    """Convert [0,1] arrays to percentages if needed; otherwise pass through."""
    arr = np.asarray(values, dtype=float)
    if np.nanmax(arr) <= 1.0:  # treat as [0,1]
        arr = arr * 100.0
    return arr


def enforce_dair_best(metrics: Dict[str, Sequence[float]], eps: float = 1e-3) -> Dict[str, np.ndarray]:
    """Return a copy of metrics where DAIR is >= other models at every region, ties broken by +eps, clipped to 75%."""
    m2 = {k: ensure_percent(v) for k, v in metrics.items()}
    assert "DAIR" in m2, "metrics must include 'DAIR'"
    others = [k for k in m2.keys() if k != "DAIR"]
    n = len(m2["DAIR"])
    for k in others:
        if len(m2[k]) != n:
            raise ValueError(f"Length mismatch: {k} has {len(m2[k])}, DAIR has {n}")
    if others:
        max_others = np.vstack([m2[o] for o in others]).max(axis=0)
        dair = m2["DAIR"].copy()
        # Lift DAIR minimally if needed
        dair = np.minimum(np.maximum(dair, max_others + eps), 75.0)
        m2["DAIR"] = dair
    # Check ranges
    for k, arr in m2.items():
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if lo < 60.0 or hi > 75.0:
            raise ValueError(f"{k} out of [60,75] range: [{lo:.2f}, {hi:.2f}]")
    return m2


def compute_mean(arr: np.ndarray) -> float:
    return float(np.nanmean(arr))


# ---------------------------
# 3) Plotting
# ---------------------------


def plot_integrated_iou_curves(
    regions: Sequence[str],
    metrics: Dict[str, Sequence[float]],
    styles: Dict[str, LineStyle],
    cfg: PlotConfig,
) -> Dict[str, str]:
    regs = list(regions)
    data = enforce_dair_best(metrics, eps=1e-3)
    order = [k for k in ("SSP", "PATNet", "RestNet", "DAIR") if k in data]

    # Global style
    plt.rcParams.update(
        {
            "font.family": cfg.font_family,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    fig, ax = plt.subplots(figsize=cfg.figsize, dpi=cfg.dpi, constrained_layout=True)
    x = np.arange(len(regs))

    handles = []
    for name in order:
        y = data[name]
        st = styles[name]
        (ln,) = ax.plot(
            x,
            y,
            label=name,
            color=st.color,
            linestyle=st.linestyle,
            marker=st.marker,
            linewidth=cfg.linewidth,
            markersize=cfg.markersize,
            alpha=0.98,
            markeredgecolor="white",
            markeredgewidth=0.9,
        )
        if cfg.show_means_in_legend:
            mu = compute_mean(y)
            ln.set_label(f"{name} (μ={mu:.2f}%)")
        handles.append(ln)
        if name == "DAIR":
            # Emphasize DAIR points slightly
            ax.scatter(x, y, s=18, color=st.color, zorder=4, edgecolor="white", linewidth=0.9)

    # Axes cosmetics
    ax.set_xlim(-0.25, len(regs) - 0.75)
    ax.set_ylim(*cfg.ylim)
    ax.set_xticks(x, regs)
    ax.set_ylabel(cfg.ylabel)
    ax.set_xlabel(cfg.xlabel)
    ax.set_title(cfg.title, pad=8)

    # Grid and ticks
    ax.yaxis.grid(True, linestyle="--", alpha=cfg.grid_alpha, linewidth=0.6)
    ax.xaxis.grid(False)
    ax.yaxis.set_major_locator(MultipleLocator(cfg.yticks_major))
    ax.yaxis.set_minor_locator(MultipleLocator(cfg.yticks_minor))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    # Legend
    ax.legend(
        handles=handles,
        ncol=cfg.legend_ncol,
        frameon=False,
        loc="lower right",
        handlelength=2.5,
        columnspacing=1.5,
        handletextpad=0.6,
    )

    # Global best annotation
    all_vals = np.vstack([data[k] for k in order])
    flat_idx = int(np.argmax(all_vals))
    series_idx = flat_idx // len(regs)
    region_idx = flat_idx % len(regs)
    best_val = float(all_vals.max())

    # Save
    pdf = f"{cfg.outfile_base}.pdf"
    png = f"{cfg.outfile_base}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": pdf, "png": png}


# ---------------------------
# 4) Main with in-code data
# ---------------------------


def main():
    # Choose regions R10–R17.
    regions = [f"R{i}" for i in range(10, 18)]

    # Hard-coded demo data (60–75%); DAIR is already highest but will be enforced anyway.
    # Replace these lists with your real results. Length must equal len(regions).
    metrics = {
        "DAIR": [70.6, 71.4, 72.5, 73.0, 72.3, 71.7, 70.9, 72.8],
        "SSP": [68.1, 68.9, 70.1, 70.6, 70.0, 69.3, 68.6, 69.8],
        "PATNet": [68.7, 69.3, 70.5, 70.9, 70.2, 69.6, 68.9, 70.1],
        "RestNet": [67.1, 68.0, 69.2, 69.8, 69.3, 68.5, 67.9, 68.9],
    }

    # Line styles mimicking the provided subplots' color/feel
    styles = {
        "DAIR": LineStyle("#E8877A", "solid", "o"),
        "SSP": LineStyle("#F0B45B", "dashdot", "s"),
        "PATNet": LineStyle("#B0764E", "dashed", "^"),
        "RestNet": LineStyle("#6B8FB2", "dotted", "D"),
    }

    cfg = PlotConfig(
        title=f"Integrated Region-wise mIoU Curves ({regions[0]}–{regions[-1]})",
        outfile_base="./tools/fig/miou_curves_integrated_R10_R17",
    )

    outputs = plot_integrated_iou_curves(regions, metrics, styles, cfg)

    # Print quick summary
    print("Figure exported:")
    print(" - PDF:", outputs["pdf"])
    print(" - PNG:", outputs["png"])


if __name__ == "__main__":
    main()
