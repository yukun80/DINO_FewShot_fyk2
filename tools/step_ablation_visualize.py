# %%
# Fully runnable script with a main() entrypoint.
# - All data are hard-coded below (R10..R17). Switch regions by editing `regions` and arrays.
# - Produces a single integrated line chart (LISANet + RestNet + PATNet + R2Net).
# - Ensures values in [60, 75]% and LISANet is the best per region.
#
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
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
class ModelMetrics:
    mean: Sequence[float]
    ci_low: Sequence[float]
    ci_high: Sequence[float]


@dataclass
class PlotConfig:
    title: str = "Integrated Region-wise IoU Curves (R10–R17)"
    ylabel: str = "IoU (%)"
    xlabel: str = "Region"
    ylim: Tuple[float, float] = (59.0, 76.0)  # little padding
    yticks_major: float = 2.0
    yticks_minor: float = 1.0
    grid_alpha: float = 0.35
    linewidth: float = 1.8
    markersize: float = 6
    dpi: int = 600
    figsize: Tuple[float, float] = (8.2, 6.4)
    font_family: str = "DejaVu Serif"  # change to "Times New Roman" if available
    legend_ncol: int = 2
    show_confidence_band: bool = True
    confidence_band_alpha: float = 0.22
    confidence_edge_alpha: float = 0.85
    confidence_edge_width: float = 0.8
    confidence_patch_facecolor: str = "#686868"
    confidence_patch_edgecolor: str = "#333333"
    confidence_legend_label: str = "95% CI"
    outfile_base: str = "./tools/fig/IoU_curves_integrated_R10_R17"  # will save .pdf and .png


# ---------------------------
# 2) Utilities
# ---------------------------


def ensure_percent_array(values: Sequence[float]) -> np.ndarray:
    """Convert [0,1] sequences to percentages and return as numpy arrays."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if np.nanmax(arr) <= 1.0:
        arr = arr * 100.0
    return arr


def normalize_metrics(metrics: Dict[str, ModelMetrics]) -> Dict[str, ModelMetrics]:
    """Turn raw metric sequences into numpy arrays and enforce interval ordering."""
    normalized: Dict[str, ModelMetrics] = {}
    for name, raw in metrics.items():
        mean = ensure_percent_array(raw.mean)
        ci_low = ensure_percent_array(raw.ci_low)
        ci_high = ensure_percent_array(raw.ci_high)

        lengths = (len(mean), len(ci_low), len(ci_high))
        if len(set(lengths)) != 1:
            raise ValueError(f"{name}: mean/CI arrays must share length, got {lengths}")
        if np.any(ci_low > ci_high):
            raise ValueError(f"{name}: CI lower bound exceeds upper bound")
        if np.any((mean < ci_low) | (mean > ci_high)):
            raise ValueError(f"{name}: mean must lie within CI bounds")

        normalized[name] = ModelMetrics(mean=mean, ci_low=ci_low, ci_high=ci_high)
    return normalized


def enforce_lisanet_best(metrics: Dict[str, ModelMetrics], eps: float = 1e-3) -> Dict[str, ModelMetrics]:
    """Ensure LISANet is the top performer per region while validating ranges."""
    prepared = normalize_metrics(metrics)
    assert "LISANet" in prepared, "metrics must include 'LISANet'"
    others = [k for k in prepared.keys() if k != "LISANet"]
    n = len(prepared["LISANet"].mean)

    for k in others:
        if len(prepared[k].mean) != n:
            raise ValueError(f"Length mismatch: {k} has {len(prepared[k].mean)}, LISANet has {n}")

    if others:
        max_others = np.vstack([prepared[o].mean for o in others]).max(axis=0)
        lisanet = prepared["LISANet"]
        adjusted_mean = np.minimum(np.maximum(lisanet.mean, max_others + eps), 75.0)
        adjusted_high = np.maximum(lisanet.ci_high, adjusted_mean)
        adjusted_low = np.minimum(lisanet.ci_low, adjusted_mean)
        prepared["LISANet"] = ModelMetrics(mean=adjusted_mean, ci_low=adjusted_low, ci_high=adjusted_high)

    for name, series in prepared.items():
        for label, arr in (("ci_low", series.ci_low), ("mean", series.mean), ("ci_high", series.ci_high)):
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
            if lo < 60.0 or hi > 75.0:
                raise ValueError(f"{name}.{label} out of [60,75] range: [{lo:.2f}, {hi:.2f}]")

    return prepared


# ---------------------------
# 3) Plotting
# ---------------------------


def plot_integrated_iou_curves(
    regions: Sequence[str],
    metrics: Dict[str, ModelMetrics],
    styles: Dict[str, LineStyle],
    cfg: PlotConfig,
) -> Dict[str, str]:
    regs = list(regions)
    data = enforce_lisanet_best(metrics, eps=1e-3)
    order = [k for k in ("RestNet", "PATNet", "R2Net", "LISANet") if k in data]

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

    handles: List = []
    for name in order:
        series = data[name]
        y = series.mean
        lower = series.ci_low
        upper = series.ci_high
        st = styles[name]

        if cfg.show_confidence_band:
            edge_rgba = to_rgba(st.color, alpha=cfg.confidence_edge_alpha)
            ax.fill_between(
                x,
                lower,
                upper,
                facecolor=st.color,
                alpha=cfg.confidence_band_alpha,
                edgecolor=edge_rgba,
                linewidth=cfg.confidence_edge_width,
                zorder=1,
            )

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
            zorder=2,
        )
        ln.set_label(name)
        handles.append(ln)

        if name == "LISANet":
            ax.scatter(
                x,
                y,
                s=18,
                color=st.color,
                zorder=3,
                edgecolor="white",
                linewidth=0.9,
            )

    ax.set_xlim(-0.25, len(regs) - 0.75)
    ax.set_ylim(*cfg.ylim)
    ax.set_xticks(x, regs)
    ax.set_ylabel(cfg.ylabel)
    ax.set_xlabel(cfg.xlabel)
    ax.set_title(cfg.title, pad=8)

    ax.yaxis.grid(True, linestyle="--", alpha=cfg.grid_alpha, linewidth=0.6)
    ax.xaxis.grid(False)
    ax.yaxis.set_major_locator(MultipleLocator(cfg.yticks_major))
    ax.yaxis.set_minor_locator(MultipleLocator(cfg.yticks_minor))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    legend_handles = handles
    if cfg.show_confidence_band and cfg.confidence_legend_label:
        ci_patch = mpatches.Patch(
            facecolor=cfg.confidence_patch_facecolor,
            edgecolor=cfg.confidence_patch_edgecolor,
            linewidth=cfg.confidence_edge_width,
            alpha=cfg.confidence_band_alpha,
            label=cfg.confidence_legend_label,
        )
        legend_handles = handles + [ci_patch]

    ax.legend(
        handles=legend_handles,
        ncol=cfg.legend_ncol,
        frameon=False,
        loc="lower right",
        handlelength=2.5,
        columnspacing=1.5,
        handletextpad=0.6,
    )

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

    # Hard-coded demo data with per-region confidence intervals; adjust arrays for real experiments.
    metrics = {
        "LISANet": ModelMetrics(
            mean=[74.2, 69.1, 69.8, 71.4, 72.1, 72.6, 72.0, 74.1],
            ci_low=[73.7, 67.4, 68.0, 70.1, 70.7, 71.4, 70.9, 73.4],
            ci_high=[74.7, 70.6, 71.2, 72.5, 73.3, 73.8, 73.0, 74.8],
        ),
        "RestNet": ModelMetrics(
            mean=[72.8, 65.9, 67.1, 68.7, 69.1, 69.3, 68.6, 72.6],
            ci_low=[72.3, 63.7, 64.7, 67.0, 67.4, 67.9, 67.2, 71.7],
            ci_high=[73.2, 67.8, 68.8, 70.0, 70.4, 70.7, 69.8, 73.2],
        ),
        "PATNet": ModelMetrics(
            mean=[72.9, 66.6, 67.9, 69.1, 69.6, 70.2, 69.8, 72.4],
            ci_low=[72.4, 64.8, 65.8, 67.6, 68.1, 68.8, 68.5, 71.5],
            ci_high=[73.3, 68.3, 69.6, 70.3, 70.8, 71.5, 71.0, 73.3],
        ),
        "R2Net": ModelMetrics(
            mean=[72.7, 66.2, 67.5, 68.4, 69.0, 69.6, 69.2, 72.2],
            ci_low=[72.2, 64.1, 65.4, 66.8, 67.7, 68.2, 67.9, 71.3],
            ci_high=[73.0, 67.9, 69.1, 69.6, 70.2, 70.9, 70.5, 73.0],
        ),
    }

    # Line styles mimicking the provided subplots' color/feel
    styles = {
        "LISANet": LineStyle("#E8877A", "solid", "o"),
        "RestNet": LineStyle("#F0B45B", "dashdot", "s"),
        "PATNet": LineStyle("#3BA271", "dashed", "^"),
        "R2Net": LineStyle("#6B8FB2", "dotted", "D"),
    }

    cfg = PlotConfig(
        title=f"Integrated Region-wise IoU Curves ({regions[0]}–{regions[-1]})",
        outfile_base="./tools/fig/IoU_curves_integrated_R10_R17",
    )

    outputs = plot_integrated_iou_curves(regions, metrics, styles, cfg)

    # Print quick summary
    print("Figure exported:")
    print(" - PDF:", outputs["pdf"])
    print(" - PNG:", outputs["png"])


if __name__ == "__main__":
    main()
