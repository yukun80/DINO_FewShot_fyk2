# Refactored plotting per user's new requirements:
# - Single x-axis and two y-axes (left: IoU, right: ΔIoU)
# - IoU left axis starts near 65 to keep the line high in the figure
# - ΔIoU bars are visually short (set right-axis limits tightly around data)
# - Clean, publication-ready styling; save PNG & PDF

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

"""
python3 -m tools.step10_iteritet
"""


@dataclass
class CCMIterationData:
    iterations: np.ndarray  # (T,)
    iou: np.ndarray  # (T,)

    @property
    def delta_iou(self) -> np.ndarray:
        d = np.diff(self.iou, prepend=self.iou[0])
        d[0] = 0.0
        return d


@dataclass
class PlotConfig:
    plateau_start: int = 3
    y1_min: float = 63.0  # left axis minimum (IoU)
    y1_pad: float = 0.6  # padding above max
    y2_pad_ratio: float = 0.25  # padding ratio for right axis
    y2_span_scale: float = 1.8  # factor to expand right-axis span without changing ticks
    y2_bottom_expand_ratio: float = 0.2  # fraction of extra span allocated below zero
    y2_label_offset_ratio: float = 0.7  # offset of bar annotations from bottom of the axis span
    line_color: str = "#1b6fd1"
    bar_face_color: str = "#d45500"
    bar_edge_color: str = "#984800"
    baseline_color: str = "#4f4f4f"
    title: str = "Effect of Iteration Times on CCM Performance"
    figsize: Tuple[float, float] = (5.4, 4.6)
    dpi: int = 400
    png_path: str = "./tools/fig/ccm_iteration_analysis_v3.png"
    pdf_path: str = "./tools/fig/ccm_iteration_analysis_v3.pdf"


def build_default_data() -> CCMIterationData:
    iou_values = np.array([71.24, 73.49, 74.34, 74.36, 73.99, 73.73, 73.82, 74.56, 74.89, 75.12], dtype=float)
    iterations = np.arange(1, len(iou_values) + 1, dtype=int)
    return CCMIterationData(iterations=iterations, iou=iou_values)


def make_single_axes_dual_y_plot(data: CCMIterationData, cfg: PlotConfig) -> plt.Figure:
    it = data.iterations
    iou = data.iou
    delta = data.delta_iou

    fig, ax_left = plt.subplots(figsize=cfg.figsize)

    # --- Left Y: IoU line ---
    line = ax_left.plot(
        it,
        iou,
        marker="o",
        linewidth=2,
        label="IoU",
        color=cfg.line_color,
        markerfacecolor="white",
        markeredgewidth=1.1,
    )[0]
    ax_left.set_xlabel("Iteration")
    ax_left.set_ylabel("IoU (%)")
    y1max = np.ceil(iou.max() + cfg.y1_pad)
    ax_left.set_ylim(cfg.y1_min, y1max)
    x_left_limit = it.min() - 0.45
    x_right_limit = it.max() + 0.45
    ax_left.set_xlim(x_left_limit, x_right_limit)
    ax_left.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    # Plateau indicator
    ax_left.axvline(cfg.plateau_start, linestyle="--", linewidth=1, color=cfg.baseline_color, alpha=0.8)
    ax_left.fill_betweenx(
        [cfg.y1_min, y1max],
        cfg.plateau_start,
        x_right_limit,
        alpha=0.06,
        color=cfg.line_color,
    )
    ax_left.text(
        cfg.plateau_start + 0.25,
        y1max - 0.8,
        "Plateau after 3 iterations",
        fontsize=9,
        color=cfg.baseline_color,
    )
    ax_left.text(
        it[0] - 0.35,
        iou[0] + 1.4,
        f"{iou[0]:.2f}%",
        fontsize=8,
        color=cfg.line_color,
        fontstyle="italic",
        ha="left",
        va="bottom",
    )

    # --- Right Y: ΔIoU bars (kept visually short by tight limits) ---
    ax_right = ax_left.twinx()
    bar = ax_right.bar(
        it[1:],
        delta[1:],
        width=0.55,
        label="ΔIoU (pp)",
        alpha=0.9,
        color=cfg.bar_face_color,
        edgecolor=cfg.bar_edge_color,
        linewidth=0.8,
    )
    # Right axis limits tightly around delta to keep bars short and grounded at zero
    dmin, dmax = float(delta[1:].min()), float(delta[1:].max())
    span = max(abs(dmin), abs(dmax))
    pad = max(0.05, cfg.y2_pad_ratio * span)
    base_lower = min(0.0, dmin - pad)
    base_upper = max(0.0, dmax + pad)
    if np.isclose(base_lower, base_upper):
        base_upper = base_lower + 1.0
    ax_right.set_ylim(base_lower, base_upper)
    original_ticks = ax_right.get_yticks()

    # Expand the right-axis span while preserving tick labels for readability.
    if cfg.y2_span_scale > 1.0:
        extra_span = (cfg.y2_span_scale - 1.0) * (base_upper - base_lower)
        bottom_extra = extra_span * np.clip(cfg.y2_bottom_expand_ratio, 0.0, 1.0)
        top_extra = extra_span - bottom_extra
        lower = base_lower - bottom_extra
        upper = base_upper + top_extra
    else:
        lower, upper = base_lower, base_upper
    ax_right.set_ylim(lower, upper)
    ax_right.set_yticks(original_ticks)
    ax_right.set_ylabel("ΔIoU (pp)")
    ax_right.axhline(0.0, color=cfg.baseline_color, linewidth=0.8, linestyle="-", alpha=0.7)

    # Annotate bar deltas beneath the x-axis using italic signed values for quick reading.
    label_y = lower + cfg.y2_label_offset_ratio * (0.0 - lower)
    for x_val, delta_val in zip(it[1:], delta[1:]):
        ax_right.text(
            x_val,
            label_y,
            f"{delta_val:+.2f}",
            ha="center",
            va="top",
            fontsize=8,
            fontstyle="italic",
            color=cfg.bar_edge_color,
        )

    # Keep line above bars visually
    line.set_zorder(3)
    for b in bar:
        b.set_zorder(2)

    # Legend (combined, tucked inside lower-left corner)
    lines1, labels1 = ax_left.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax_left.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="lower left",
        fontsize=8,
        frameon=True,
        facecolor="white",
        edgecolor="#d0d0d0",
        framealpha=0.85,
    )

    # Layout
    fig.tight_layout()

    return fig


def main():
    data = build_default_data()
    cfg = PlotConfig()
    fig = make_single_axes_dual_y_plot(data, cfg)
    fig.savefig(cfg.png_path, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(cfg.pdf_path, bbox_inches="tight")
    print("Saved files:")
    print("PNG:", cfg.png_path)
    print("PDF:", cfg.pdf_path)


if __name__ == "__main__":
    main()
