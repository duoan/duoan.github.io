"""Shared matplotlib helpers for LLM training series figures.

Used by playground/llm_training_series_figures.py. Style matches the GPU
optimization playbook figures (light schematic SVGs, English labels).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
    }
)

INK = "#1f2933"
MUTED = "#64748b"
BLUE = "#4C72B0"
GREEN = "#55A868"
RED = "#C44E52"
ORANGE = "#DD8452"
PURPLE = "#8172B3"
LINE = "#94a3b8"

FILL = "#f8fafc"
BLUE_FILL = "#e8eef7"
GREEN_FILL = "#e7f1ea"
RED_FILL = "#f7e9e8"
ORANGE_FILL = "#faefe6"
PURPLE_FILL = "#efecf5"
YELLOW_FILL = "#fef9e7"
GRAY_FILL = "#eef1f4"


def box(ax, xy, w, h, text, *, fc=FILL, ec=INK, lw=1.4, fontsize=10, weight="normal", tc=INK):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            mutation_aspect=1.0,
        )
    )
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=tc,
            fontweight=weight,
        )


def rect(ax, xy, w, h, *, fc=FILL, ec=INK, lw=1.0, alpha=1.0):
    ax.add_patch(Rectangle(xy, w, h, facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha))


def arrow(ax, start, end, *, color=LINE, style="-|>", lw=1.4):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=13,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def clean(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def save(fig, path: Path, title: str | None = None):
    if title:
        fig.suptitle(title, fontsize=12, color=INK, y=0.98)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
