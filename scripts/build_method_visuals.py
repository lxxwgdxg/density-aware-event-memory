from __future__ import annotations

"""Build the Journal of Hydrology graphical abstract and technical method figure.

The drawings are deliberately code-native vector graphics.  All workflow arrows
terminate exactly on box boundaries, and the top-level layouts are checked for
box overlap before export.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


OUT = Path.cwd() / "reproduced_visuals"

NAVY = "#17324D"
BLUE = "#2369A6"
BLUE_LIGHT = "#EDF5FC"
TEAL = "#008F83"
TEAL_LIGHT = "#EAF7F4"
ORANGE = "#D97706"
ORANGE_LIGHT = "#FFF4E6"
GREEN = "#2E7D32"
GREEN_LIGHT = "#EDF7ED"
PURPLE = "#6B5B95"
PURPLE_LIGHT = "#F2EFF8"
RED = "#C53B3E"
INK = "#27343D"
MUTED = "#5E6E78"
GRID = "#D8E2E8"
PAPER = "#FFFFFF"
SOFT = "#F6F8FA"


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float
    name: str

    @property
    def left(self) -> tuple[float, float]:
        return self.x, self.y + self.h / 2

    @property
    def right(self) -> tuple[float, float]:
        return self.x + self.w, self.y + self.h / 2

    @property
    def top(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h

    @property
    def bottom(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": PAPER,
        }
    )


def rounded_box(
    ax: plt.Axes,
    box: Box,
    *,
    face: str,
    edge: str,
    title: str,
    body: str = "",
    title_size: float = 11,
    body_size: float = 9.2,
    linewidth: float = 1.7,
    radius: float = 0.025,
    title_y: float = 0.69,
    body_y: float = 0.36,
    zorder: int = 4,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (box.x, box.y),
        box.w,
        box.h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        box.x + box.w / 2,
        box.y + box.h * title_y,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=edge,
        zorder=zorder + 1,
    )
    if body:
        ax.text(
            box.x + box.w / 2,
            box.y + box.h * body_y,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=INK,
            linespacing=1.12,
            zorder=zorder + 1,
        )
    return patch


def small_card(
    ax: plt.Axes,
    box: Box,
    top: str,
    bottom: str,
    *,
    face: str,
    edge: str,
    top_size: float = 10.2,
    bottom_size: float = 8.2,
) -> None:
    patch = FancyBboxPatch(
        (box.x, box.y), box.w, box.h,
        boxstyle="round,pad=0.004,rounding_size=0.014",
        linewidth=1.2, edgecolor=edge, facecolor=face, zorder=5,
    )
    ax.add_patch(patch)
    ax.text(box.x + box.w / 2, box.y + box.h * 0.66, top, ha="center", va="center",
            fontsize=top_size, fontweight="bold", color=edge, zorder=6)
    ax.text(box.x + box.w / 2, box.y + box.h * 0.29, bottom, ha="center", va="center",
            fontsize=bottom_size, color=INK, linespacing=1.05, zorder=6)


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    width: float = 1.8,
    mutation: float = 14,
    zorder: int = 3,
    connectionstyle: str = "arc3,rad=0",
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=width,
        color=color,
        shrinkA=0,
        shrinkB=0,
        connectionstyle=connectionstyle,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def connect_boxes(ax: plt.Axes, source: Box, target: Box, **kwargs) -> FancyArrowPatch:
    return arrow(ax, source.right, target.left, **kwargs)


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(-0.035, 1.035, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=15, fontweight="bold", color=INK)
    ax.text(0.02, 1.035, title, transform=ax.transAxes, ha="left", va="top",
            fontsize=12.2, fontweight="bold", color=INK)


def boxes_overlap(a: Box, b: Box, pad: float = 0.004) -> bool:
    return not (
        a.x + a.w + pad <= b.x
        or b.x + b.w + pad <= a.x
        or a.y + a.h + pad <= b.y
        or b.y + b.h + pad <= a.y
    )


def assert_no_overlap(boxes: Iterable[Box]) -> None:
    boxes = list(boxes)
    for i, first in enumerate(boxes):
        for second in boxes[i + 1 :]:
            if boxes_overlap(first, second):
                raise RuntimeError(f"Boxes overlap: {first.name} and {second.name}")


def save(fig: plt.Figure, stem: str, *, dpi: int = 300) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def build_graphical_abstract() -> None:
    fig, ax = plt.subplots(figsize=(13.28, 5.31))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)

    ax.text(0.5, 0.93, "GOVERN THE OBSERVATION, NOT JUST THE FORECAST",
            ha="center", va="center", fontsize=17.5, fontweight="bold", color=NAVY)
    ax.text(0.5, 0.875,
            "Protect forecast evidence  •  admit only a positive state margin  •  contract authority with lead",
            ha="center", va="center", fontsize=10.8, color=MUTED)

    forecast = Box(0.025, 0.55, 0.17, 0.22, "forecast input")
    state = Box(0.025, 0.22, 0.17, 0.22, "state input")
    anchor = Box(0.245, 0.55, 0.19, 0.22, "anchor")
    signal = Box(0.245, 0.22, 0.19, 0.22, "signal")
    gate = Box(0.535, 0.275, 0.25, 0.44, "admission gate")
    outcome = Box(0.835, 0.22, 0.14, 0.55, "outcome")
    assert_no_overlap([forecast, state, anchor, signal, gate, outcome])

    # FancyBboxPatch expands by its pad. Use the rendered outer boundary rather
    # than the nominal Box coordinates, and keep connectors below the boxes so
    # line caps and arrow tips cannot intrude into the fills.
    box_pad = 0.008

    def outer_left(box: Box) -> tuple[float, float]:
        return box.x - box_pad, box.y + box.h / 2

    def outer_right(box: Box) -> tuple[float, float]:
        return box.x + box.w + box_pad, box.y + box.h / 2

    arrow(ax, outer_right(forecast), outer_left(anchor), color=BLUE, width=2.0, zorder=3)
    arrow(ax, outer_right(state), outer_left(signal), color=ORANGE, width=2.0, zorder=3)

    decision_center = (0.485, 0.495)
    decision_half_w = 0.027
    decision_half_h = 0.055
    decision_upper_left = (
        decision_center[0] - decision_half_w / 2,
        decision_center[1] + decision_half_h / 2,
    )
    decision_lower_left = (
        decision_center[0] - decision_half_w / 2,
        decision_center[1] - decision_half_h / 2,
    )
    decision_right = (decision_center[0] + decision_half_w, decision_center[1])
    arrow(ax, outer_right(anchor), decision_upper_left, color=BLUE, width=1.9, zorder=3)
    arrow(ax, outer_right(signal), decision_lower_left, color=ORANGE, width=1.9, zorder=3)
    arrow(ax, decision_right, outer_left(gate), color=TEAL, width=2.2, zorder=3)
    arrow(ax, outer_right(gate), outer_left(outcome), color=GREEN, width=2.2, zorder=3)

    rounded_box(ax, forecast, face=BLUE_LIGHT, edge=BLUE, title="FORECAST EVIDENCE",
                body="Meteorological trajectory\nBasin attributes\nTransferred analogues", title_size=10.4, body_size=8.7)
    rounded_box(ax, state, face=ORANGE_LIGHT, edge=ORANGE, title="ISSUE-TIME STATE",
                body="Current discharge  $q_t$\nFrozen local threshold  $q^*$", title_size=10.4, body_size=8.8)
    rounded_box(ax, anchor, face=BLUE_LIGHT, edge=BLUE, title="PROTECTED ANCHOR  A",
                body="Base rank + positive margins\nfrom meteorological and\ntransferred evidence", title_size=10.4, body_size=8.6)
    rounded_box(ax, signal, face=ORANGE_LIGHT, edge=ORANGE, title="BOUNDED SIGNAL  S",
                body="Threshold-relative river state\ninformative before and\nduring exceedance", title_size=10.4, body_size=8.6)

    decision_points = np.array(
        [
            (decision_center[0], decision_center[1] + decision_half_h),
            (decision_center[0] + decision_half_w, decision_center[1]),
            (decision_center[0], decision_center[1] - decision_half_h),
            (decision_center[0] - decision_half_w, decision_center[1]),
        ]
    )
    ax.add_patch(
        Polygon(
            decision_points,
            closed=True,
            facecolor=PAPER,
            edgecolor=TEAL,
            linewidth=1.7,
            joinstyle="round",
            zorder=5,
        )
    )
    ax.text(
        *decision_center,
        "S > A?",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=TEAL,
        zorder=6,
    )

    rounded_box(ax, gate, face=TEAL_LIGHT, edge=TEAL, title="LEAD-ADAPTIVE ADMISSION",
                body=r"$D = A + w(h,r)\,\max(S-A,0)$", title_size=11.2, body_size=10.0,
                title_y=0.82, body_y=0.61)
    ax.text(gate.x + gate.w / 2, gate.y + gate.h * 0.47,
            "State can add evidence;\nit cannot erase the anchor",
            ha="center", va="center", fontsize=8.2, color=TEAL, fontweight="bold",
            linespacing=1.05, zorder=7)
    cards = [
        Box(gate.x + 0.018, gate.y + 0.055, 0.062, 0.105, "H1"),
        Box(gate.x + 0.094, gate.y + 0.055, 0.062, 0.105, "H5"),
        Box(gate.x + 0.170, gate.y + 0.055, 0.062, 0.105, "H7"),
    ]
    small_card(ax, cards[0], "H1", "w = 1.00", face=PAPER, edge=TEAL, top_size=9.5, bottom_size=7.8)
    small_card(ax, cards[1], "H5", "0–0.50", face=PAPER, edge=TEAL, top_size=9.5, bottom_size=7.8)
    small_card(ax, cards[2], "H7", "0–0.30", face=PAPER, edge=TEAL, top_size=9.5, bottom_size=7.8)

    rounded_box(ax, outcome, face=GREEN_LIGHT, edge=GREEN, title="EVIDENCE AT SCALE",
                body="981 basins  •  10 groups\nQ95 / Q98 / Q99\nH1 / H5 / H7",
                title_size=10.0, body_size=8.5, title_y=0.87, body_y=0.67)
    ax.text(outcome.x + outcome.w / 2, outcome.y + outcome.h * 0.43,
            "449 / 450", ha="center", va="center", fontsize=15, fontweight="bold", color=GREEN, zorder=7)
    ax.text(outcome.x + outcome.w / 2, outcome.y + outcome.h * 0.33,
            "source–seed gains\nnonnegative", ha="center", va="center", fontsize=8.2, color=INK, zorder=7)
    ax.text(outcome.x + outcome.w / 2, outcome.y + outcome.h * 0.13,
            "MORE TRUE EXTREMES\nSAME ALERT VOLUME", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=GREEN, zorder=7)

    ax.text(0.5, 0.085,
            "Conditional river-state authority improves extreme-flow ranking without increasing alert volume",
            ha="center", va="center", fontsize=11.3, fontweight="bold", color=NAVY)

    save(fig, "Graphical_Abstract", dpi=300)


def panel_a(ax: plt.Axes) -> None:
    panel_label(ax, "a", "Build a protected forecast anchor")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    stack = Box(0.03, 0.29, 0.30, 0.52, "score stack")
    margin = Box(0.39, 0.35, 0.30, 0.40, "margin rule")
    anchor = Box(0.77, 0.39, 0.19, 0.32, "anchor A")
    assert_no_overlap([stack, margin, anchor])
    connect_boxes(ax, stack, margin, color=BLUE, width=2.0)
    connect_boxes(ax, margin, anchor, color=BLUE, width=2.0)
    rounded_box(ax, stack, face=BLUE_LIGHT, edge=BLUE, title="FORECAST SCORE STACK", body="",
                title_size=9.3, title_y=0.88)
    rows = [
        ("B", "meteorology + attributes\n+ support"),
        ("M", "meteorological–static\nrank"),
        ("T", "support-aware transferred\nevidence"),
    ]
    for i, (symbol, desc) in enumerate(rows):
        y = stack.y + stack.h * (0.64 - i * 0.23)
        ax.text(stack.x + 0.045, y, symbol, ha="center", va="center", fontsize=12,
                fontweight="bold", color=BLUE, zorder=7)
        ax.text(stack.x + 0.080, y, desc, ha="left", va="center", fontsize=7.15,
                color=INK, linespacing=1.05, zorder=7)
    rounded_box(ax, margin, face=PAPER, edge=BLUE, title="POSITIVE-MARGIN\nPROTECTION",
                body="Only evidence above B can raise\nthe anchor; weaker components\ncannot reverse the base ranking",
                title_size=8.6, body_size=7.7, title_y=0.76, body_y=0.35)
    rounded_box(ax, anchor, face=BLUE_LIGHT, edge=BLUE, title="PROTECTED\nANCHOR",
                body="A", title_size=8.8, body_size=24, title_y=0.76, body_y=0.32)
    ax.text(0.50, 0.17, "A = B + 0.5 max(M − B, 0) + 0.5 max(T − B, 0)",
            ha="center", va="center", fontsize=10.2, fontweight="bold", color=NAVY)
    ax.text(0.50, 0.085, "Forecast evidence is protected before local discharge is consulted",
            ha="center", va="center", fontsize=8.9, color=MUTED)


def panel_b(ax: plt.Axes) -> None:
    panel_label(ax, "b", "Transform state and contract its authority with lead")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    state = Box(0.025, 0.55, 0.23, 0.30, "state")
    transform = Box(0.335, 0.55, 0.32, 0.30, "transform")
    signal = Box(0.735, 0.55, 0.235, 0.30, "signal")
    assert_no_overlap([state, transform, signal])
    connect_boxes(ax, state, transform, color=ORANGE, width=1.9)
    connect_boxes(ax, transform, signal, color=ORANGE, width=1.9)
    rounded_box(ax, state, face=ORANGE_LIGHT, edge=ORANGE, title="LOCAL STATE",
                body=r"$q_t$ relative to frozen $q^*$", title_size=9.0, body_size=8.2,
                title_y=0.72, body_y=0.33)
    rounded_box(ax, transform, face=PAPER, edge=ORANGE, title="THRESHOLD-RELATIVE\nTRANSFORM",
                body=r"$\log(1+q_t)-\log(1+q^*)$", title_size=8.2, body_size=8.6,
                title_y=0.72, body_y=0.31)
    rounded_box(ax, signal, face=ORANGE_LIGHT, edge=ORANGE, title="BOUNDED\nSTATE SIGNAL",
                body="$S\\in(0,1)$\nactive before exceedance", title_size=8.8, body_size=8.0,
                title_y=0.72, body_y=0.30)

    ax.text(0.04, 0.43, "Lead contract  w(h,r)", ha="left", va="center",
            fontsize=10.2, fontweight="bold", color=TEAL)
    x0, width = 0.19, 0.72
    labels = ["H1", "H5", "H7"]
    supported = [1.00, 0.35, 0.21]
    frontier = [1.00, 0.50, 0.30]
    for i, label in enumerate(labels):
        y = 0.34 - i * 0.105
        ax.text(0.10, y, label, ha="center", va="center", fontsize=9.3, fontweight="bold")
        ax.add_patch(FancyBboxPatch((x0, y - 0.027), width, 0.054,
                                    boxstyle="round,pad=0,rounding_size=0.018",
                                    facecolor="#EDF1F3", edgecolor="none", zorder=1))
        ax.add_patch(FancyBboxPatch((x0, y - 0.027), width * frontier[i], 0.054,
                                    boxstyle="round,pad=0,rounding_size=0.018",
                                    facecolor="#8BD0C6", edgecolor="none", zorder=2))
        ax.add_patch(FancyBboxPatch((x0, y - 0.017), width * supported[i], 0.034,
                                    boxstyle="round,pad=0,rounding_size=0.012",
                                    facecolor=TEAL, edgecolor="none", zorder=3))
        ax.text(x0 + width + 0.025, y, f"{frontier[i]:.2f}", ha="left", va="center", fontsize=8.2, color=TEAL)
    ax.text(0.19, 0.064, "dark: supported regime     light: support-frontier regime",
            ha="left", va="center", fontsize=7.8, color=MUTED)
    ax.text(0.19, 0.025, "hyper-wet supported regime: state admission is off at H5 and H7",
            ha="left", va="center", fontsize=7.4, color=MUTED)


def panel_c(ax: plt.Axes) -> None:
    panel_label(ax, "c", "Admit state only through an auditable positive margin")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    anchor = Box(0.04, 0.52, 0.19, 0.24, "A")
    signal = Box(0.04, 0.19, 0.19, 0.24, "S")
    decision_center = (0.39, 0.475)
    update = Box(0.56, 0.52, 0.38, 0.24, "update")
    preserve = Box(0.56, 0.19, 0.38, 0.24, "preserve")
    assert_no_overlap([anchor, signal, update, preserve])
    arrow(ax, anchor.right, decision_center, color=BLUE, width=1.9)
    arrow(ax, signal.right, decision_center, color=ORANGE, width=1.9)
    arrow(ax, (0.47, 0.475), update.left, color=TEAL, width=1.9,
          connectionstyle="angle3,angleA=45,angleB=180")
    arrow(ax, (0.47, 0.475), preserve.left, color=MUTED, width=1.8,
          connectionstyle="angle3,angleA=-45,angleB=180")
    rounded_box(ax, anchor, face=BLUE_LIGHT, edge=BLUE, title="ANCHOR", body="A",
                title_size=9.8, body_size=18)
    rounded_box(ax, signal, face=ORANGE_LIGHT, edge=ORANGE, title="STATE", body="S",
                title_size=9.8, body_size=18)
    diamond = Polygon([(0.39, 0.57), (0.47, 0.475), (0.39, 0.38), (0.31, 0.475)],
                      closed=True, facecolor=TEAL_LIGHT, edgecolor=TEAL, linewidth=1.7, zorder=4)
    ax.add_patch(diamond)
    ax.text(0.39, 0.475, "S > A?", ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=TEAL, zorder=5)
    ax.text(0.49, 0.59, "YES", ha="center", va="center", fontsize=7.8, fontweight="bold", color=TEAL)
    ax.text(0.49, 0.36, "NO", ha="center", va="center", fontsize=7.8, fontweight="bold", color=MUTED)
    rounded_box(ax, update, face=TEAL_LIGHT, edge=TEAL, title="ADMIT INCREMENTAL\nEVIDENCE",
                body=r"$D=A+w(h,r)\,[S-A]$", title_size=8.6, body_size=9.2,
                title_y=0.72, body_y=0.29)
    rounded_box(ax, preserve, face=SOFT, edge=MUTED, title="PRESERVE FORECAST\nRANK",
                body=r"$D=A$", title_size=8.6, body_size=10.5,
                title_y=0.72, body_y=0.29)
    ax.text(0.50, 0.085,
            "Monotone contract: local state may lift the score,\nbut it cannot suppress stronger forecast evidence",
            ha="center", va="center", fontsize=7.8, color=NAVY, fontweight="bold", linespacing=1.05)


def panel_d(ax: plt.Axes) -> None:
    panel_label(ax, "d", "Test breadth, transfer and independent confirmation")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    flow = [
        Box(0.025, 0.66, 0.19, 0.20, "memberships"),
        Box(0.275, 0.66, 0.19, 0.20, "unique"),
        Box(0.525, 0.66, 0.19, 0.20, "final"),
        Box(0.775, 0.66, 0.19, 0.20, "units"),
    ]
    assert_no_overlap(flow)
    for first, second in zip(flow, flow[1:]):
        connect_boxes(ax, first, second, color=PURPLE, width=1.8, mutation=12)
    small_card(ax, flow[0], "1,043", "source\nmemberships", face=PURPLE_LIGHT, edge=PURPLE, bottom_size=7.2)
    small_card(ax, flow[1], "1,005", "unique external\ngauges", face=PURPLE_LIGHT, edge=PURPLE, bottom_size=7.2)
    small_card(ax, flow[2], "981", "non-overlapping\nbasins", face=PURPLE_LIGHT, edge=PURPLE, bottom_size=7.2)
    small_card(ax, flow[3], "450", "source–seed\nevaluations", face=PURPLE_LIGHT, edge=PURPLE, bottom_size=7.2)

    cards = [
        Box(0.025, 0.23, 0.215, 0.27, "primary"),
        Box(0.27, 0.23, 0.215, 0.27, "nested"),
        Box(0.515, 0.23, 0.215, 0.27, "bootstrap"),
        Box(0.76, 0.23, 0.215, 0.27, "postlock"),
    ]
    assert_no_overlap(cards)
    small_card(ax, cards[0], "449 / 450", "primary AP units\nnonnegative", face=GREEN_LIGHT, edge=GREEN, bottom_size=7.2)
    small_card(ax, cards[1], "90 / 90", "held-out medians\nnonnegative", face=GREEN_LIGHT, edge=GREEN, bottom_size=7.2)
    small_card(ax, cards[2], "9 / 9", "block-bootstrap\nlower bounds > 0", face=GREEN_LIGHT, edge=GREEN, bottom_size=7.2)
    small_card(ax, cards[3], "45 / 45", "post-lock units\nnonnegative", face=GREEN_LIGHT, edge=GREEN, bottom_size=7.2)
    ax.plot([0.13, 0.87], [0.57, 0.57], color=GRID, linewidth=1.0)
    ax.text(0.50, 0.085, "Breadth  →  policy transfer  →  source-level uncertainty  →  fresh-archive confirmation",
            ha="center", va="center", fontsize=8.7, color=NAVY, fontweight="bold")


def build_method_figure() -> None:
    fig = plt.figure(figsize=(12.2, 9.2), facecolor=PAPER)
    positions = [
        (0.055, 0.535, 0.43, 0.405),
        (0.535, 0.535, 0.43, 0.405),
        (0.055, 0.065, 0.43, 0.405),
        (0.535, 0.065, 0.43, 0.405),
    ]
    axes = [fig.add_axes(pos) for pos in positions]
    for ax in axes:
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
                                    boxstyle="round,pad=0.012,rounding_size=0.018",
                                    facecolor=PAPER, edgecolor=GRID, linewidth=1.0, zorder=-5))
    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])
    panel_d(axes[3])
    save(fig, "Figure_1", dpi=400)


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(
        description="Build JoH Figure 1 and the graphical abstract."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT,
        help="Directory for Figure_1 and Graphical_Abstract PDF/PNG files.",
    )
    args = parser.parse_args()
    OUT = args.out_dir.resolve()
    setup()
    build_graphical_abstract()
    build_method_figure()
    print(OUT)


if __name__ == "__main__":
    main()
