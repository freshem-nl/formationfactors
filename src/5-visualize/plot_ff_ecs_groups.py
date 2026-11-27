"""Figure of the final FF/ECs groups used for the Monte Carlo look-up table.

One row per lithoclass. Columns follow the look-up priority:
  1) stratigraphy-specific statistics (special_stats in look_up_table_monte_carlo.py)
  2) facies groups (lithofacies_for_monte_carlo.csv): one box = one distribution,
     several chips in a box = merged facies
  3) remaining facies -> lithoclass statistics (litho_for_monte_carlo.csv)
Chips are coloured by facies; stratigraphy chips take the colour of their facies.
Lithoclasses g, sch and a (literature / proxy values) are not shown.

Run from the repository root:  python src/5-visualize/plot_ff_ecs_groups.py [--sip3-only]
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.textpath import TextPath

sys.path.append(str(Path(__file__).resolve().parent))
from ff_ecs_groups import FACIES_OF, build_groups  # noqa: E402

FACIES_ORDER = ["marien", "fluviatiel", "glaciaal", "eolisch", "organisch"]
FACIES_COLOR = {"marien": "#2a78d6", "fluviatiel": "#1baf7a", "glaciaal": "#6250d6",
                "eolisch": "#eda100", "organisch": "#eb6834"}
FACIES_TEXT = {"marien": "white", "fluviatiel": "#0b0b0b", "glaciaal": "white",
               "eolisch": "#0b0b0b", "organisch": "#0b0b0b"}
INK, INK2, MUTED, BOX_EDGE, SURFACE, RULE = "#0b0b0b", "#52514e", "#8a8984", "#b9b8b2", "#fcfcfb", "#e6e5e0"

LITHO_ROWS = [("v", "veen"), ("k", "klei"), ("kz", "kleiig zand, zandige\nklei en leem"),
              ("zf", "zand fijn"), ("zm", "zand midden"), ("zg", "zand grof")]

# all geometry in points (1 data unit = 1 pt)
FS_CHIP, FS_CAP, FS_CODE, FS_NAME, FS_HEAD = 9, 7.5, 11, 7.5, 10
CHIP_H, CHIP_PADX, CHIP_GAP = 16, 5, 4
BOX_PAD, BOX_GAP, COL_GAP = 6, 8, 22
ROW_H, BOX_H = 52, 42
FONT = FontProperties(family="DejaVu Sans")
FONT_B = FontProperties(family="DejaVu Sans", weight="bold")


def tw(s, size, bold=False):
    """Rendered width of a (possibly multi-line) string in points."""
    fp = FONT_B if bold else FONT
    return max(TextPath((0, 0), line, size=size, prop=fp).get_extents().width for line in s.split("\n"))


def chip_w(label):
    return tw(label, FS_CHIP) + 2 * CHIP_PADX


def box_w(chips, caption):
    inner = sum(chip_w(c[0]) for c in chips) + CHIP_GAP * max(len(chips) - 1, 0)
    return max(inner, tw(caption, FS_CAP)) + 2 * BOX_PAD


def cells_w(cells):
    return sum(box_w(*c) for c in cells) + BOX_GAP * max(len(cells) - 1, 0)


def draw_box(ax, x, yc, chips, caption, dashed=False):
    w = box_w(chips, caption)
    ax.add_patch(FancyBboxPatch((x, yc - BOX_H / 2), w, BOX_H, boxstyle="round,pad=0,rounding_size=4",
                                fc=SURFACE, ec=BOX_EDGE, lw=1.0, ls=(0, (3, 2)) if dashed else "-"))
    cx, cy = x + BOX_PAD, yc + BOX_H / 2 - BOX_PAD - CHIP_H / 2
    for label, fac in chips:
        cw = chip_w(label)
        ax.add_patch(FancyBboxPatch((cx, cy - CHIP_H / 2), cw, CHIP_H,
                                    boxstyle="round,pad=0,rounding_size=3", fc=FACIES_COLOR[fac], ec="none"))
        ax.text(cx + cw / 2, cy, label, ha="center", va="center_baseline", fontsize=FS_CHIP,
                color=FACIES_TEXT[fac], fontproperties=FONT)
        cx += cw + CHIP_GAP
    ax.text(x + BOX_PAD, yc - BOX_H / 2 + 5, caption, ha="left", va="bottom", fontsize=FS_CAP,
            color=INK2, fontproperties=FONT)
    return w


def main(sip5=True, out=None):
    facies, special, _, litho_n = build_groups(sip5)

    rows = []
    for litho, name in LITHO_ROWS:
        c1 = [([(m, FACIES_OF[m]) for m in r.members], f"n = {r.n}")
              for r in special.loc[special["litho"] == litho].itertuples()]
        fg = facies.loc[facies["litho"] == litho]
        c2 = [([(m, m) for m in r.members], f"n = {r.n}") for r in fg.itertuples()]
        used = {m for ms in fg["members"] for m in ms}
        rest = [f for f in FACIES_ORDER if f not in used]
        c3 = [([(f, f) for f in rest], f"lithoklasse {litho}, n = {litho_n.get(litho, 0)}")] if rest else []
        rows.append((litho, name, c1, c2, c3))

    heads = ["1  Stratigrafie-specifiek", "2  Faciesgroepen", "3  Overige facies → lithoklasse"]
    label_w = max(max(tw(n, FS_NAME) for _, n, *_ in rows), max(tw(l, FS_CODE, True) for l, *_ in rows))
    col_w = [max(max(cells_w(r[i + 2]) for r in rows), tw(heads[i], FS_HEAD, True)) for i in range(3)]
    x0 = 4
    xs = [x0 + label_w + COL_GAP]
    for w in col_w[:-1]:
        xs.append(xs[-1] + w + COL_GAP)
    width = xs[-1] + col_w[-1] + 8

    head_h, legend_h = 26, 46
    height = head_h + len(rows) * ROW_H + legend_h
    fig = plt.figure(figsize=(width / 72, height / 72))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    ytop = height - 4
    for x, h in zip(xs, heads):
        ax.text(x, ytop, h, fontsize=FS_HEAD, fontproperties=FONT_B, color=INK, va="top")

    yc = height - head_h - ROW_H / 2
    for litho, name, c1, c2, c3 in rows:
        ax.text(x0, yc + BOX_H / 2, litho, fontsize=FS_CODE, fontproperties=FONT_B, color=INK, va="top")
        ax.text(x0, yc + BOX_H / 2 - 16, name, fontsize=FS_NAME, color=INK2, va="top",
                fontproperties=FONT, linespacing=1.1)
        for x, cells, dashed in [(xs[0], c1, False), (xs[1], c2, False), (xs[2], c3, True)]:
            if not cells:
                ax.text(x + 2, yc, "–", color=MUTED, va="center", fontsize=FS_CHIP)
            for chips, cap in cells:
                x += draw_box(ax, x, yc, chips, cap, dashed=dashed) + BOX_GAP
        ax.plot([x0, width - 8], [yc - ROW_H / 2, yc - ROW_H / 2], color=RULE, lw=0.8)
        yc -= ROW_H

    handles = [Patch(fc=FACIES_COLOR[f], ec="none", label=f) for f in FACIES_ORDER]
    leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(x0 / width, 4 / height), ncol=5,
                    frameon=False, fontsize=8.5, handlelength=1.2, columnspacing=1.2, borderaxespad=0)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.text(width - 8, 6,
            "Eén kader = één verdeling (lognormaal); meerdere chips in een kader = samengevoegd.\n"
            "Kolom 1 gaat voor op 2, kolom 2 voor op 3. " + ("SIP3 + SIP5." if sip5 else "Alleen SIP3."),
            ha="right", va="bottom", fontsize=FS_CAP, color=INK2, fontproperties=FONT)

    out = out or Path("reports/figures") / f"ff_ecs_groups{'_SIP3_SIP5' if sip5 else '_SIP3'}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2])
    p = argparse.ArgumentParser()
    p.add_argument("--sip3-only", action="store_true", help="exclude SIP5 fills (default: SIP3 + SIP5)")
    a = p.parse_args()
    main(sip5=not a.sip3_only)
