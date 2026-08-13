#!/usr/bin/env python3
"""Plot the public vector-residual correction summary on Test-4000."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "figure_source" / "fig3_vector_correction_all4000.csv"
OUT = ROOT / "results" / "figures"


def main() -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8.0,
                         "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none"})
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT)
    required = {"D", "xtb_lowest_energy_mu", "vector_mu", "xtb_abs_error", "vector_abs_error"}
    if len(data) != 4000 or not required.issubset(data.columns):
        raise ValueError("Expected the released 4,000-record vector correction table")
    xtb_error = data["xtb_abs_error"].to_numpy(float)
    vector_error = data["vector_abs_error"].to_numpy(float)
    grid = np.linspace(0, max(xtb_error.max(), vector_error.max()), 400)
    fig, ax = plt.subplots(figsize=(5.1, 3.45), constrained_layout=True)
    for values, color, label in ((xtb_error, "#8e99a2", f"xTB baseline; median = {np.median(xtb_error):.2f} D"),
                                 (vector_error, "#4f779e", f"Equivariant vector; median = {np.median(vector_error):.2f} D")):
        ax.plot(grid, np.searchsorted(np.sort(values), grid, side="right") / len(values), linewidth=2.0, color=color, label=label)
    ax.set(xlabel="Absolute dipole-magnitude error (D)", ylabel="Cumulative fraction", xlim=(0, grid.max()), ylim=(0, 1.02),
           title="Residual learning improves the xTB dipole baseline")
    ax.legend(frameon=False, loc="lower right", fontsize=6.9)
    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})):
        fig.savefig(OUT / f"Fig_vector_residual_summary.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
