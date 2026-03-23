from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from glori.plotting.utils import auto_log_bins
import numpy as np


def source_matching_scatter_plot(
    combined_dict,
    label=None,
    save_path=None,
    fig_ax=None,
):
    all_inp = combined_dict["all_inp"]
    all_outp = combined_dict["all_outp"]

    # Initialize figure
    fig, axs = (
        fig_ax
        if fig_ax is not None
        else plt.subplots(
            2,
            3,
            figsize=(18, 12),
            sharex="col",
            constrained_layout=True,
            subplot_kw=dict(
                box_aspect=1,
            ),
        )
    )

    titles = ["Total_flux", "Peak_flux", "Maj"]

    for i, title in enumerate(titles):

        # First row: scatter plot with binned median and 1-sigma lines
        ax = axs[0][i]

        # Scatter plot
        ax.scatter(
            all_inp[i], all_outp[i], s=0.5, alpha=0.6, label=label, color="lightskyblue"
        )
        if label is not None:
            ax.legend()

        # binning for median and 1-sigma
        bin_edges = auto_log_bins([all_inp[i], all_outp[i]], num=20)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        digitized = np.digitize(all_inp[i], bin_edges)
        medians, means, sigmas_low, sigmas_high = [
            np.zeros(len(bin_edges) - 1) for _ in range(4)
        ]
        for j in range(1, len(bin_edges)):
            bin_vals = all_outp[i][digitized == j]
            if len(bin_vals):
                medians[j - 1] = np.median(bin_vals)
                means[j - 1] = np.mean(bin_vals)
                sigmas_low[j - 1] = np.percentile(bin_vals, 16)
                sigmas_high[j - 1] = np.percentile(bin_vals, 84)
        nonzero_mask = medians > 0
        medians = medians[nonzero_mask]
        means = means[nonzero_mask]
        sigmas_low = sigmas_low[nonzero_mask]
        sigmas_high = sigmas_high[nonzero_mask]

        # Plot lines
        ax.plot(
            bin_centers[nonzero_mask],
            medians,
            color="forestgreen",
            lw=2,
            label="Binned median",
        )
        ax.fill_between(
            bin_centers[nonzero_mask],
            sigmas_high,
            sigmas_low,
            color="forestgreen",
            alpha=0.4,
            label="16-84th percentile",
        )
        ax.set_title(title.replace("_", " ").capitalize())

        # Second row:2D histogram
        ax = axs[1][i]
        ax.hist2d(
            all_inp[i],
            all_outp[i],
            bins=auto_log_bins([all_inp[i], all_outp[i]], num=100),
            cmap="viridis",
            norm=LogNorm(),
        )

        # set the axis limits of row 1 to those of row 2
        # (Becuase here we use optimized bin range in auto_log_bins())
        axs[0][i].set_xlim(ax.get_xlim())
        axs[0][i].set_ylim(ax.get_ylim())

        # Plot 1:1 line
        for ax in axs.flatten():
            ax.axline(
                [
                    all_inp[i].mean(),
                ]
                * 2,
                slope=1,
                ls="--",
                color="red",
                alpha=0.3,
            )

    # Set axis scales and grid
    for ax in axs.flatten():
        # ax.set_box_aspect(1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)

    # Add labels
    fig.supxlabel("Input Values")
    axs[0][0].set_ylabel("Generated Values")
    axs[1][0].set_ylabel("Generated Values")

    # Save plot
    if save_path is not None:
        if isinstance(save_path, str):
            save_path = Path(save_path)
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)

        fig.savefig(str(save_path), dpi=300)

    return fig, axs
