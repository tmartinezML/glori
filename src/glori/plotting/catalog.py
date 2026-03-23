import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from astropy.table import Table
import glori.settings.paths as paths
import pandas as pd

from glori.plotting.utils import add_distribution_plot, auto_log_bins
from glori.analysis.stats_utils import err_poisson


def match_inv_cumul(d2d, logax=(True, False)):
    counts, bins = np.histogram(d2d, bins=auto_log_bins(d2d))

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.stairs(1 - counts.cumsum() / counts.sum(), bins)

    if logax[0]:
        ax.set_xscale("log")
    if logax[1]:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)

    ax.set_xlabel("Distance (arcsec)")
    ax.set_ylabel("Cumulative Fraction above Distance")

    return fig, ax


def match_scatter_plots(cat1, cat2, logax=(True, True)):
    fig, axs = plt.subplots(3, 1, figsize=(8, 18))

    for i, col in enumerate(["Total_flux", "Peak_flux", "Maj"]):

        ax = axs[i]
        ax.scatter(
            cat1[col], cat2[col], alpha=0.5, s=10, color="red", label="Matched Catalog"
        )
        ax.set_xlabel(
            f'SRL {col.replace("_", " ")} (mJy)'
            if col != "Maj"
            else f'Catalog {col.replace("_", " ")} (arcsec)'
        )
        ax.set_ylabel(
            f'Catalog {col.replace("_", " ")} (mJy)'
            if col != "Maj"
            else f'SRL {col.replace("_", " ")} (arcsec)'
        )
        if logax[0]:
            ax.set_xscale("log")
        if logax[1]:
            ax.set_yscale("log")

        xmin, xmax, ymin, ymax = ax.axis()
        ax.axline(
            (min(xmin, ymin),) * 2,
            (max(xmax, ymax),) * 2,
            color="black",
            linestyle="--",
            label="y=x",
            alpha=0.5,
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.3)
        ax.legend()

    return fig, axs


def holy_trinity_histograms(
    catalogs,
    catalog_names=None,
    make_plot=[True, True, True],
    areas_degsq=None,
    density=False,
    fig_width=12,
    n_bins=100,
    nonzero_bin_thresh=10,
    save=False,
    out_prefix="",
    output_dir=paths.ANALYSIS_PARENT / "plots",
    logax=(True, True),
    cmap="Set1",
    cmap_max=None,
    stairs_kw={},
    **plot_dict_kw,
):

    if isinstance(catalogs, pd.DataFrame):
        catalogs = [catalogs]
    assert isinstance(catalogs, list), "catalogs must be a list of DataFrames"
    # Plot those quantities:
    # (this will be used for axis labeling)
    metrics = ["Integrated Flux", "Peak Flux", "Major Axis"]
    units = ["Jy", "Jy/beam", "arcsec"]
    # Define getter functions
    get_raw = lambda df, kw: df[kw].values

    if density and areas_degsq is not None:
        print("Ignoring density=True because areas were passed.")
        density = False

    cmap = plt.get_cmap(cmap)

    plot_dict = {
        "catalog_names": catalog_names,
        "catalogs": catalogs,
        # The following are: (column name, getter function)
        # Used to extract data from catalogs
        "Integrated Flux": [
            ("Total_flux", get_raw),
        ]
        * len(catalogs),
        "Peak Flux": [
            ("Peak_flux", get_raw),
        ]
        * len(catalogs),
        "Major Axis": [
            ("Maj", get_raw),
        ]
        * len(catalogs),
        "colors": [
            cmap(i)
            for i in np.linspace(
                0, cmap_max or ((len(catalogs) - 1) / len(cmap.colors)), len(catalogs)
            )
        ],
    }
    plot_dict.update(plot_dict_kw)

    figs = {}
    print("Making plots...")
    for i_metric, metric in enumerate(metrics):
        if not make_plot[i_metric]:
            continue
        # Initialize figure
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width * 2 / 3), dpi=100)

        metric_values = []
        for i_cat, cat in enumerate(plot_dict["catalogs"]):
            # Get catalog and metric getter
            kw, get_metric = plot_dict[metric][i_cat]
            # Get metric values
            metric_values.append(get_metric(cat, kw))

        # Calculate histograms
        bins = (
            auto_log_bins([m for m in metric_values], num=n_bins)
            if logax[0]
            else n_bins
        )

        # These will be used for setting the x limits
        all_counts = []
        all_areas = []
        for i_cat, values in enumerate(metric_values):
            c, bins = np.histogram(values, bins=bins, density=False)
            err_lo, err_hi = err_poisson(c)
            # Calculate factor for normalization
            area = 1
            if density:
                # Area under the histogram
                area = np.sum(c) * (bins[1] - bins[0])
            elif areas_degsq is not None:
                # Survey area in sr
                area = areas_degsq[i_cat] * (np.pi / 180) ** 2
            all_areas.append(area)
            # Normalize counts to sr
            c = c / area
            err_lo = err_lo / area
            err_hi = err_hi / area
            kw = dict(lw=1.5, alpha=0.5)
            kw.update(stairs_kw)
            p = ax.stairs(
                c,
                bins,
                label=(
                    plot_dict["catalog_names"][i_cat]
                    if plot_dict["catalog_names"] is not None
                    else None
                ),
                color=plot_dict["colors"][i_cat],
                **kw,
            )
            ax.errorbar(
                (bins[1:] + bins[:-1]) / 2,
                c,
                yerr=(err_lo, err_hi),
                ls="none",
                color=p.get_edgecolor(),
                alpha=kw["alpha"] * 0.75,
                elinewidth=0.5,
                capsize=2.5,
                capthick=0.5,
            )

            all_counts.append(c)

        # Set x range
        nonzero_idx = np.logical_or.reduce(
            [c * a > nonzero_bin_thresh for c, a in zip(all_counts, all_areas)]
        )
        xmin, xmax = bins[:-1][nonzero_idx].min(), bins[1:][nonzero_idx].max()
        if metric == "Major Axis":
            pass
            # xmin, xmax = 3.5e0, 2.5e2
        side_gap = 0.05
        log_factor = xmax ** (side_gap) * xmin ** (-side_gap)
        ax.set_xlim(xmin / log_factor, xmax * log_factor)

        # Axis settings
        ax.set_xlabel(f"{metric} ({units[i_metric]})")
        ax.set_ylabel(
            "Count"
            if not density and areas_degsq is None
            else "Normalized Count" if density else "Count Density (sr$^{-1}$)"
        )
        ax.set_xscale("log" if logax[0] else "linear")
        ax.set_yscale("log" if logax[1] else "linear")
        ax.legend()
        ax.set_ylim(bottom=nonzero_bin_thresh / max(all_areas))
        ax.grid(alpha=0.3)

        # Add figure to dict
        figs[metric] = fig

        # Save figure
        if save:
            output_dir.mkdir(exist_ok=True)
            out_name = f"Holy_Trinity_{metric.replace(' ', '_')}"
            if len(out_prefix):
                out_name = f"{out_prefix}_{out_name}"
            fig.savefig(output_dir / f"{out_name}.pdf")

    return figs
