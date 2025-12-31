from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import muon as mu
import anndata as ad
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from peak2vec.preprocess import prepare_adata

console = Console()


def plot_qc_histograms(
    adata: ad.AnnData,
    outdir: Path,
    var_columns: Optional[list[str]] = None,
    obs_columns: Optional[list[str]] = None,
    binrange_cells: tuple[int, int] = (0, 2000),
    binrange_fragments: tuple[int, int] = (0, 150000),
    suffix: str = "",
) -> None:
    """
    Plot QC histograms for peaks and cells.

    Parameters
    ----------
    var_columns : List of peak/feature columns to plot (max 2)
    obs_columns : List of cell/observation columns to plot (max 6)
    suffix : Suffix to add to output filenames (e.g., '_raw', '_filtered')
    """
    sns.set_theme(style="white")

    # Default columns if not specified
    if var_columns is None:
        var_columns = []
        if "cisTopic_log_nr_frag" in adata.var.columns:
            var_columns.append("cisTopic_log_nr_frag")
        if "n_cells_per_feature" in adata.var.columns:
            var_columns.append("n_cells_per_feature")

    # Filter to only existing columns
    var_columns = [col for col in var_columns if col in adata.var.columns]

    # Peak statistics
    if var_columns:
        n_plots = min(len(var_columns), 2)
        fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        for idx, col in enumerate(var_columns[:2]):
            ax = axes[idx]
            binrange = binrange_cells if "cells" in col.lower() else None
            sns.histplot(adata.var, x=col, ax=ax, binrange=binrange)
            median = adata.var[col].median()
            ax.axvline(
                x=median, color="black", linestyle="--", label=f"median: {median:.1f}"
            )
            ax.set_xlabel(col)
            ax.set_title(f"Peak: {col}")
            ax.legend()

        plt.tight_layout()
        fig.savefig(outdir / f"qc_peaks{suffix}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Default columns if not specified
    if obs_columns is None:
        obs_columns = []
        default_cols = [
            "cisTopic_log_nr_frag",
            "n_features_per_cell",
            "total_fragment_counts",
            "unique_fragments_count",
            "nucleosome_signal",
            "tss_enrichment",
        ]
        obs_columns = [col for col in default_cols if col in adata.obs.columns]

    # Filter to only existing columns
    obs_columns = [col for col in obs_columns if col in adata.obs.columns]

    # Cell statistics
    if obs_columns:
        n_plots = min(len(obs_columns), 6)
        n_rows = (n_plots + 2) // 3  # Ceiling division
        n_cols = min(n_plots, 3)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flat

        for idx, col in enumerate(obs_columns[:6]):
            ax = axes[idx]

            # Determine binrange based on column name
            binrange = None
            if "fragment" in col.lower() and "total" in col.lower():
                binrange = binrange_fragments
            elif "fragment" in col.lower() and "unique" in col.lower():
                binrange = (0, 100000)
            elif "nucleosome" in col.lower():
                binrange = (0, 2)

            sns.histplot(adata.obs, x=col, ax=ax, binrange=binrange)
            median = adata.obs[col].median()
            ax.axvline(
                x=median, color="black", linestyle="--", label=f"median: {median:.1f}"
            )
            ax.set_xlabel(col)
            ax.set_title(f"Cell: {col}")
            ax.legend()

        # Hide empty subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        fig.savefig(outdir / f"qc_cells{suffix}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def preprocess_atac(
    input_h5ad: Path,
    output_h5ad: Optional[Path],
    *,
    min_cells_per_peak: Optional[int] = 70,
    max_cells_per_peak: Optional[int] = 9000,
    min_peaks_per_cell: Optional[int] = 500,
    max_peaks_per_cell: Optional[int] = 25000,
    max_nucleosome_signal: Optional[float] = None,
    chrom_col: str = "Chromosome",
    start_col: str = "Start",
    end_col: str = "End",
    center_col: str = "Center",
    peak_source: str = "var_names",
    peak_name_col: Optional[str] = None,
    overwrite_coords: bool = False,
    plot_qc: bool = True,
    qc_only: bool = False,
    qc_var_columns: Optional[list[str]] = None,
    qc_obs_columns: Optional[list[str]] = None,
    qc_plots_dir: Optional[Path] = None,
) -> ad.AnnData:
    """
    Preprocess ATAC-seq data with QC metrics, filtering, and plotting.

    Returns the preprocessed AnnData object.
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=console,
    )

    with progress:
        # Load data
        task = progress.add_task("[cyan]Loading data...", total=None)
        adata = ad.read_h5ad(str(input_h5ad))
        console.print(f"   Initial: {adata.n_obs} cells × {adata.n_vars} peaks")
        progress.remove_task(task)

        # Calculate QC metrics
        task = progress.add_task("[cyan]Calculating QC metrics...", total=None)
        import scanpy as sc

        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

        # Rename columns
        adata.obs.rename(
            columns={
                "n_genes_by_counts": "n_features_per_cell",
                "total_counts": "total_fragment_counts",
            },
            inplace=True,
        )
        adata.var.rename(
            columns={
                "n_cells_by_counts": "n_cells_per_feature",
            },
            inplace=True,
        )
        progress.remove_task(task)

        # Determine QC plots directory
        default_qc_dir = (
            output_h5ad.parent / "qc" if output_h5ad else input_h5ad.parent / "qc"
        )
        qc_dir = qc_plots_dir if qc_plots_dir else default_qc_dir

        # Plot QC before filtering
        if plot_qc:
            task = progress.add_task(
                (
                    "[cyan]Plotting QC metrics (before filtering)..."
                    if not qc_only
                    else "[cyan]Plotting QC metrics..."
                ),
                total=None,
            )
            qc_dir.mkdir(parents=True, exist_ok=True)
            plot_qc_histograms(
                adata,
                qc_dir,
                var_columns=qc_var_columns,
                obs_columns=qc_obs_columns,
                suffix="_raw" if not qc_only else "",
            )
            progress.remove_task(task)

        # Skip filtering if qc_only mode
        if qc_only:
            console.print("   QC-only mode: skipping filtering")

        # Filter peaks
        if not qc_only and (
            min_cells_per_peak is not None or max_cells_per_peak is not None
        ):
            task = progress.add_task(
                f"[cyan]Filtering peaks (cells per peak: {min_cells_per_peak}-{max_cells_per_peak})...",
                total=None,
            )
            mu.pp.filter_var(
                adata,
                "n_cells_per_feature",
                lambda x: (
                    (x >= (min_cells_per_peak or 0))
                    & (x <= (max_cells_per_peak or float("inf")))
                ),
            )
            console.print(
                f"   After peak filtering: {adata.n_obs} cells × {adata.n_vars} peaks"
            )
            progress.remove_task(task)

        # Filter cells
        if not qc_only and (
            min_peaks_per_cell is not None or max_peaks_per_cell is not None
        ):
            task = progress.add_task(
                f"[cyan]Filtering cells (peaks per cell: {min_peaks_per_cell}-{max_peaks_per_cell})...",
                total=None,
            )
            mu.pp.filter_obs(
                adata,
                "n_features_per_cell",
                lambda x: (
                    (x >= (min_peaks_per_cell or 0))
                    & (x <= (max_peaks_per_cell or float("inf")))
                ),
            )
            console.print(
                f"   After cell filtering: {adata.n_obs} cells × {adata.n_vars} peaks"
            )
            progress.remove_task(task)

        # Filter by nucleosome signal
        if (
            not qc_only
            and max_nucleosome_signal is not None
            and "nucleosome_signal" in adata.obs.columns
        ):
            task = progress.add_task(
                f"[cyan]Filtering by nucleosome signal (<= {max_nucleosome_signal})...",
                total=None,
            )
            mu.pp.filter_obs(
                adata, "nucleosome_signal", lambda x: x <= max_nucleosome_signal
            )
            console.print(
                f"   After nucleosome filtering: {adata.n_obs} cells × {adata.n_vars} peaks"
            )
            progress.remove_task(task)

        # Plot QC after filtering
        if plot_qc and not qc_only:
            task = progress.add_task(
                "[cyan]Plotting QC metrics (after filtering)...", total=None
            )
            plot_qc_histograms(
                adata,
                qc_dir,
                var_columns=qc_var_columns,
                obs_columns=qc_obs_columns,
                suffix="_filtered",
            )
            progress.remove_task(task)

        if plot_qc:
            console.print(f"   QC plots saved to [bold]{qc_dir}[/bold]")

        # Prepare data for training (skip if qc_only)
        if not qc_only:
            task = progress.add_task(
                "[cyan]Preparing data (CSC format, peak coordinates)...", total=None
            )
            prepare_adata(
                adata,
                chrom_col=chrom_col,
                start_col=start_col,
                end_col=end_col,
                center_col=center_col,
                peak_name_source=peak_source,
                peak_name_col=peak_name_col,
                overwrite_coords=overwrite_coords,
            )
            progress.remove_task(task)

        # Save (skip if qc_only)
        if not qc_only and output_h5ad:
            task = progress.add_task("[cyan]Saving preprocessed data...", total=None)
            output_h5ad.parent.mkdir(parents=True, exist_ok=True)
            adata.write_h5ad(str(output_h5ad))
            progress.remove_task(task)

    if qc_only:
        console.print("✅ QC plots generated!")
    else:
        console.print(
            f"✅ Preprocessing complete! Final: {adata.n_obs} cells × {adata.n_vars} peaks"
        )
