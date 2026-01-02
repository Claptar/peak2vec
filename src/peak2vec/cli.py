from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
import anndata as ad
from rich.console import Console
from rich.traceback import install as install_rich_traceback

from peak2vec.config import ExperimentConfig, load_config, save_config
from peak2vec.trainer import train as run_train
from peak2vec.visualize import visualize_embeddings
from peak2vec.preprocessing import preprocess_atac

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="peak2vec experiments (ATAC peak embeddings) — notebook code turned into a CLI.",
)

console = Console()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@app.command()
def train(
    adata_h5ad: Optional[Path] = typer.Option(
        None, "--adata", exists=True, dir_okay=False, help="Prepared AnnData (.h5ad)"
    ),
    outdir: Optional[Path] = typer.Option(
        None, "--outdir", dir_okay=True, help="Output directory for this run"
    ),
    # Config file
    config: Optional[Path] = typer.Option(
        None, "--config", exists=True, dir_okay=False, help="YAML config file"
    ),
    # Preprocessing params
    chrom_col: Optional[str] = typer.Option(None, "--chrom-col"),
    start_col: Optional[str] = typer.Option(None, "--start-col"),
    end_col: Optional[str] = typer.Option(None, "--end-col"),
    center_col: Optional[str] = typer.Option(None, "--center-col"),
    peak_name_col: Optional[str] = typer.Option(None, "--peak-name-col"),
    peak_source: Optional[str] = typer.Option(None, "--peak-source"),
    overwrite_coords: Optional[bool] = typer.Option(None, "--overwrite-coords"),
    # Sampling hyperparams
    samples_per_epoch: Optional[int] = typer.Option(None, "--samples-per-epoch"),
    n_pairs: Optional[int] = typer.Option(None, "--n-pairs"),
    n_negative: Optional[int] = typer.Option(None, "--n-negative"),
    trans_fraction: Optional[float] = typer.Option(None, "--trans-fraction"),
    cis_window: Optional[int] = typer.Option(None, "--cis-window"),
    same_chr_negative_prob: Optional[float] = typer.Option(
        None, "--same-chr-negative-prob"
    ),
    subsample_t: Optional[float] = typer.Option(None, "--subsample-t"),
    neg_power: Optional[float] = typer.Option(None, "--neg-power"),
    # Train hyperparams
    embedding_dim: Optional[int] = typer.Option(None, "--embedding-dim"),
    pos_weight: Optional[float] = typer.Option(None, "--pos-weight"),
    sparse: Optional[bool] = typer.Option(None, "--sparse"),
    tie_weights: Optional[bool] = typer.Option(None, "--tie-weights"),
    lr: Optional[float] = typer.Option(None, "--lr"),
    weight_decay: Optional[float] = typer.Option(None, "--weight-decay"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size"),
    epochs: Optional[int] = typer.Option(None, "--epochs"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    device: Optional[str] = typer.Option(None, "--device", help="auto|cpu|cuda|mps"),
    num_workers: Optional[int] = typer.Option(None, "--num-workers"),
    pin_memory: Optional[bool] = typer.Option(None, "--pin-memory"),
    checkpoint_every_epochs: Optional[int] = typer.Option(
        None, "--checkpoint-every-epochs"
    ),
    save_embeddings_every_epochs: Optional[int] = typer.Option(
        None, "--save-embeddings-every-epochs"
    ),
    # W&B
    wandb_project: Optional[str] = typer.Option(None, "--wandb-project"),
    wandb_entity: Optional[str] = typer.Option(None, "--wandb-entity"),
    wandb_group: Optional[str] = typer.Option(None, "--wandb-group"),
    wandb_name: Optional[str] = typer.Option(None, "--wandb-name"),
    wandb_mode: Optional[str] = typer.Option(
        None, "--wandb-mode", help="online|offline|disabled"
    ),
    wandb_n_per_chromosome: Optional[int] = typer.Option(
        None, "--wandb-n-per-chromosome"
    ),
    wandb_save_table: Optional[bool] = typer.Option(None, "--wandb-save-table"),
    wandb_visualize_embeddings: Optional[bool] = typer.Option(
        None, "--wandb-visualize-embeddings"
    ),
    wandb_viz_metadata_cols: Optional[str] = typer.Option(
        None,
        "--wandb-viz-metadata-cols",
        help="Comma-separated list of adata.var columns for visualization",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose logging"),
):
    """Train peak2vec and optionally log to Weights & Biases."""
    install_rich_traceback()

    cfg = load_config(config) if config else ExperimentConfig()

    # Apply CLI overrides
    if adata_h5ad is not None:
        cfg.adata_path = adata_h5ad
    if outdir is not None:
        cfg.outdir = outdir

    # Preprocessing
    cfg.preprocessing.chrom_col = (
        chrom_col if chrom_col is not None else cfg.preprocessing.chrom_col
    )
    cfg.preprocessing.start_col = (
        start_col if start_col is not None else cfg.preprocessing.start_col
    )
    cfg.preprocessing.end_col = (
        end_col if end_col is not None else cfg.preprocessing.end_col
    )
    cfg.preprocessing.center_col = (
        center_col if center_col is not None else cfg.preprocessing.center_col
    )
    cfg.preprocessing.peak_name_col = (
        peak_name_col if peak_name_col is not None else cfg.preprocessing.peak_name_col
    )
    cfg.preprocessing.source = (
        peak_source if peak_source is not None else cfg.preprocessing.source
    )
    cfg.preprocessing.overwrite = (
        overwrite_coords
        if overwrite_coords is not None
        else cfg.preprocessing.overwrite
    )

    # Sampling
    cfg.sampling.samples_per_epoch = (
        samples_per_epoch
        if samples_per_epoch is not None
        else cfg.sampling.samples_per_epoch
    )
    cfg.sampling.n_pairs = n_pairs if n_pairs is not None else cfg.sampling.n_pairs
    cfg.sampling.n_negative = (
        n_negative if n_negative is not None else cfg.sampling.n_negative
    )
    cfg.sampling.trans_fraction = (
        trans_fraction if trans_fraction is not None else cfg.sampling.trans_fraction
    )
    cfg.sampling.cis_window = (
        cis_window if cis_window is not None else cfg.sampling.cis_window
    )
    cfg.sampling.same_chr_negative_prob = (
        same_chr_negative_prob
        if same_chr_negative_prob is not None
        else cfg.sampling.same_chr_negative_prob
    )
    cfg.sampling.subsample_t = (
        subsample_t if subsample_t is not None else cfg.sampling.subsample_t
    )
    cfg.sampling.neg_power = (
        neg_power if neg_power is not None else cfg.sampling.neg_power
    )

    # Train
    cfg.train.embedding_dim = (
        embedding_dim if embedding_dim is not None else cfg.train.embedding_dim
    )
    cfg.train.pos_weight = (
        pos_weight if pos_weight is not None else cfg.train.pos_weight
    )
    cfg.train.sparse = sparse if sparse is not None else cfg.train.sparse
    cfg.train.tie_weights = (
        tie_weights if tie_weights is not None else cfg.train.tie_weights
    )
    cfg.train.lr = lr if lr is not None else cfg.train.lr
    cfg.train.weight_decay = (
        weight_decay if weight_decay is not None else cfg.train.weight_decay
    )
    cfg.train.batch_size = (
        batch_size if batch_size is not None else cfg.train.batch_size
    )
    cfg.train.epochs = epochs if epochs is not None else cfg.train.epochs
    cfg.train.seed = seed if seed is not None else cfg.train.seed
    cfg.train.device = device if device is not None else cfg.train.device
    cfg.train.num_workers = (
        num_workers if num_workers is not None else cfg.train.num_workers
    )
    cfg.train.pin_memory = (
        pin_memory if pin_memory is not None else cfg.train.pin_memory
    )
    cfg.train.checkpoint_every_epochs = (
        checkpoint_every_epochs
        if checkpoint_every_epochs is not None
        else cfg.train.checkpoint_every_epochs
    )
    cfg.train.save_embeddings_every_epochs = (
        save_embeddings_every_epochs
        if save_embeddings_every_epochs is not None
        else cfg.train.save_embeddings_every_epochs
    )

    # Wandb
    cfg.wandb.project = (
        wandb_project if wandb_project is not None else cfg.wandb.project
    )
    cfg.wandb.entity = wandb_entity if wandb_entity is not None else cfg.wandb.entity
    cfg.wandb.group = wandb_group if wandb_group is not None else cfg.wandb.group
    cfg.wandb.name = wandb_name if wandb_name is not None else cfg.wandb.name
    cfg.wandb.mode = wandb_mode if wandb_mode is not None else cfg.wandb.mode
    cfg.wandb.n_per_chromosome = (
        wandb_n_per_chromosome
        if wandb_n_per_chromosome is not None
        else cfg.wandb.n_per_chromosome
    )
    cfg.wandb.save_table = (
        wandb_save_table if wandb_save_table is not None else cfg.wandb.save_table
    )
    cfg.wandb.visualize_embeddings = (
        wandb_visualize_embeddings
        if wandb_visualize_embeddings is not None
        else cfg.wandb.visualize_embeddings
    )
    if wandb_viz_metadata_cols is not None:
        cfg.wandb.viz_metadata_cols = wandb_viz_metadata_cols.split(",")

    # Persist the resolved config into the run dir
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, cfg.outdir / "config.yaml")

    console.print(f"🚀 Starting training run in [bold]{cfg.outdir}[/bold]")

    run_train(cfg, verbose=verbose)
    console.print(f"✅ Done. Outputs written to [bold]{cfg.outdir}[/bold]")


@app.command()
def visualize(
    checkpoint: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Model checkpoint file"
    ),
    metadata: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Peak metadata CSV file"
    ),
    outdir: Optional[Path] = typer.Option(
        None, "--outdir", dir_okay=True, help="Output directory for visualizations"
    ),
    index_col: Optional[str] = typer.Option(
        None, "--index-col", help="Column to use as index when loading metadata CSV"
    ),
    n_pcs: Optional[int] = typer.Option(
        None, "--n-pcs", help="Number of principal components for PCA"
    ),
    n_neighbors: int = typer.Option(
        15, "--n-neighbors", help="Number of neighbors for UMAP"
    ),
    metric: str = typer.Option(
        "cosine", "--metric", help="Distance metric for UMAP (e.g., euclidean, cosine)"
    ),
    random_state: int = typer.Option(
        4, "--random-state", help="Random state for UMAP reproducibility"
    ),
    which: str = typer.Option(
        "in", "--which", help="Which embeddings to visualize: in|out"
    ),
) -> None:
    """
    Visualize peak embeddings from a trained model checkpoint.
    """
    install_rich_traceback()

    # Determine output directory
    if outdir is None:
        outdir = checkpoint.parent.parent / "visualizations"
    outdir.mkdir(parents=True, exist_ok=True)

    console.print(f"🎨 Creating visualizations in [bold]{outdir}[/bold]")

    visualize_embeddings(
        checkpoint_path=checkpoint,
        metadata_path=metadata,
        outdir=outdir,
        index_col=index_col,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=random_state,
        which=which,
    )

    console.print(f"✅ Visualizations saved to [bold]{outdir}[/bold]")


@app.command()
def preprocess(
    input_h5ad: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input AnnData (.h5ad) file"
    ),
    output_h5ad: Optional[Path] = typer.Argument(
        None,
        dir_okay=False,
        help="Output AnnData (.h5ad) file (not required with --qc-only)",
    ),
    min_cells_per_peak: Optional[int] = typer.Option(
        70, "--min-cells-per-peak", help="Minimum cells per peak"
    ),
    max_cells_per_peak: Optional[int] = typer.Option(
        9000, "--max-cells-per-peak", help="Maximum cells per peak"
    ),
    min_peaks_per_cell: Optional[int] = typer.Option(
        500, "--min-peaks-per-cell", help="Minimum peaks per cell"
    ),
    max_peaks_per_cell: Optional[int] = typer.Option(
        25000, "--max-peaks-per-cell", help="Maximum peaks per cell"
    ),
    max_nucleosome_signal: Optional[float] = typer.Option(
        None, "--max-nucleosome-signal", help="Maximum nucleosome signal (if available)"
    ),
    chrom_col: str = typer.Option(
        "Chromosome", "--chrom-col", help="Column name for chromosome"
    ),
    start_col: str = typer.Option("Start", "--start-col", help="Column name for start"),
    end_col: str = typer.Option("End", "--end-col", help="Column name for end"),
    center_col: str = typer.Option(
        "Center", "--center-col", help="Column name for center"
    ),
    peak_source: str = typer.Option(
        "var_names", "--peak-source", help="Source of peak names: var_names|column"
    ),
    peak_name_col: Optional[str] = typer.Option(
        None,
        "--peak-name-col",
        help="Column name containing peak names (if source=column)",
    ),
    overwrite_coords: bool = typer.Option(
        False, "--overwrite-coords", help="Overwrite existing coordinates"
    ),
    plot_qc: bool = typer.Option(
        True, "--plot-qc/--no-plot-qc", help="Generate QC plots"
    ),
    qc_only: bool = typer.Option(
        False, "--qc-only", help="Only generate QC plots without filtering"
    ),
    qc_var_columns: Optional[str] = typer.Option(
        None,
        "--qc-var-columns",
        help="Comma-separated list of peak columns to plot (max 2)",
    ),
    qc_obs_columns: Optional[str] = typer.Option(
        None,
        "--qc-obs-columns",
        help="Comma-separated list of cell columns to plot (max 6)",
    ),
    qc_plots_dir: Optional[Path] = typer.Option(
        None,
        "--qc-plots-dir",
        help="Directory to save QC plots (default: same as output with 'qc_plots' subdirectory)",
    ),
) -> None:
    """
    Preprocess ATAC-seq data: calculate QC metrics, filter cells/peaks, and prepare for training.
    """
    install_rich_traceback()

    # Validate output_h5ad requirement
    if not qc_only and output_h5ad is None:
        console.print(
            "[red]Error: output_h5ad is required unless --qc-only is specified[/red]"
        )
        raise typer.Exit(1)

    console.print(f"📊 Preprocessing [bold]{input_h5ad}[/bold]")

    # Parse QC columns
    parsed_var_cols = qc_var_columns.split(",") if qc_var_columns else None
    parsed_obs_cols = qc_obs_columns.split(",") if qc_obs_columns else None

    preprocess_atac(
        input_h5ad=input_h5ad,
        output_h5ad=output_h5ad,
        min_cells_per_peak=min_cells_per_peak,
        max_cells_per_peak=max_cells_per_peak,
        min_peaks_per_cell=min_peaks_per_cell,
        max_peaks_per_cell=max_peaks_per_cell,
        max_nucleosome_signal=max_nucleosome_signal,
        chrom_col=chrom_col,
        start_col=start_col,
        end_col=end_col,
        center_col=center_col,
        peak_source=peak_source,
        peak_name_col=peak_name_col,
        overwrite_coords=overwrite_coords,
        plot_qc=plot_qc,
        qc_only=qc_only,
        qc_var_columns=parsed_var_cols,
        qc_obs_columns=parsed_obs_cols,
        qc_plots_dir=qc_plots_dir,
    )

    if not qc_only and output_h5ad:
        console.print(f"💾 Saved to [bold]{output_h5ad}[/bold]")


if __name__ == "__main__":
    app()
