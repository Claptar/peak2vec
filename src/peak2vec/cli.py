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
    else:
        # Ensure we don't overwrite "outputs/run" repeatedly
        cfg.outdir = Path("outputs") / f"run_{_timestamp()}"

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

    # Persist the resolved config into the run dir
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, cfg.outdir / "config.yaml")

    console.print(f"🚀 Starting training run in [bold]{cfg.outdir}[/bold]")

    run_train(cfg, verbose=verbose)
    console.print(f"✅ Done. Outputs written to [bold]{cfg.outdir}[/bold]")


if __name__ == "__main__":
    app()
