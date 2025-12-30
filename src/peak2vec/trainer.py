from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import torch
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import anndata as ad
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track
from torch.utils.data import DataLoader

from peak2vec.dataset import PeakDataset, peak2vec_collate
from peak2vec.models.peak2vec import Peak2Vec
from peak2vec.preprocess import (
    prepare_adata,
    get_sampling_distributions,
    balanced_downsample,
)
from peak2vec.config import ExperimentConfig
from peak2vec.utils.device import resolve_device
from peak2vec.utils.io import ensure_dir, save_json, save_npy, save_yaml
from peak2vec.utils.seed import seed_everything


try:
    load_dotenv()
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None  # type: ignore


console = Console()


def _setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )


def _init_wandb(cfg: ExperimentConfig) -> Optional[wandb.sdk.wandb_run.Run]:
    """Initialize Weights & Biases logging based on the configuration."""
    if cfg.wandb.mode == "disabled":
        return None
    if wandb is None:
        raise RuntimeError(
            "wandb is not installed but wandb.mode != 'disabled'. Install wandb or disable it."
        )

    # If project not provided, fallback to env var WANDB_PROJECT (W&B does this too),
    # but it's better to set explicitly.
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        dir=cfg.outdir,
        name=cfg.wandb.name,
        tags=cfg.wandb.tags or None,
        config=cfg.to_dict(),
        mode=cfg.wandb.mode,
    )
    # Metric conventions
    # try:
    #     wandb.define_metric("train/step")
    #     wandb.define_metric("train/*", step_metric="train/step")
    #     wandb.define_metric("epoch")
    #     wandb.define_metric("epoch/*", step_metric="epoch")
    # except Exception:
    #     # define_metric isn't strictly required
    #     pass
    return run


def train(cfg: ExperimentConfig, *, verbose: bool = False) -> None:
    """Train Peak2Vec model based on the provided configuration."""
    # Setup logging
    _setup_logging(verbose=verbose)
    log = logging.getLogger(__name__)

    # Prepare output directory
    outdir = ensure_dir(cfg.outdir)
    (outdir / "checkpoints").mkdir(exist_ok=True, parents=True)

    # Reproducibility
    seed_everything(cfg.train.seed)

    # Persist config
    save_yaml(cfg.to_dict(), outdir / "config.yaml")

    # Prepare AnnData
    log.info(f"Loading and preparing AnnData: [bold]{cfg.adata_path}[/bold]")
    adata = ad.read_h5ad(cfg.adata_path)
    adata = prepare_adata(
        adata,
        chrom_col=cfg.preprocessing.chrom_col,
        start_col=cfg.preprocessing.start_col,
        end_col=cfg.preprocessing.end_col,
        center_col=cfg.preprocessing.center_col,
        peak_name_col=cfg.preprocessing.peak_name_col,
        peak_name_source=cfg.preprocessing.source,
        overwrite_coords=cfg.preprocessing.overwrite,
    )
    log.info(f"AnnData: {adata.n_obs} cells, {adata.n_vars} peaks")

    # Get sampling distributions
    log.info(
        f"Computing sampling distributions with parameters: [bold]subsample_t={cfg.sampling.subsample_t}, neg_power={cfg.sampling.neg_power}[/bold]"
    )
    neg_distribution, keep_distribution = get_sampling_distributions(
        adata,
        t=cfg.sampling.subsample_t,
        power=cfg.sampling.neg_power,
    )

    # Dataset and DataLoader
    log.info("Setting up dataset and dataloader")
    dataset = PeakDataset(
        X=adata.X,
        chr=adata.var[cfg.preprocessing.chrom_col].values,
        centers=adata.var[cfg.preprocessing.center_col].values,
        neg_distribution=neg_distribution,
        keep_distribution=keep_distribution,
        samples_per_epoch=cfg.sampling.samples_per_epoch,
        n_pairs=cfg.sampling.n_pairs,
        n_negative=cfg.sampling.n_negative,
        seed=cfg.train.seed,
        trans_fraction=cfg.sampling.trans_fraction,
        cis_window=cfg.sampling.cis_window,
        same_chr_negative_prob=cfg.sampling.same_chr_negative_prob,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        collate_fn=peak2vec_collate,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
    )

    # Model, optimizer, device
    log.info("Initializing model and optimizer")
    device = resolve_device(cfg.train.device)
    model = Peak2Vec(
        n_peaks=adata.n_vars,
        embedding_dim=cfg.train.embedding_dim,
        pos_weight=cfg.train.pos_weight,
        sparse=cfg.train.sparse,
        tie_weights=cfg.train.tie_weights,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    # Downsample training data for embedding logging (if needed)
    if cfg.wandb.mode != "disabled" and cfg.wandb.save_table:
        log.info("Preparing downsampled data for W&B embedding logging")
        downsampled_peaks = balanced_downsample(
            adata.var[cfg.preprocessing.chrom_col].to_frame().reset_index(),
            cfg.preprocessing.chrom_col,
            cfg.wandb.n_per_chromosome,
        )
        downsampled_idx = downsampled_peaks.index.tolist()
        downsampled_chr = downsampled_peaks[cfg.preprocessing.chrom_col].tolist()

    # W&B setup
    run = _init_wandb(cfg)
    if run is not None:
        log.info(f"W&B run: {run.url if hasattr(run, 'url') else '(initialized)'}")

    global_step = 0

    for epoch in track(range(1, cfg.train.epochs + 1), description="Training epochs"):
        model.train()

        running_loss = 0.0
        running_pos_loss = 0.0
        running_neg_loss = 0.0
        running_pos_score = 0.0
        running_neg_score = 0.0
        epoch_start_time = time.time()

        for step, (peaks, peak_pairs, negatives) in enumerate(loader, 1):
            peaks = peaks.to(device, non_blocking=True)
            peak_pairs = peak_pairs.to(device, non_blocking=True)
            negatives = negatives.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            batch_loss, stats = model(peaks, peak_pairs, negatives)
            batch_loss.backward()

            optimizer.step()

            running_loss += float(batch_loss.detach().cpu())
            running_pos_loss += float(stats["pos_loss_mean"].cpu())
            running_neg_loss += float(stats["neg_loss_mean"].cpu())
            running_pos_score += float(stats["pos_score_mean"].cpu())
            running_neg_score += float(stats["neg_score_mean"].cpu())

            # print(f"Epoch {epoch} | Step {step:03d} | Loss: {running / step:.4f} | Pos Loss: {stats['pos_loss_mean']:.4f} | Neg Loss: {stats['neg_loss_mean']:.4f} | Pos Score: {stats['pos_score_mean']:.4f} | Neg Score: {stats['neg_score_mean']:.4f}")

        epoch_time = time.time() - epoch_start_time
        # Log epoch stats
        if run is not None:
            run.log(
                {
                    "epoch": epoch,
                    "loss": running_loss / step,
                    "pos_loss": running_pos_loss / step,
                    "neg_loss": running_neg_loss / step,
                    "pos_score": running_pos_score / step,
                    "neg_score": running_neg_score / step,
                    "epoch_time_sec": epoch_time,
                    "samples_per_second": cfg.sampling.samples_per_epoch / epoch_time,
                }
            )

        # Checkpointing
        if epoch % cfg.train.checkpoint_every_epochs == 0:
            checkpoint_path = outdir / "checkpoints" / f"checkpoint_epoch{epoch:04d}.pt"
            log.info(f"Saving checkpoint to {checkpoint_path}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )

            if run is not None:
                artifact = wandb.Artifact(
                    f"{run.name}_checkpoint_epoch{epoch:04d}", type="model"
                )
                artifact.add_file(str(checkpoint_path))
                run.log_artifact(artifact)

        # Save table of embeddings to W&B
        if (
            epoch % cfg.train.save_embeddings_every_epochs == 0
            and cfg.wandb.save_table
            and run is not None
        ):
            embedding_norm = model.get_peak_embeddings(normalize=True).numpy()
            emb_df = pd.DataFrame(
                embedding_norm[downsampled_idx],
                columns=[f"dim_{i}" for i in range(embedding_norm.shape[1])],
            )
            emb_df["chromosome"] = downsampled_chr
            table = wandb.Table(dataframe=emb_df)

            try:
                run.log({"embedding": table})
            except FileNotFoundError as e:
                wandb.termwarn(
                    f"Skipping W&B table log at epoch {epoch} (temp dir issue): {e}"
                )
            except Exception as e:
                wandb.termwarn(
                    f"Skipping W&B table log at epoch {epoch} (unexpected): {e}"
                )

    # Save final model
    final_checkpoint = outdir / "checkpoints" / "final_model.pt"
    log.info(f"Saving final model to {final_checkpoint}")
    torch.save(
        {
            "epoch": cfg.train.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        final_checkpoint,
    )

    if run is not None:
        wandb.finish()

    log.info(f"Training complete. Outputs saved to {outdir}")
