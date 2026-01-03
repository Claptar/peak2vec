from pathlib import Path
from typing import Union, Optional
import logging

import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from peak2vec.models.peak2vec import Peak2Vec

console = Console()


def visualize_embeddings(
    checkpoint_path: Union[Path, Peak2Vec],
    outdir: Path,
    metadata: Optional[Union[Path, pd.DataFrame]] = None,
    index_col: Optional[str] = None,
    n_pcs: Optional[int] = None,
    n_neighbors: int = 15,
    metric: str = "cosine",
    random_state: int = 4,
    which: str = "in",
    show_progress: bool = True,
    log: Optional[logging.Logger] = None,
) -> Path:
    """
    Load checkpoint, extract embeddings, and create visualizations.

    Args:
        checkpoint_path: Path to checkpoint file or Peak2Vec model instance
        outdir: Output directory for visualizations
        metadata: Path to CSV file, DataFrame with peak metadata, or None for empty metadata
        index_col: Column to use as index when loading CSV (only used if metadata is a Path)
        n_pcs: Number of PCs for PCA (None = auto-determined by scanpy)
        n_neighbors: Number of neighbors for UMAP
        metric: Distance metric for UMAP
        random_state: Random seed for UMAP
        which: Which embeddings to visualize ("in" or "out")
        show_progress: Whether to show progress bar
        log: Optional logger instance

    Returns:
        Path to saved h5ad file
    """
    # Show progress if explicitly requested OR if a logger is provided
    should_show_progress = show_progress or (log is not None)

    progress = (
        Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            console=console,
        )
        if should_show_progress
        else None
    )

    def _log(msg: str):
        if log:
            log.info(msg)
        elif should_show_progress and not log:
            console.print(msg)

    def _add_task(desc: str):
        return progress.add_task(desc, total=None) if progress else None

    def _remove_task(task):
        if progress and task is not None:
            progress.remove_task(task)

    ctx = progress if progress else DummyContext()

    with ctx:
        # Load checkpoint or use provided model
        task = _add_task("[cyan]Loading checkpoint...")
        if isinstance(checkpoint_path, Peak2Vec):
            model = checkpoint_path
            n_peaks = model.in_embedding.weight.shape[0]
            embedding_dim = model.in_embedding.weight.shape[1]
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            n_peaks = checkpoint["model_state_dict"]["in_embedding.weight"].shape[0]
            embedding_dim = checkpoint["model_state_dict"]["in_embedding.weight"].shape[
                1
            ]
            model = Peak2Vec(
                n_peaks=n_peaks,
                embedding_dim=embedding_dim,
                pos_weight=1.0,
                sparse=False,
                tie_weights=False,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
        _remove_task(task)

        # Load metadata
        task = _add_task("[cyan]Loading metadata...")
        if metadata is None:
            metadata_df = pd.DataFrame(index=range(n_peaks))
        elif isinstance(metadata, pd.DataFrame):
            metadata_df = metadata
        else:
            metadata_df = pd.read_csv(metadata, index_col=index_col)
        _remove_task(task)

        # Extract embeddings from model
        task = _add_task("[cyan]Extracting embeddings...")
        embeddings = model.get_peak_embeddings(normalize=True, which=which).numpy()
        _remove_task(task)

        # Create AnnData
        task = _add_task("[cyan]Creating AnnData object...")
        emb_adata = ad.AnnData(embeddings)
        emb_adata.obs = metadata_df.copy()
        _remove_task(task)

        # Calculate PCA
        task = _add_task("[cyan]Computing PCA...")
        sc.pp.pca(emb_adata, n_comps=n_pcs)
        _remove_task(task)

        # Calculate neighbors
        task = _add_task("[cyan]Computing neighbors...")
        sc.pp.neighbors(emb_adata, n_neighbors=n_neighbors, metric=metric)
        _remove_task(task)

        # Calculate UMAP
        task = _add_task("[cyan]Computing UMAP...")
        sc.tl.umap(emb_adata, random_state=random_state)
        _remove_task(task)

        # Plot PCA
        task = _add_task("[cyan]Generating PCA plot...")
        color_cols = [
            col for col in metadata_df.columns if col in emb_adata.obs.columns
        ]
        if color_cols:
            pca_fig = sc.pl.pca(
                emb_adata,
                color=color_cols,
                return_fig=True,
                show=False,
            )
            pca_path = outdir / f"peak_embeddings_{which}_pca.png"
            pca_fig.savefig(pca_path, bbox_inches="tight", dpi=300)
            _log(f"Saved PCA plot to {pca_path}")
        _remove_task(task)

        # Plot UMAP
        task = _add_task("[cyan]Generating UMAP plot...")
        if color_cols:
            umap_fig = sc.pl.umap(
                emb_adata,
                color=color_cols,
                return_fig=True,
                show=False,
            )
            umap_path = outdir / f"peak_embeddings_{which}_umap.png"
            umap_fig.savefig(umap_path, bbox_inches="tight", dpi=300)
            _log(f"Saved UMAP plot to {umap_path}")
        _remove_task(task)

        # Save processed AnnData
        task = _add_task("[cyan]Saving processed data...")
        h5ad_path = outdir / f"peak_embeddings_{which}.h5ad"
        emb_adata.write_h5ad(h5ad_path)
        _remove_task(task)

        return h5ad_path


class DummyContext:
    """Dummy context manager for when progress is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
