from pathlib import Path

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
    checkpoint_path: Path,
    metadata_path: Path,
    outdir: Path,
    index_col: str | None = None,
    n_pcs: int | None = None,
    n_neighbors: int = 15,
    metric: str = "cosine",
    random_state: int = 4,
    which: str = "in",
) -> None:
    """
    Load checkpoint, extract embeddings, and create visualizations.
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=console,
    )

    with progress:
        # Load checkpoint
        task = progress.add_task("[cyan]Loading checkpoint...", total=None)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        progress.remove_task(task)

        # Load metadata
        task = progress.add_task("[cyan]Loading metadata...", total=None)
        metadata = pd.read_csv(metadata_path, index_col=index_col)
        progress.remove_task(task)

        # Extract embeddings from model
        task = progress.add_task("[cyan]Extracting embeddings...", total=None)
        n_peaks = checkpoint["model_state_dict"]["in_embedding.weight"].shape[0]
        embedding_dim = checkpoint["model_state_dict"]["in_embedding.weight"].shape[1]

        model = Peak2Vec(
            n_peaks=n_peaks,
            embedding_dim=embedding_dim,
            pos_weight=1.0,
            sparse=False,
            tie_weights=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        embeddings = model.get_peak_embeddings(normalize=True, which=which).numpy()
        progress.remove_task(task)

        # Create AnnData
        task = progress.add_task("[cyan]Creating AnnData object...", total=None)
        emb_adata = ad.AnnData(embeddings)
        emb_adata.obs = metadata.copy()
        progress.remove_task(task)

        # Calculate PCA
        task = progress.add_task("[cyan]Computing PCA...", total=None)
        sc.pp.pca(emb_adata, n_comps=n_pcs)
        progress.remove_task(task)

        # Calculate neighbors
        task = progress.add_task("[cyan]Computing neighbors...", total=None)
        sc.pp.neighbors(emb_adata, n_neighbors=n_neighbors, metric=metric)
        progress.remove_task(task)

        # Calculate UMAP
        task = progress.add_task("[cyan]Computing UMAP...", total=None)
        sc.tl.umap(emb_adata, random_state=random_state)
        progress.remove_task(task)

        # Plot PCA
        task = progress.add_task("[cyan]Generating PCA plot...", total=None)
        pca_fig = sc.pl.pca(
            emb_adata,
            color=metadata.columns.tolist(),
            return_fig=True,
            show=False,
        )
        pca_fig.savefig(
            outdir / f"peak_embeddings_{which}_pca.png",
            bbox_inches="tight",
            dpi=300,
        )
        progress.remove_task(task)

        # Plot UMAP
        task = progress.add_task("[cyan]Generating UMAP plot...", total=None)
        umap_fig = sc.pl.umap(
            emb_adata,
            color=metadata.columns.tolist(),
            return_fig=True,
            show=False,
        )
        umap_fig.savefig(
            outdir / f"peak_embeddings_{which}_umap.png",
            bbox_inches="tight",
            dpi=300,
        )
        progress.remove_task(task)

        # Save processed AnnData
        task = progress.add_task("[cyan]Saving processed data...", total=None)
        emb_adata.write_h5ad(outdir / f"peak_embeddings_{which}.h5ad")
        progress.remove_task(task)
