"""Profile the PeakDataset sampling process to identify bottlenecks."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
from rich.console import Console
from rich.table import Table

from peak2vec.dataset import PeakDataset
from peak2vec.preprocess import get_sampling_distributions, prepare_adata

console = Console()


def profile_dataset_operations(
    adata_path: Path,
    n_samples: int = 1000,
    batch_size: int = 256,
) -> None:
    """Profile individual operations in the dataset sampling process."""

    console.print(f"[bold]Loading data from {adata_path}[/bold]")
    adata = ad.read_h5ad(str(adata_path))

    # Prepare data
    prepare_adata(adata, overwrite_coords=False)
    neg_distribution, keep_distribution = get_sampling_distributions(adata)

    console.print(f"Dataset: {adata.n_obs} cells, {adata.n_vars} peaks")

    # Create dataset
    dataset = PeakDataset(
        X=adata.X,
        chr=adata.var["Chromosome"].values,
        centers=adata.var["Center"].values,
        neg_distribution=neg_distribution,
        keep_distribution=keep_distribution,
        samples_per_epoch=n_samples,
        n_pairs=10,
        n_negative=20,
        seed=42,
    )

    # Profile individual operations
    timing_stats = defaultdict(list)

    console.print(f"\n[bold]Profiling {n_samples} samples...[/bold]")

    rng = np.random.Generator(np.random.PCG64(42))

    # Profile _open_cell_peaks
    console.print("Profiling _open_cell_peaks...")
    for _ in range(100):
        cell_idx = rng.integers(0, dataset.n_cells)
        start = time.perf_counter()
        peaks = dataset._open_cell_peaks(cell_idx, rng)
        timing_stats["open_cell_peaks"].append(time.perf_counter() - start)

    # Profile _sample_pair (requires open peaks)
    console.print("Profiling _sample_pair...")
    for _ in range(100):
        cell_idx = rng.integers(0, dataset.n_cells)
        open_peaks = dataset._open_cell_peaks(cell_idx, rng)
        if len(open_peaks) >= 2:
            peak_idx = rng.choice(open_peaks)
            start = time.perf_counter()
            pair_idx = dataset._sample_pair(peak_idx, open_peaks, rng)
            timing_stats["sample_pair"].append(time.perf_counter() - start)

    # Profile _sample_negatives
    console.print("Profiling _sample_negatives...")
    for _ in range(100):
        peak_idx = rng.integers(0, dataset.n_peaks)
        pair_idx = rng.integers(0, dataset.n_peaks)
        start = time.perf_counter()
        negatives = dataset._sample_negatives(peak_idx, pair_idx, rng)
        timing_stats["sample_negatives"].append(time.perf_counter() - start)

    # Profile full iteration
    console.print("Profiling full iteration...")
    start = time.perf_counter()
    samples_generated = 0
    for i, (peak, pair, negs) in enumerate(dataset):
        samples_generated += 1
        if samples_generated >= n_samples:
            break
    total_time = time.perf_counter() - start

    # Display results
    table = Table(title="Dataset Sampling Performance Profile")
    table.add_column("Operation", style="cyan", no_wrap=True)
    table.add_column("Mean Time (ms)", justify="right", style="magenta")
    table.add_column("Std Time (ms)", justify="right", style="yellow")
    table.add_column("Min Time (ms)", justify="right", style="green")
    table.add_column("Max Time (ms)", justify="right", style="red")
    table.add_column("Calls", justify="right", style="blue")

    for op_name, times in timing_stats.items():
        times_ms = np.array(times) * 1000
        table.add_row(
            op_name,
            f"{times_ms.mean():.4f}",
            f"{times_ms.std():.4f}",
            f"{times_ms.min():.4f}",
            f"{times_ms.max():.4f}",
            str(len(times)),
        )

    console.print(table)

    console.print(f"\n[bold]Full Iteration Stats:[/bold]")
    console.print(f"  Total samples: {samples_generated}")
    console.print(f"  Total time: {total_time:.2f}s")
    console.print(f"  Samples/sec: {samples_generated / total_time:.2f}")
    console.print(f"  Time per sample: {(total_time / samples_generated) * 1000:.4f}ms")

    # Estimate impact in training
    console.print(f"\n[bold]Training Impact Estimate:[/bold]")
    samples_per_epoch = 10_000
    estimated_data_time = (total_time / samples_generated) * samples_per_epoch
    console.print(
        f"  Estimated data generation time per epoch: {estimated_data_time:.2f}s"
    )
    console.print(
        f"  For batch_size={batch_size}: ~{samples_per_epoch / batch_size:.0f} batches"
    )

    # Analyze chromosome distribution
    console.print(f"\n[bold]Chromosome Distribution in Samples:[/bold]")
    chr_counts = dataset.counter
    if chr_counts:
        chr_table = Table()
        chr_table.add_column("Chromosome", style="cyan")
        chr_table.add_column("Count", justify="right", style="magenta")
        chr_table.add_column("Percentage", justify="right", style="yellow")

        total_chr_samples = sum(chr_counts.values())
        for chr_name, count in sorted(chr_counts.items()):
            pct = (count / total_chr_samples) * 100
            chr_table.add_row(chr_name, str(count), f"{pct:.2f}%")

        console.print(chr_table)


def profile_sparse_matrix_access(adata_path: Path) -> None:
    """Profile sparse matrix access patterns."""
    console.print(f"\n[bold]Profiling sparse matrix access patterns...[/bold]")

    adata = ad.read_h5ad(str(adata_path))
    X = adata.X

    console.print(f"Matrix shape: {X.shape}")
    console.print(f"Matrix format: {type(X).__name__}")
    console.print(f"Matrix density: {X.nnz / (X.shape[0] * X.shape[1]):.6f}")

    # Profile row access (CSR format)
    times = []
    for _ in range(1000):
        idx = np.random.randint(0, X.shape[0])
        start = time.perf_counter()
        row = X.getrow(idx)
        peaks = row.indices
        times.append(time.perf_counter() - start)

    times_ms = np.array(times) * 1000
    console.print(f"\nRow access (getrow + indices):")
    console.print(f"  Mean: {times_ms.mean():.4f}ms")
    console.print(f"  Std: {times_ms.std():.4f}ms")
    console.print(f"  Min: {times_ms.min():.4f}ms")
    console.print(f"  Max: {times_ms.max():.4f}ms")

    # Profile column access if CSC
    if hasattr(X, "tocsc"):
        X_csc = X.tocsc()
        times = []
        for _ in range(1000):
            idx = np.random.randint(0, X_csc.shape[1])
            start = time.perf_counter()
            col = X_csc.getcol(idx)
            cells = col.indices
            times.append(time.perf_counter() - start)

        times_ms = np.array(times) * 1000
        console.print(f"\nColumn access (getcol + indices, CSC format):")
        console.print(f"  Mean: {times_ms.mean():.4f}ms")
        console.print(f"  Std: {times_ms.std():.4f}ms")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scripts/profile_dataset.py <path_to_h5ad>")
        sys.exit(1)

    adata_path = Path(sys.argv[1])

    profile_sparse_matrix_access(adata_path)
    profile_dataset_operations(adata_path, n_samples=1000)
