from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

_PEAK_RE_COLON = re.compile(r"^(?P<chr>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")
_PEAK_RE_UNDERSCORE = re.compile(r"^(?P<chr>[^_]+)_(?P<start>\d+)_(?P<end>\d+)$")


def ensure_csc(adata: AnnData) -> None:
    """Ensure adata.X is CSC (required by PeakDataset for fast row slicing)."""
    if sp.issparse(adata.X) and not sp.isspmatrix_csc(adata.X):
        adata.X = adata.X.tocsc()
    elif not sp.issparse(adata.X):
        # Dense is supported but not recommended at ATAC scale
        adata.X = sp.csc_matrix(adata.X)


def _parse_peak_name(name: str) -> Optional[Tuple[str, int, int]]:
    """Parse common peak name formats into (chrom, start, end)."""
    m = _PEAK_RE_COLON.match(name)
    if m:
        return m.group("chr"), int(m.group("start")), int(m.group("end"))
    m = _PEAK_RE_UNDERSCORE.match(name)
    if m:
        return m.group("chr"), int(m.group("start")), int(m.group("end"))
    return None


def add_peak_coordinates(
    adata: AnnData,
    *,
    chrom_col: str = "Chromosome",
    start_col: str = "Start",
    end_col: str = "End",
    center_col: str = "Center",
    source: str = "var_names",
    peak_name_col: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """Add genomic coordinates to `adata.var` if missing.

    This is required by the current PeakDataset sampling logic.
    By default we parse `adata.var_names` (common formats: chr1:1-10 or chr1_1_10).

    Parameters
    ----------
    source:
        - "var_names": parse `adata.var_names`
        - "column": parse `adata.var[peak_name_col]`
    """
    have_all = all(c in adata.var.columns for c in (chrom_col, start_col, end_col))
    if have_all and not overwrite and center_col not in adata.var.columns:
        adata.var[center_col] = (
            (adata.var[start_col].values + adata.var[end_col].values) // 2
        ).astype(np.int64)
        return
    elif have_all and not overwrite:
        return

    if source == "column":
        if not peak_name_col:
            raise ValueError("peak_name_col must be provided when source='column'")
        names = adata.var[peak_name_col].astype(str).tolist()
    elif source == "var_names":
        names = [str(x) for x in adata.var_names]
    else:
        raise ValueError("source must be one of: 'var_names', 'column'")

    chroms: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    failed: list[str] = []

    for n in names:
        parsed = _parse_peak_name(n)
        if parsed is None:
            failed.append(n)
            chroms.append("NA")
            starts.append(-1)
            ends.append(-1)
        else:
            c, s, e = parsed
            chroms.append(c)
            starts.append(s)
            ends.append(e)

    if failed:
        preview = ", ".join(failed[:5])
        raise ValueError(
            f"Failed to parse {len(failed)}/{len(names)} peak names. "
            f"Examples: {preview}. "
            "Expected formats like 'chr1:123-456' or 'chr1_123_456'."
        )

    adata.var[chrom_col] = np.asarray(chroms, dtype=object)
    adata.var[start_col] = np.asarray(starts, dtype=np.int64)
    adata.var[end_col] = np.asarray(ends, dtype=np.int64)
    adata.var[center_col] = (
        (adata.var[start_col].values + adata.var[end_col].values) // 2
    ).astype(np.int64)


def get_sampling_distributions(
    adata: AnnData, t: float = 5e-7, power: float = 0.75, eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get negative sampling and subsampling distributions for PeakDataset.

    adata (AnnData): Annotated data matrix with shape (n_cells, n_peaks).
    t (float):       Subsampling threshold. Default is 5e-7.
    power (float):   Exponent for negative sampling distribution. Default is 0.75.
    eps (float):     Small constant to avoid division by zero. Default is 1e-12.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing:
        - neg_distribution: Negative sampling distribution (global).
        - keep_distribution: Subsampling distribution.
    """
    peak_counts = np.asarray((adata.X > 0).sum(axis=0)).squeeze()
    # Negative sampling distribution
    f = (peak_counts.astype(np.float64) + eps) ** power
    neg_distribution = f / f.sum()

    # Subsampling distribution
    keep_distribution = np.minimum(1.0, (t / (peak_counts + eps)))

    return neg_distribution, keep_distribution


def prepare_adata(
    adata: AnnData,
    *,
    chrom_col: str = "Chromosome",
    start_col: str = "Start",
    end_col: str = "End",
    peak_name_source: str = "var_names",
    peak_name_col: Optional[str] = None,
    subsample_t: float = 5e-7,
    neg_power: float = 0.75,
    overwrite_coords: bool = False,
) -> None:
    """
    Prepare AnnData for PeakDataset usage. This includes:
     - Ensuring adata.X is in CSC format.
     - Adding peak coordinates to adata.var if missing.
     - Parsing peak names from var_names or a specified column.
    """
    ensure_csc(adata)
    add_peak_coordinates(
        adata,
        chrom_col=chrom_col,
        start_col=start_col,
        end_col=end_col,
        source=peak_name_source,
        peak_name_col=peak_name_col,
        overwrite=overwrite_coords,
    )
