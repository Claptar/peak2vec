from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterator, Tuple, List, Union

import numpy as np
import torch
from anndata import AnnData
from scipy.sparse import csc_matrix, csr_matrix, issparse
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


class PeakDataset(IterableDataset[Tuple[int, int, torch.Tensor]]):
    """Iterable dataset yielding (anchor_peak, context_peak, negatives).

    Expects `X` as sparse matrix with shape (n_cells, n_peaks).
    Will convert to CSR format if needed for fast row access.

    Expects peak metadata containing:
      - Chromosome, Start, End
      - cisTopic_nr_acc
      - neg: negative-sampling probabilities (global)
      - keep: subsampling probabilities
    """

    def __init__(
        self,
        X: Union[csr_matrix, csc_matrix],
        chr: np.ndarray,
        centers: np.ndarray,
        neg_distribution: np.ndarray,
        keep_distribution: np.ndarray,
        *,
        samples_per_epoch: int = 10_000,
        n_pairs: int = 10,
        n_negative: int = 20,
        seed: int = 4,
        trans_fraction: float = 0.2,
        cis_window: int = 500_000,
        same_chr_negative_prob: float = 0.5,
    ) -> None:
        super().__init__()
        # Data - ensure CSR format for fast row access
        if not isinstance(X, csr_matrix):
            if issparse(X):
                self.X = X.tocsr()
            else:
                raise ValueError("X must be a sparse matrix (CSR or CSC)")
        else:
            self.X = X
        self.n_cells, self.n_peaks = self.X.shape
        self.chr = chr
        self.centers = centers

        # Hyperparameters
        self.samples_per_epoch = samples_per_epoch
        self.n_pairs = n_pairs
        self.n_negative = n_negative
        self.seed = seed
        self.trans_fraction = trans_fraction
        self.cis_window = cis_window
        self.same_chr_negative_prob = same_chr_negative_prob

        # Negative sampling distribution (global)
        self.neg_distribution = torch.from_numpy(neg_distribution)
        self.neg_cat_all = torch.distributions.categorical.Categorical(
            probs=self.neg_distribution
        )

        # Subsampling distribution
        self.keep_distribution = keep_distribution

        # Negative sampling distribution (per chromosome)
        self.by_chr: Dict[str, np.ndarray] = {}
        self.neg_cat_chr: Dict[str, torch.distributions.categorical.Categorical] = {}
        for u in np.unique(self.chr):
            idxs = np.where(self.chr == u)[0]
            self.by_chr[u] = idxs
            f = neg_distribution[idxs]
            self.neg_cat_chr[u] = torch.distributions.categorical.Categorical(
                probs=torch.from_numpy(f / (f.sum() + 1e-12)).float()
            )

        # Optional: debugging / stats
        self.counter = defaultdict(int)

    def _open_cell_peaks(self, cell_idx: int, rng: np.random.Generator) -> np.ndarray:
        """
        Get open peaks for a given cell, applying subsampling.
        """
        row = self.X.getrow(cell_idx)  # returns a 1×n sparse row (CSR-like)
        peaks = row.indices
        if peaks.size < 2:
            return peaks
        mask = rng.random(peaks.size) < self.keep_distribution[peaks]
        peaks = peaks[mask] if mask.any() else peaks
        return peaks

    def _sample_pair(
        self, peak_idx: int, open_peaks: np.ndarray, rng: np.random.Generator
    ) -> int | None:
        """
        Sample a context peak for a given anchor peak from the open peaks of a cell.
        """
        chr = self.chr[peak_idx]
        c_mid = self.centers[peak_idx]

        if rng.random() > self.trans_fraction:
            candidates = open_peaks[
                (self.chr[open_peaks] == chr) & (open_peaks != peak_idx)
            ]
            if candidates.size > 0:
                distances = np.abs(self.centers[candidates] - c_mid)
                in_cis_window = distances <= self.cis_window
                if in_cis_window.any():
                    weights = np.exp(-distances[in_cis_window] / self.cis_window)
                    weights = weights / (weights.sum() + 1e-12)
                    return int(rng.choice(candidates[in_cis_window], p=weights))

        paired_peaks = open_peaks[open_peaks != peak_idx]
        return int(rng.choice(paired_peaks)) if paired_peaks.size > 0 else None

    def _sample_negatives(
        self, peak_idx: int, pair_idx: int, rng: np.random.Generator
    ) -> torch.Tensor:
        """
        Sample negative peaks, with a probability of sampling from the same chromosome.
        """
        if rng.random() < self.same_chr_negative_prob:
            chr = self.chr[peak_idx]
            idxs = self.by_chr[chr]
            draws = self.neg_cat_chr[chr].sample((self.n_negative,))
            negatives = torch.from_numpy(idxs[draws.numpy().astype(int)])
        else:
            negatives = self.neg_cat_all.sample((self.n_negative,))

        if isinstance(negatives, torch.Tensor):
            negatives = torch.where(
                negatives == peak_idx, (negatives + 1) % self.n_peaks, negatives
            )
            negatives = torch.where(
                negatives == pair_idx, (negatives + 2) % self.n_peaks, negatives
            )
        return negatives

    def __iter__(self) -> Iterator[Tuple[int, int, torch.Tensor]]:
        """
        Yield (anchor_peak, context_peak, negatives) tuples.
        """
        wi = get_worker_info()
        worker_id = wi.id if wi else 0
        rng = np.random.Generator(np.random.PCG64(self.seed + 1337 * (worker_id + 1)))

        produced = 0
        while produced < self.samples_per_epoch:
            cell_idx = rng.integers(0, self.n_cells)
            open_peaks = self._open_cell_peaks(cell_idx, rng)
            # print(f"Cell {cell_idx} has {open_peaks.size} open peaks.")
            values, counts = np.unique(self.chr[open_peaks], return_counts=True)
            # Creates a dictionary
            count_dict = dict(zip(values, counts))
            # print(f"Open peaks by chromosome: {count_dict}")
            if open_peaks.size < 2:
                continue

            sampled_peaks = rng.choice(
                open_peaks, size=self.n_pairs, replace=(len(open_peaks) < self.n_pairs)
            )
            # print(self.chr[sampled_peaks].to_list())
            for peak_idx in sampled_peaks:
                # print(self.chr[peak_idx])
                pair_idx = self._sample_pair(peak_idx, open_peaks, rng)
                if pair_idx is None:
                    continue

                negatives = self._sample_negatives(peak_idx, pair_idx, rng)
                self.counter[self.chr[peak_idx]] += 1
                yield int(peak_idx), int(pair_idx), negatives

                produced += 1
                if produced >= self.samples_per_epoch:
                    break


def peak2vec_collate(
    batch: List[Tuple[int, int, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for PeakDataset."""
    peaks, peak_pairs, negatives = zip(*batch)
    peaks = torch.tensor(peaks, dtype=torch.long)
    peak_pairs = torch.tensor(peak_pairs, dtype=torch.long)
    negatives = torch.stack(negatives, dim=0).long()
    return peaks, peak_pairs, negatives


def steps_per_epoch(samples_per_epoch: int, batch_size: int) -> int:
    """Calculate number of steps per epoch given samples and batch size."""
    return int(np.ceil(samples_per_epoch / batch_size))
