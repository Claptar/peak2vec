from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class PreprocessingConfig:
    """Preprocessing settings."""

    chrom_col: str = "Chromosome"
    start_col: str = "Start"
    end_col: str = "End"
    center_col: str = "Center"
    source: str = "var_names"  # var_names|column
    peak_name_col: Optional[str] = None
    overwrite: bool = False


@dataclass
class SamplingConfig:
    """How we sample positive pairs and negatives from the ATAC matrix."""

    samples_per_epoch: int = 20_000
    n_pairs: int = 20
    n_negative: int = 20
    trans_fraction: float = 0.2
    cis_window: int = 500_000
    same_chr_negative_prob: float = 0.5

    # Distributions computed from peak frequency
    subsample_t: float = 5e-7
    neg_power: float = 0.75


@dataclass
class TrainConfig:
    """Model + optimizer settings."""

    embedding_dim: int = 128
    pos_weight: float = 1.0
    sparse: bool = True
    tie_weights: bool = True

    lr: float = 2e-3
    weight_decay: float = 0.0
    batch_size: int = 512
    epochs: int = 200

    seed: int = 4
    device: str = "auto"  # auto|cpu|cuda|mps
    num_workers: int = 0
    pin_memory: bool = True

    # Logging / checkpoints
    checkpoint_every_epochs: int = 10
    save_embeddings_every_epochs: int = 10

    # Numerical stability / training
    grad_clip_norm: Optional[float] = None


@dataclass
class WandbConfig:
    """Weights & Biases settings.

    Set `mode` to:
      - "online" (default): logs to the server
      - "offline": logs locally, sync later
      - "disabled": turns W&B off
    """

    project: Optional[str] = None
    entity: Optional[str] = None
    group: Optional[str] = None
    name: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    mode: str = "online"  # online|offline|disabled
    save_table: bool = (
        True  # Whether to save downsampled peak table for embedding logging
    )
    n_per_chromosome: int = 200  # Number of peaks to log per chromosome

    # Final embedding visualization
    visualize_embeddings: bool = True  # Whether to generate and upload final embeddings
    viz_metadata_cols: list[str] = field(
        default_factory=lambda: ["Chromosome"]
    )  # Columns from adata.var to use for coloring
    viz_n_neighbors: int = 15  # Number of neighbors for UMAP
    viz_metric: str = "cosine"  # Distance metric for UMAP


@dataclass
class ExperimentConfig:
    """Top-level experiment config."""

    adata_path: Path = Path("data/pbmc10k_eda.h5ad")
    outdir: Path = Path("outputs/run")

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Make Paths YAML-friendly
        d["adata_path"] = str(self.adata_path)
        d["outdir"] = str(self.outdir)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExperimentConfig":
        # Merge with defaults to be resilient to missing keys
        cfg = ExperimentConfig()
        if "adata_path" in d:
            cfg.adata_path = Path(d["adata_path"])
        if "outdir" in d:
            cfg.outdir = Path(d["outdir"])

        if "preprocessing" in d:
            cfg.preprocessing = PreprocessingConfig(
                **{**asdict(cfg.preprocessing), **d["preprocessing"]}
            )
        if "sampling" in d:
            cfg.sampling = SamplingConfig(**{**asdict(cfg.sampling), **d["sampling"]})
        if "train" in d:
            cfg.train = TrainConfig(**{**asdict(cfg.train), **d["train"]})
        if "wandb" in d:
            cfg.wandb = WandbConfig(**{**asdict(cfg.wandb), **d["wandb"]})

        return cfg


def load_config(path: Path) -> ExperimentConfig:
    with path.open("r") as f:
        d = yaml.safe_load(f) or {}
    return ExperimentConfig.from_dict(d)


def save_config(cfg: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
