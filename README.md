# peak2vec

Peak2Vec: Learning peak embeddings from single-cell ATAC-seq data using Skip-gram with Negative Sampling.

## Install

```bash
uv sync
```

## Quick Start

### Using a config file (recommended)

```bash
uv run peak2vec train --config configs/pbmc10k.yaml
```

### Using command-line arguments

```bash
uv run peak2vec train \
  --adata data/pbmc10k_eda.h5ad \
  --outdir outputs/my_run \
  --epochs 200 \
  --embedding-dim 128 \
  --batch-size 512 \
  --lr 0.002
```

## Configuration

### Config File

Create a YAML config file (see [configs/pbmc10k.yaml](configs/pbmc10k.yaml) for a complete example):

```yaml
adata_path: data/pbmc10k_eda.h5ad
outdir: outputs/pbmc10k_run

preprocessing:
  chrom_col: Chromosome
  start_col: Start
  end_col: End
  center_col: Center
  source: var_names  # or "column" to parse from peak_name_col
  
sampling:
  samples_per_epoch: 20000
  n_pairs: 20
  n_negative: 20
  trans_fraction: 0.2
  cis_window: 500000
  subsample_t: 5.0e-7
  neg_power: 0.75

train:
  embedding_dim: 128
  pos_weight: 1.0
  sparse: true
  tie_weights: true
  lr: 0.002
  weight_decay: 0.0
  batch_size: 512
  epochs: 200
  device: auto  # auto|cpu|cuda|mps
  num_workers: 4  # Parallel data loading workers

wandb:
  project: peak2vec
  mode: disabled  # online|offline|disabled
  visualize_embeddings: true  # Generate PCA/UMAP plots at end
  viz_metadata_cols:  # Columns from adata.var for coloring
    - Chromosome
  viz_n_neighbors: 15  # UMAP neighbors
  viz_metric: cosine  # Distance metric for UMAP
```

### Command-line Arguments

Any config parameter can be overridden via command-line flags:

```bash
# Preprocessing
--chrom-col, --start-col, --end-col, --center-col
--peak-source, --peak-name-col, --overwrite-coords

# Sampling
--samples-per-epoch, --n-pairs, --n-negative
--trans-fraction, --cis-window, --same-chr-negative-prob
--subsample-t, --neg-power

# Training
--embedding-dim, --pos-weight, --sparse, --tie-weights
--lr, --weight-decay, --batch-size, --epochs, --seed
--device, --num-workers, --pin-memory
--checkpoint-every-epochs, --save-embeddings-every-epochs

# Weights & Biases
--wandb-project, --wandb-entity, --wandb-group, --wandb-name
--wandb-mode, --wandb-n-per-chromosome, --wandb-save-table
--wandb-visualize-embeddings, --wandb-viz-metadata-cols
```

## Weights & Biases Integration

### Setup

Set your API key:
```bash
export WANDB_API_KEY=your_key_here
# or
wandb login
```

### Online logging

```bash
uv run peak2vec train \
  --config configs/pbmc10k.yaml \
  --wandb-project peak2vec \
  --wandb-entity your-entity \
  --wandb-mode online
```

### Offline logging (for clusters without internet)

```bash
export WANDB_MODE=offline
uv run peak2vec train \
  --config configs/pbmc10k.yaml \
  --wandb-project peak2vec \
  --wandb-mode offline

# Later, sync the run:
wandb sync outputs/your_run/wandb/
```

## Output Structure

```
outputs/
└── run_YYYYMMDD_HHMMSS/
    ├── config.yaml                      # Resolved configuration
    ├── checkpoints/
    │   ├── checkpoint_epoch0010.pt
    │   ├── checkpoint_epoch0020.pt
    │   └── final_model.pt
    ├── visualizations/                  # Final embeddings (if enabled)
    │   ├── peak_embeddings_in.h5ad     # AnnData with PCA/UMAP
    │   ├── peak_embeddings_in_pca.png
    │   ├── peak_embeddings_in_umap.png
    │   ├── peak_embeddings_out.h5ad    # (if tie_weights=False)
    │   ├── peak_embeddings_out_pca.png
    │   └── peak_embeddings_out_umap.png
    └── wandb/                           # W&B logs (if enabled)
```

## Data Requirements

Your AnnData object should have:
- `.X` as a sparse matrix (cells × peaks) - will be converted to CSR format for optimal performance
- `.var` with peak coordinates:
  - Option 1: Chromosome/Start/End columns
  - Option 2: Peak names in `var_names` (e.g., `chr1:100-200` or `chr1_100_200`)

The training script will automatically:
- Convert sparse matrix to CSR format for fast row access
- Parse peak coordinates if not present
- Compute sampling distributions (`neg` and `keep`)
- Prepare the data for training

## Embedding Visualization

Peak2Vec automatically generates embedding visualizations at the end of training:

- **PCA and UMAP** plots colored by metadata columns
- **Input and output embeddings** (if `tie_weights=False`)
- **Uploaded to W&B** automatically (if enabled)
- **Saved locally** as `.h5ad` files for further analysis

### Customizing Visualizations

```yaml
wandb:
  visualize_embeddings: true
  viz_metadata_cols:
    - Chromosome
    - peak_type        # Add any column from adata.var
    - gene_annotation
  viz_n_neighbors: 15  # Adjust for local vs global structure
  viz_metric: cosine   # cosine, euclidean, correlation, etc.
```

### Standalone Visualization

Generate visualizations from a saved checkpoint:

```bash
uv run peak2vec visualize \
  checkpoints/final_model.pt \
  metadata.csv \
  --outdir visualizations/ \
  --which in \
  --n-neighbors 15
```

See [docs/VISUALIZATION.md](docs/VISUALIZATION.md) for detailed documentation.

## Performance Optimization

For large datasets, optimize training speed:

```bash
# Increase data loading workers
uv run peak2vec train --config config.yaml --num-workers 8

# Enable pin_memory for faster GPU transfer
uv run peak2vec train --config config.yaml --pin-memory

# Larger batch sizes (if GPU memory allows)
uv run peak2vec train --config config.yaml --batch-size 1024
```

**Profile your dataset:**
```bash
python scripts/profile_dataset.py data/your_data.h5ad
```

See [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) for comprehensive performance tuning.

## Additional Commands

### Preprocessing

Preprocess ATAC-seq data with QC filtering:

```bash
uv run peak2vec preprocess \
  input.h5ad \
  output_filtered.h5ad \
  --min-cells-per-peak 70 \
  --max-cells-per-peak 9000 \
  --min-peaks-per-cell 500 \
  --plot-qc
```
