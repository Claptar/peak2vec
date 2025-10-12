# Peak2Vec

**Peak2Vec** is a PyTorch-based machine learning project for analyzing single-cell ATAC-seq data. The repository implements a word2vec-inspired embedding model specifically designed for chromatin accessibility peaks in single-cell genomics data.

## Key Features

- **Deep learning embeddings** for chromatin accessibility peaks using PyTorch
- **Single-cell ATAC-seq analysis** pipeline with scanpy and muon
- **Skip-gram architecture** with negative sampling for learning peak representations
- **Comprehensive EDA workflows** for quality control and visualization
- **CUDA support** for accelerated training

## Main Components

- **Data Processing**: Uses PBMC 10k ATAC-seq dataset for demonstration
- **Model Architecture**: Custom Peak2Vec neural network with embedding layers
- **Analysis Pipeline**: Complete workflow from raw data to embeddings and visualization
- **Jupyter Notebooks**: Interactive analysis in `notebooks/eda.ipynb` and `notebooks/peak2vec.ipynb`

## Dependencies

- PyTorch (with CUDA support)
- scanpy, muon (single-cell analysis)
- scipy, numpy (numerical computing)
- matplotlib, seaborn (visualization)

## Overview

The project demonstrates how to apply natural language processing techniques (word2vec) to genomics data, learning meaningful representations of chromatin accessibility patterns that can be used for downstream analysis like clustering and visualization.

## Getting Started

### Prerequisites

- Python 3.12.10
- uv package manager

### Installation

1. Clone the repository
2. Install dependencies with uv:

   ```bash
   uv sync
   ```

### Usage

1. Start with the EDA notebook: `notebooks/eda.ipynb`
2. Train the Peak2Vec model: `notebooks/peak2vec.ipynb`