# Enhanced Node Distinguishability Heterogenous Graph---ENDHG
# ENDHG-Aligned: Heterogeneous Graph Neural Network

## Disclaimer

This is not the official implementation of the referenced paper. It is an independent reconstruction built from scratch based on the mathematical formulations described in:

Zhao, Y., Wang, W., Wang, S., Dong, J., and Duan, H. — *Over-smoothing problem of heterogeneous graph neural networks: A heterogeneous graph neural network with enhanced node differentiability.*

The goal of this project is to carefully translate the equations and structural ideas from the paper into a clean, modular PyTorch implementation.

---

## Quick Start

Below is a minimal example of how to initialize the model, preprocess the graph, and run training.

### ! Make sure to toggle `inputs_are_preprocessed` - `True` if your inputs are preprocessed else `False`
> Double Preprocessing can lead to heavy loss on information and sometimes no signal representation for the model to learn from it.

```python
model = ENDHG_Aligned_v2(
    d_tx=X_tx.size(1),
    attr_input_dims=attr_input_dims,
    hidden_dim=128,
    num_classes=2,
    numlayers=3
)

# Preprocess meta-path adjacencies once before training
meta_adjs_tilde, A_He_hat = preprocess_graph_once(
    meta_adjs_raw, A_He_raw
)

model, test_stats, _ = train_model(
    model,
    X_tx, X_attrs,
    meta_adjs_tilde, A_He_hat,
    info, y_tx,
    train_mask, val_mask, test_mask,
    epochs=200,
    device="cpu",
    inputs_are_preprocessed=True
)
```

---

## Overview

This repository implements an ENDHG-aligned heterogeneous graph neural network designed to address the following problems:

- Over-smoothing in heterogeneous graph neural networks
- Loss of node-level differentiation
- Meta-path based aggregation instability
- Heterogeneous information fusion

The implementation follows Equations 16 through 25 from the paper as closely as possible while maintaining practical engineering constraints such as sparse efficiency and memory stability.

---

## Key Design Principles

**Strict equation alignment.** The implementation directly mirrors the mathematical structure described in the paper, including meta-path convolution, semantic attention, and heterogeneous aggregation.

**Sparse graph operations.** All adjacency matrices are constructed and processed using sparse COO tensors and sparse matrix multiplication.

**Preprocessing outside the model.** Meta-path adjacencies are normalized once before training to ensure stability and reproducibility.

**Controlled meta-path growth.** Transaction-wise top-K pruning is used to prevent memory explosion in T-A-T constructions.

---

## Mathematical Components

### Meta-Path Construction

Given a transaction-to-attribute adjacency matrix, the meta-path adjacency is constructed by multiplying the adjacency matrix with its transpose. The result is then preprocessed using row normalization, followed by resetting the diagonal to 1. This diagonal reset prevents self-feature dilution during convolution.

### Meta-Path Convolution (Equation 16)

Each meta-path embedding is computed by applying the preprocessed adjacency matrix to the node feature matrix, then passing the result through a learnable weight matrix and a non-linear activation. Each meta-path adjacency is square, preprocessed once before training, and applied via sparse matrix multiplication.

### Semantic Attention (Equations 20-21)

When multiple meta-path embeddings are available, each one is assigned a scalar attention weight. These weights are normalized using softmax across all meta-paths, producing a weighted combination of the embeddings that reflects their relative importance for the downstream task.

### Heterogeneous Aggregation (Equation 22)

Attribute node embeddings are projected into a shared hidden space using a learnable weight matrix. The transaction-to-attribute adjacency, after row normalization, is then used to aggregate these projected attribute embeddings into the transaction node space.

### Fusion and Classification (Equations 23-24)

The meta-path embedding and the heterogeneous aggregation embedding are summed to form a unified node representation. This representation is then passed through a final linear layer to produce classification logits.

---

## Implementation Structure

The codebase is organized around the following core components:

**row_norm_adj** handles row normalization of adjacency matrices.

**delta_set_diag_to_one** implements the diagonal reset operator described in the paper.

**preprocess_meta_adjs** runs the full adjacency preprocessing pipeline once before training begins.

**build_meta_adjs_and_A_He_raw** constructs raw graph structures from the input data.

**ENDHG_Aligned_v2** is the main model class that ties all components together.

**HeteroBuildInfo** is a container that holds graph structure metadata used throughout the model.

---

## Meta-Path Pruning

To keep memory usage manageable, transaction-wise pruning is applied during graph construction. For each transaction node, only the top-K strongest outgoing edges are retained. This reduces the density of the adjacency matrix while preserving the most structurally significant connections. This pruning step is applied before normalization and is separate from any of the equation-level preprocessing.

---

## Training Recommendations

Heterogeneous GNNs can be sensitive to random initialization, so it is recommended to run training across multiple seeds and report both mean and standard deviation of results. For imbalanced classification tasks, ROC-AUC and PR-AUC are more informative metrics than accuracy. Threshold selection for binary classification should never be done using the test set.

---

## Limitations

This is not the official implementation of the paper. Some architectural details were inferred from the equations since the paper does not provide a reference codebase. Hyperparameters used here are implementation-driven rather than sourced from the paper. If you plan to stack multiple layers, note that deeper architectures may require residual connections to avoid reintroducing over-smoothing.

---

## Citation

If you use this implementation in your research, please cite the original paper:

```
Zhao, Y., Wang, W., Wang, S., Dong, J., and Duan, H.
Over-smoothing problem of heterogeneous graph neural networks:
A heterogeneous graph neural network with enhanced node differentiability.
```

---

## Author Note

This project was developed to build a deeper understanding of over-smoothing in heterogeneous GNNs, meta-path semantic aggregation, sparse adjacency processing, and node-level differentiation preservation. The implementation prioritizes clarity, structural alignment with the paper, and reproducibility over engineering optimization.
