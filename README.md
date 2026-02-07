# SAC: Semantic Aware Convoys

**A Python framework for mining trajectory patterns with semantic and spatiotemporal constraints.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

## 📌 Overview

**SAC** (Semantic Aware Convoys) is a research code repository designed to detect moving groups of objects (convoys) that satisfy both **spatial proximity** and **semantic diversity** over time.

This repository implements algorithms to solve the *Semantic-Aware Convoy* problem, providing both exact baseline implementations and optimized approaches using grid-based indexing and incremental updates.

The core algorithms address the challenge of discovering **Relaxed Moving Clusters** (RMCs) where:
1.  **Spatial Constraint:** Objects must be density-connected (e.g., via DBSCAN-like density) at each timestamp.
2.  **Semantic Constraint:** Objects must share specific semantic attributes or satisfy interaction rules to satisfy semantic diversity.
3.  **Temporal Constraint:** The cluster must persist for at least a minimum duration ($k$ consecutive timestamps).

## 📂 Repository Structure

The codebase is organized into modules representing different algorithmic approaches to the problem:

├── brute_force/          # Baseline implementation
│   └── ...               # Exact but computationally expensive method (O(N^2))
├── ed_sac/               # Efficient Detection of SAC
│   └── ...               # Incremental algorithm using incremental diversity
├── ed_sac_grid/          # Grid-Based SAC (Optimized)
│   └── ...               # Optimized algorithm using spatial grid indexing for faster neighbor search
├── runExperiments.py     # Main entry point for running benchmarks and experiments
├── utils.py              # Helper functions for data loading, preprocessing, and logging
└── .gitignore            # Git configuration