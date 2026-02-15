# SDC: Semantically Diverse Convoys

**A Python framework for mining trajectory patterns with semantic and spatiotemporal constraints.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

## 📌 Overview

**SDC** (Semantically Diverse Convoys) is a research code repository designed to detect moving groups of objects (convoys) that satisfy both **spatial proximity** and **semantic diversity** over time.

This repository implements algorithms to solve the *Semantically Diverse Convoy* problem, providing both exact baseline implementations and optimized approaches using grid-based indexing and incremental updates.

The core algorithms address the challenge of discovering **Relaxed Moving Clusters** (RMCs) where:
1.  **Spatial Constraint:** Objects must be density-connected (e.g., via DBSCAN-like density) at each timestamp.
2.  **Semantic Constraint:** Objects must share specific semantic attributes or satisfy interaction rules to satisfy semantic diversity.
3.  **Temporal Constraint:** The cluster must persist for at least a minimum duration ($k$ consecutive timestamps).

## 📂 Repository Structure

The codebase is organized into modules representing different algorithmic approaches to the problem:

```text

├── brute_force/          # Baseline implementation
│   └── ...               # Exact but computationally expensive method (O(N^2))
├── ed_SDC/               # Efficient Detection of SDC
│   └── ...               # Incremental algorithm using incremental diversity
├── ed_SDC_grid/          # Grid-Based SDC (Optimized)
│   └── ...               # Optimized algorithm using spatial grid indexing for faster neighbor search
├── runExperiments.py     # Main entry point for running experiments (currently written for synthethic data)
├── utils.py              # Helper functions for data loading, preprocessing, and logging
└── .gitignore            # Git configuration
```

# 🚀 Getting Started

### Prerequisites

This project relies on standard Python scientific computing libraries. We recommend using Python 3.8 or higher.

* **NumPy:** Vectorized operations.
* **Pandas:** Data manipulation and CSV I/O.
* **SciPy / Scikit-learn:** Spatial queries and distance metrics.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/abdullahshamail/SDC.git](https://github.com/abdullahshamail/SDC.git)
    cd SDC
    ```

2.  **Set up a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    
    # On Windows:
    venv\Scripts\activate
    
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install numpy pandas scipy scikit-learn matplotlib
    pip install -U sentence-transformers
    ```

4.  **Run experiments (on synthetic data):**
    ```bash
    python runExperiments.py
    ```