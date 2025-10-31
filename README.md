# Scalable Discovery of Sustainable Biobased Packaging via Multimodal Predictive Modeling for Improved Postharvest Preservation

Machine learning framework for predicting and optimizing biopackaging material properties using ensemble neural networks.

## Overview

PyTorch-based pipeline for biopackaging materials research:
- **Multi-property prediction**: Grade, optical, tensile, and fire resistance properties
- **Composition optimization**: Generate and optimize material recipes using Monte Carlo/Bayesian methods
- **Design boundary analysis**: Analyze feasible design spaces
- **Model interpretability**: SHAP analysis and clustering sensitivity
- **Uncertainty quantification**: Ensemble predictions with confidence estimates

## Features

- **Ensemble Neural Networks**: 5 models with shared encoder and task-specific heads
- **Properties Predicted**:
  - Grade: Detachability, Flatness Uniformity, Feasibility
  - Optical: Visible/IR/UV transmission
  - Tensile: Strength, Strain, Modulus, Specific Energy Density
  - Fire Resistance
- **Composition Tools**: Random/uniform composition generators with feasibility filtering
- **Analysis**: Design boundaries, SHAP interpretability, clustering sensitivity, data augmentation

## Installation

**Requirements**: Python 3.8+, CUDA (optional for GPU)

```bash
git clone <repository-url>
cd biopackaging
pip install -r requirements.txt
```

## Project Structure

```
biopackaging/
├── matal/                          # Main package
│   ├── ann_model/                  # Neural network models (builder, trainer, config, metrics)
│   ├── utils/                      # Utilities (cache, log, array operations)
│   ├── recipe.py                   # Composition generation and optimization
│   └── settings.py                 # Project settings
├── model/v3hp/                     # Trained ensemble models (5 models, LZ4 compressed)
│   ├── pt_db.v3hp.2HNP/           # Model weights, parameters, training history
│   ├── pt_db.v3hp.CPUT/
│   ├── pt_db.v3hp.PFFZ/
│   ├── pt_db.v3hp.RE5C/
│   └── pt_db.v3hp.VPLV/
├── data/                           # Dataset files
├── models.py                       # Model prediction interface
├── cluster_sensitivity_analysis.ipynb  # DBSCAN clustering sensitivity
├── data_augmentation.ipynb         # Data augmentation techniques
├── design_boundary*.ipynb          # Design space exploration (3 notebooks)
└── shap.ipynb                      # SHAP interpretability
```

## Quick Start

### Predict Properties
```python
from models import gen_pred_df
import numpy as np

composition = np.random.random((1, 23))
composition = composition / composition.sum()
predictions = gen_pred_df(composition)  # Returns all properties + uncertainty
```

### Generate Compositions
```python
from matal.recipe import generate_random_compositions

compositions = generate_random_compositions(
    materials=['LAP', 'MMT', 'CMC', ...],  # All 23 materials
    n_comps=30,
    n_sources_per_comp=5,
    random_state=42
)
```

### Optimize Compositions
```python
from matal.recipe import UniformCompositionGenerator
from models import MODELS

optimizer = UniformCompositionGenerator(
    materials=materials,
    n_comps=20,
    n_iters=100000,
    perf_model=MODELS['pt_db.v3hp.RE5C'],
    random_state=42
)
optimal_comps = optimizer.optimize()
```

## Model

**Ensemble**: 5 independent models (RE5C, VPLV, PFFZ, CPUT, 2HNP)

- **Prediction**: Mean across 5 models; Uncertainty = standard deviation

## Model Files

Each model directory (`model/v3hp/pt_db.v3hp.*/`) contains:
- `*.model.pt.lz4`: PyTorch model weights (LZ4 compressed)
- `*.opt.pt.lz4`: Optimizer state (LZ4 compressed)
- `*.param.pk`: Pickled parameters

**Loading models**:
```python
from matal.ann_model.loader import load_model
model = load_model('pt_db.v3hp.RE5C')
```

## Analysis Notebooks

- **`cluster_sensitivity_analysis.ipynb`**: DBSCAN clustering sensitivity analysis across eps/min_samples parameter grids. Evaluates cluster stability using ARI, NMI, silhouette scores with uncertainty filtering.
- **`data_augmentation.ipynb`**: Synthetic data generation techniques for training data expansion
- **`design_boundary*.ipynb`** (3 notebooks): Design space exploration, model-based boundaries, variance analysis
- **`evaluate_models.ipynb`**: Comprehensive model evaluation including grade task losses, 5-fold cross-validation, ensemble vs individual model comparison, and performance on tensile/optical properties (ANN demo dataset).
- **`shap.ipynb`**: SHAP feature importance and model interpretability

---

