# Biopackaging Materials Research

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

## Material Components (23 Total)

**Structural** (5): LAP (Laponite), MMT (Montmorillonite), CMC, CNF, SLK (Silk)  
**Biopolymers** (10): AGR, ALG, CAR, CHS, PEC, PUL, STA, GEL, GLU, ZIN  
**Additives/Plasticizers** (8): GLY, FFA, LAC, LEV, PHA, SRB, SUA, XYL

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

### Train Models
```bash
python -m matal.ann_model.main --data_name mydata --epochs 100 --lr 1e-5 --device cuda
```

## Model Architecture

**Ensemble**: 5 independent models (RE5C, VPLV, PFFZ, CPUT, 2HNP)

- **Encoder**: 23D input → 12 hidden layers (ELU) → 8D latent space
- **Task Heads** (4 layers each):
  - Grade → 3 outputs (Detachability, Flatness, Feasibility)
  - Optical → 3 outputs (Vis/IR/UV transmission)
  - Tensile → 4 outputs (Strength, Strain, Modulus, SED)
  - Fire → 1 output (Fire Resistance)
- **Prediction**: Mean across 5 models; Uncertainty = standard deviation

## Model Files

Each model directory (`model/v3hp/pt_db.v3hp.*/`) contains:
- `*.model.pt.lz4`: PyTorch model weights (LZ4 compressed)
- `*.opt.pt.lz4`: Optimizer state (LZ4 compressed)
- `*.param.json`: Model hyperparameters and training configuration
- `*.param.pk`: Pickled parameters
- `*.hist.csv`: Training history (loss, metrics per epoch)

**Loading models**:
```python
from matal.ann_model.loader import load_model
model = load_model('pt_db.v3hp.RE5C')
```

## Analysis Notebooks

- **`cluster_sensitivity_analysis.ipynb`**: DBSCAN clustering sensitivity analysis across eps/min_samples parameter grids. Evaluates cluster stability using ARI, NMI, silhouette scores with uncertainty filtering.
- **`data_augmentation.ipynb`**: Synthetic data generation techniques for training data expansion
- **`design_boundary*.ipynb`** (3 notebooks): Design space exploration, model-based boundaries, variance analysis
- **`shap.ipynb`**: SHAP feature importance and model interpretability

## Configuration

**Model**: Edit `matal/ann_model/config.py` (layers, activation, dropout, init)  
**Training**: CLI arguments (lr, epochs, l2, noise_std, batch size)  
**Caching**: Automatic caching for checkpoints, compositions, predictions

## Performance

- LZ4/Zstandard compression for efficient storage
- GPU acceleration via PyTorch
- Parallel processing with joblib
- Fast data handling with polars

---

**Note**: Research project - validate predictions experimentally before production use.

