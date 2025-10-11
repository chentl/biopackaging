# Biopackaging Materials Research

A comprehensive machine learning framework for predicting and optimizing biopackaging material properties using ensemble artificial neural networks (ANNs).

## Overview

This project provides a sophisticated pipeline for biopackaging materials research, enabling:
- **Multi-property prediction**: Predict grade, optical, tensile, and fire resistance properties of material compositions
- **Composition optimization**: Generate and optimize material recipes using Monte Carlo methods and Bayesian optimization
- **Design boundary analysis**: Analyze feasible design spaces for material compositions
- **Model interpretability**: Leverage SHAP analysis to understand model predictions

The framework uses PyTorch-based ensemble neural networks trained on experimental biopackaging materials data to predict material properties from composition ratios.

## Features

### Machine Learning Models
- **Ensemble Neural Networks**: PyTorch-based multi-head architecture with shared encoder
- **Multi-target Prediction**: Simultaneous prediction of multiple material properties
  - **Grade**: Detachability, Flatness Uniformity, Feasibility
  - **Optical**: Visible, IR, and UV transmission
  - **Tensile**: Strength, Strain, Modulus, Specific Energy Density
  - **Fire Resistance**: Fire resistance rating
- **Uncertainty Quantification**: Ensemble predictions provide mean and standard deviation
- **Model Persistence**: Efficient model serialization with compression (LZ4)

### Composition Generation & Optimization
- **Random Composition Generator**: Generate diverse material compositions
- **Uniform Composition Generator**: Monte Carlo optimization for spaced-out compositions
- **Feasibility Filtering**: Ensure generated compositions meet practical constraints
- **Performance-guided Search**: Optimize for predicted material properties
- **Diversity Maximization**: Generate diverse compositions while maintaining performance

### Analysis Tools
- **Design Boundary Analysis**: Explore feasible design spaces (Jupyter notebooks)
- **SHAP Analysis**: Model interpretability and feature importance
- **Data Augmentation**: Techniques for expanding training datasets
- **Visualization**: Comprehensive plotting utilities for model performance

## Installation

### Requirements
- Python 3.8+ (tested with 3.8, 3.9, 3.10, 3.11)
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd biopackaging
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Installation Notes
- For GPU support, ensure CUDA is properly installed before installing PyTorch
- Apple Silicon Macs may require special installation for some packages
- All versions have been tested on Ubuntu 20.04, macOS 12+, and Windows 10+

## Project Structure

```
biopackaging/
├── matal/                          # Main package
│   ├── ann_model/                  # Artificial Neural Network models
│   │   ├── base.py                 # Base model classes
│   │   ├── builder.py              # Model architecture builder
│   │   ├── config.py               # Model and training configurations
│   │   ├── data.py                 # Data loading and preprocessing
│   │   ├── loader.py               # Model loading utilities
│   │   ├── main.py                 # Training pipeline entry point
│   │   ├── metrics.py              # Performance metrics
│   │   ├── plot.py                 # Visualization utilities
│   │   ├── serializer.py           # Model serialization
│   │   ├── settings.py             # Model settings and constants
│   │   └── trainer.py              # Training loop implementation
│   ├── utils/                      # Utility modules
│   │   ├── array.py                # Array operations
│   │   ├── bunch.py                # Dictionary-like objects
│   │   ├── cache.py                # Caching utilities
│   │   ├── datatype.py             # Data type handling
│   │   ├── dict.py                 # Dictionary operations
│   │   ├── log.py                  # Logging utilities
│   │   └── sid.py                  # Sample ID management
│   ├── recipe.py                   # Composition generation
│   └── settings.py                 # Project settings
├── model/                          # Trained models
│   └── v3hp/                       # Version 3 high-performance models
│       ├── pt_db.v3hp.2HNP/
│       ├── pt_db.v3hp.CPUT/
│       ├── pt_db.v3hp.PFFZ/
│       ├── pt_db.v3hp.RE5C/
│       └── pt_db.v3hp.VPLV/
├── data/                           # Data files
├── models.py                       # Model prediction interface
├── patched_summary_plot.py         # Custom SHAP plotting
├── data_augmentation.ipynb         # Data augmentation notebook
├── design_boundary.ipynb           # Design boundary analysis
├── design_boundary_model.ipynb     # Model-based boundary analysis
├── design_boundary_variance.ipynb  # Variance analysis
├── shap.ipynb                      # SHAP interpretability analysis
└── requirements.txt                # Python dependencies
```

## Material Components

The framework works with the following 23 material components:

### Structural Materials (5)
- **LAP**: Laponite clay
- **MMT**: Montmorillonite clay
- **CMC**: Carboxymethyl cellulose
- **CNF**: Cellulose nanofibers
- **SLK**: Silk fibroin

### Biopolymers (10)
- **AGR**: Agar
- **ALG**: Alginate
- **CAR**: Carrageenan
- **CHS**: Chitosan
- **PEC**: Pectin
- **PUL**: Pullulan
- **STA**: Starch
- **GEL**: Gelatin
- **GLU**: Glucomannan
- **ZIN**: Zein

### Additives & Plasticizers (8)
- **GLY**: Glycerol
- **FFA**: Free fatty acids
- **LAC**: Lactic acid
- **LEV**: Levulinic acid
- **PHA**: Polyhydroxyalkanoates
- **SRB**: Sorbitol
- **SUA**: Succinic acid
- **XYL**: Xylitol

## Usage

### Loading Pre-trained Models

```python
from models import MODELS, gen_pred_df
import numpy as np

# Example composition (23 components)
composition = np.random.random((1, 23))
composition = composition / composition.sum()  # Normalize to sum to 1

# Generate predictions for all properties
predictions = gen_pred_df(composition)
print(predictions)
```

### Generating Material Compositions

```python
from matal.recipe import generate_random_compositions

materials = ['LAP', 'MMT', 'CMC', 'CNF', 'SLK', 'AGR', 'ALG', 'CAR', 
             'CHS', 'PEC', 'PUL', 'STA', 'GEL', 'GLU', 'ZIN', 'GLY', 
             'FFA', 'LAC', 'LEV', 'PHA', 'SRB', 'SUA', 'XYL']

# Generate 30 random compositions
compositions = generate_random_compositions(
    materials=materials,
    n_comps=30,
    n_sources_per_comp=5,  # Use 5 materials per composition
    random_state=42
)
print(compositions)
```

### Optimizing Compositions for Performance

```python
from matal.recipe import UniformCompositionGenerator
from models import MODELS

# Create optimizer
optimizer = UniformCompositionGenerator(
    materials=materials,
    n_comps=20,
    n_iters=100000,
    perf_model=MODELS['pt_db.v3hp.RE5C'],  # Use one of the models
    perf_coeff=1.0,
    dist_coeff=1.0,
    random_state=42
)

# Run optimization
optimal_compositions = optimizer.optimize()
print(optimal_compositions)
```

### Training New Models

```bash
python -m matal.ann_model.main \
    --data_name your_data_name \
    --epochs 100 \
    --lr 1e-5 \
    --encoder__hidden_layers 12 \
    --encoder__n_output 8 \
    --device cuda  # or 'cpu'
```

## Model Architecture

The ensemble consists of 5 independent models (identified by tags: RE5C, VPLV, PFFZ, CPUT, 2HNP) with the following architecture:

1. **Shared Encoder**: 
   - Input: 23-dimensional composition vector
   - 12 hidden layers with ELU activation
   - Output: 8-dimensional latent representation

2. **Task-specific Heads**:
   - **Grade Head**: 4 layers → 3 outputs (Detachability, FlatnessUni, Feasibility)
   - **Optical Head**: 4 layers → 3 outputs (TransVis, TransIR, TransUV)
   - **Tensile Head**: 4 layers → 4 outputs (Strength, Strain, Modulus, SED)
   - **Fire Head**: 4 layers → 1 output (Fire Resistance Rating)

3. **Ensemble Prediction**:
   - Each property is predicted by all 5 models
   - Final prediction: mean across models
   - Uncertainty: standard deviation across models

## Notebooks

### `data_augmentation.ipynb`
Techniques for expanding training datasets through synthetic data generation.

### `design_boundary.ipynb`
Interactive exploration of feasible design spaces for material compositions.

### `design_boundary_model.ipynb`
Model-based analysis of design boundaries using trained neural networks.

### `design_boundary_variance.ipynb`
Analysis of prediction variance across the design space.

### `shap.ipynb`
SHAP (SHapley Additive exPlanations) analysis for model interpretability and feature importance.

## Configuration

### Model Configuration
Modify model architecture parameters in `matal/ann_model/config.py`:
- Number of hidden layers
- Layer sizes
- Activation functions
- Dropout rates
- Weight initialization methods

### Training Configuration
Adjust training parameters:
- Learning rate
- L2 regularization
- Noise augmentation
- Batch size
- Checkpoint frequency

## Advanced Features

### Caching
The framework includes sophisticated caching mechanisms for:
- Model checkpoints during training
- Generated compositions
- Prediction results

### Logging
Comprehensive logging with different levels (debug, info, warning, error) available through `matal.utils.log`.

### Experiment Tracking
Integration with Weights & Biases (wandb) for experiment tracking and visualization.

## Performance Optimization

- **Parallel Processing**: Utilizes `joblib` for parallel computations
- **Efficient Data Structures**: Leverages `polars` for fast data processing
- **Compression**: LZ4 and Zstandard compression for model storage
- **GPU Acceleration**: Full PyTorch GPU support

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this framework in your research, please cite:

```
[Add citation information here]
```

## Contact

[Add contact information here]

## Acknowledgments

This project uses several open-source libraries including PyTorch, scikit-learn, SHAP, and others listed in `requirements.txt`.

---

**Note**: This is a research project. Model predictions should be validated experimentally before use in production applications.

