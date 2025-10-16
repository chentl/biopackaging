# Materials Property Prediction with Data Augmentation - Experiments

Neural network ensemble experiments for predicting tensile and optical properties of materials with intelligent data augmentation.

## Quick Start

```python
# Run all experiments
jupyter notebook run_experiments.ipynb
```

## Core Sections

Modular pipeline loaded sequentially:

1. **01_imports_and_configuration** - Features, targets, scaling factors
2. **02_data_loading_and_preparation** - Dataset loading (343 samples)
3. **03_neural_network_architecture** - ANNModel with ModuleDict layers
4. **04_data_augmentation_methods** - UIP (hybrid noise) & SMOTE (fallback)
5. **05_training_and_hyperparameter_tuning** - Training loop, metrics (MRE, R²)
6. **06_model_architectures_and_configs** - Scaled architectures (simple/medium/complex)
7. **07_parallel_training_utilities** - Ensemble training with joblib
8. **08_experiment_runners** - Full experiment pipelines

## Augmentation Methods

### UIP (Uncertainty-Informed Perturbation)
- **Hybrid noise**: Experimental std (when available) + adaptive fallback
- **Adaptive scaling**: Noise decreases with ratio (15% @ 1:1 → 1.5% @ 1:100)

### SMOTE (Synthetic Minority Oversampling)
- **Fallback**: Simple oversampling with adaptive perturbation (2% → 0.2%)
- **Optional**: Install `smogn` for regression-specific SMOTE

### Baseline
- No augmentation, architecture scaling only

## Experiments

```python
# Run specific experiments
uip_tensile = run_uip_experiment('tensile', [1, 10, 100])
uip_optical = run_uip_experiment('optical', [1, 10, 100])
smote_tensile = run_smote_experiment('tensile', [1, 10, 100])
baseline_tensile = run_baseline_experiment('tensile', [1, 10, 100])
```

## Output Metrics

For each ratio, prints:
- **MRE** (Mean Relative Error %) - lower is better
- **R²** (Coefficient of Determination) - higher is better
- **Train/Val Losses** - individual models + ensemble average
- **Architecture details** - layers, samples, augmentation type

## Dataset

- **Source**: `data/ann_demo.csv` (342 samples, 23 features)
- **Targets**: 7 properties (3 optical, 4 tensile)
- **Split**: 80% train (273), 20% test (69)
- **Std columns**: Experimental uncertainties for hybrid augmentation


## Requirements

```bash
pip install torch numpy pandas scikit-learn joblib
pip install smogn  # optional, for optimal SMOTE
```


