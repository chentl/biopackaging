# Materials Property Prediction with Data Augmentation

Neural network ensemble experiments for predicting tensile and optical properties of materials with intelligent data augmentation.

## Quick Start

```python
# Run all experiments
jupyter notebook run_experiments_1.ipynb
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

## Architecture Scaling

Model complexity increases with data:

| Ratio | Layers | Params | Batch | Epochs | Dropout | BatchNorm |
|-------|--------|--------|-------|--------|---------|-----------|
| 1:1   | (64, 32) | ~5K | 32 | 100 | 0.1 | ✗ |
| 1:10  | (128, 64, 32) | ~18K | 64 | 150 | 0.15 | ✓ |
| 1:100 | (256, 128, 64, 32) | ~55K | 128 | 200 | 0.2 | ✓ |

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

### Example Output
```
✅ UIP Ensemble Results for 1:100:
   📊 Ensemble MRE: 74.248% (original scale)
   📊 Ensemble R²: 0.413
   📊 Architecture: (256, 128, 64, 32)
   📈 Samples: 27573 (101.0x)
   
   📉 Ensemble Average Losses:
      Train Loss: 0.014874 ± 0.000097
      Val Loss: 0.018729 ± 0.000218
```

## Dataset

- **Source**: `data/ann_demo.csv` (342 samples, 23 features)
- **Targets**: 7 properties (3 optical, 4 tensile)
- **Split**: 80% train (273), 20% test (69)
- **Std columns**: Experimental uncertainties for hybrid augmentation

## Key Features

- ✅ Adaptive noise scaling prevents overfitting at high ratios
- ✅ Hybrid augmentation uses real experimental uncertainties
- ✅ Architecture scales with data for optimal capacity
- ✅ Parallel ensemble training (5 models, 10 cores)
- ✅ Original-scale metrics (MRE, R²) for interpretability
- ✅ Loss tracking (train/val) for each model

## Requirements

```bash
pip install torch numpy pandas scikit-learn joblib
pip install smogn  # optional, for optimal SMOTE
```

## Notes

- Models train on scaled targets, evaluate on original scale
- Validation loss = test set monitoring (small dataset constraint)
- Train/val loss gap indicates overfitting (2-3× is healthy)
- Lower MRE at higher ratios shows augmentation effectiveness

