# ================================
# IMPORTS AND CONFIGURATION
# ================================
"""
Core imports, dependencies, and configuration setup for the ensemble neural network pipeline.
This module establishes the foundation for all subsequent operations.
"""

# Core Computational Libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import nn

# Machine Learning imports
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                             confusion_matrix, mean_absolute_percentage_error)
from imblearn.over_sampling import SMOTE

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Utility imports
from tqdm.auto import tqdm
import warnings
from itertools import product
from joblib import Parallel, delayed
import multiprocessing
import os

# Configuration
warnings.filterwarnings('ignore')

# Set consistent random seeds for reproducibility
RNG_SEED = 0
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)

# Parallel processing configuration
N_JOBS = 10  # Use all available cores
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent numpy from using multiple threads per process
os.environ['MKL_NUM_THREADS'] = '1'  # Intel MKL optimization

# Define input and target columns (matching ann_model/settings.py)
X_COLS = ['LAP', 'MMT', 'CMC', 'CNF', 'SLK',
          'AGR', 'ALG', 'CAR', 'CHS', 'PEC', 'PUL', 'STA', 'GEL', 'GLU', 'ZIN',
          'GLY', 'FFA', 'LAC', 'LEV', 'PHA', 'SRB', 'SUA', 'XYL']

TARGET_Y_COLS = {
    'optical': ['TransVis', 'TransIR', 'TransUV'],
    'tensile': ['TensileStress', 'TensileStrain', 'TensileModulusLog10', 'TensileToughnessMean100n90'],
}

# Corresponding standard deviation columns for experimental uncertainty
TARGET_Y_STD_COLS = {
    'optical': ['TransVis_STD', 'TransIR_STD', 'TransUV_STD'],
    'tensile': ['TensileStress_STD', 'TensileStrain_STD', 'TensileModulusLog10_STD', 'TensileToughnessMean100n90_STD'],
}

TARGET_Y_SCALES = {
    'grade': [1.0, 1.0, 1.0],
    'optical': [300.0, 100.0, 100.0],  # Updated: TransVis max=275, TransIR max=86, TransUV max=100
    'tensile': [150, 150, 3, 20],
    'fire': [1.0],
}

# Augmentation ratio configurations
AUGMENTATION_RATIOS = {
    '1:1': 1,      # 1:1 ratio (equal synthetic samples)
    '1:10': 10,     # 1:10 ratio (10x more synthetic samples)
    '1:100': 100    # 1:100 ratio (100x more synthetic samples)
}

# Validate configuration setup
def validate_configuration():
    """Validate that all imports and configuration are properly set."""
    print("✅ [01] Imports and Configuration Module Loaded")
    print(f"📊 Dataset Configuration (aligned with ann_model):")
    print(f"   📋 Input features: {len(X_COLS)}")
    print(f"   🎯 Prediction tasks: {list(TARGET_Y_COLS.keys())}")
    for task, cols in TARGET_Y_COLS.items():
        print(f"      {task}: {len(cols)} targets - {cols}")
    
    print(f"🔧 Environment Information:")
    print(f"   🔮 PyTorch version: {torch.__version__}")
    print(f"   💻 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"   🎲 Random seed: {RNG_SEED}")
    print(f"   🚀 Parallel processing: {N_JOBS} cores available")
    
    return True

# Auto-validate on import
if __name__ == "__main__" or __name__ == "__file__":
    validate_configuration()
