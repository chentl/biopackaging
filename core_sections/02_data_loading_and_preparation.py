# ================================
# DATA LOADING AND PREPARATION
# ================================
"""
Data loading, cleaning, and preparation utilities for the ensemble neural network pipeline.
This module handles all data preprocessing operations before model training.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load and prepare dataset
df = pd.read_csv('data/ann_demo.csv')
print(f"📁 Loaded dataset shape: {df.shape}")

# Clean data by removing rows with missing targets
# Only use target columns that exist in the dataset
all_target_cols = [col for cols in TARGET_Y_COLS.values() for col in cols]
existing_target_cols = [col for col in all_target_cols if col in df.columns]
missing_cols = [col for col in all_target_cols if col not in df.columns]

if missing_cols:
    print(f"⚠️  Note: Columns not in dataset (skipping): {missing_cols}")

df_clean = df.dropna(subset=existing_target_cols)

print(f"🧹 After removing missing targets: {df_clean.shape}")
print(f"📈 Data retention: {len(df_clean)/len(df)*100:.1f}%")

# Display basic information
print(f"\n📋 Dataset Summary:")
print(f"   Total samples: {len(df_clean)}")
print(f"   Input features: {len(X_COLS)}")
print(f"   Target variables: {len(all_target_cols)}")
print(f"   Missing values in targets: {df_clean[all_target_cols].isnull().sum().sum()}")

# Display target statistics (only for existing columns)
print(f"\n📊 Target Variable Ranges:")
for task, cols in TARGET_Y_COLS.items():
    existing_cols = [col for col in cols if col in df_clean.columns]
    if existing_cols:
        print(f"\n{task.upper()} Targets:")
        for col in existing_cols:
            min_val = df_clean[col].min()
            max_val = df_clean[col].max()
            mean_val = df_clean[col].mean()
            print(f"  {col}: [{min_val:.2f}, {max_val:.2f}] (mean: {mean_val:.2f})")
    else:
        print(f"\n{task.upper()}: ⚠️  No columns found in dataset")

def validate_data_preparation():
    """Validate data preparation and provide comprehensive summary."""
    print("\n✅ [02] Data Loading and Preparation Module Loaded")
    print(f"🗂️  Dataset Processing Summary:")
    print(f"   📊 Original dataset: {df.shape[0]} samples")
    print(f"   🧹 After cleaning: {df_clean.shape[0]} samples")
    print(f"   📈 Data retention: {len(df_clean)/len(df)*100:.1f}%")
    print(f"   🔗 Input features: {len(X_COLS)}")
    print(f"   🎯 Target variables: {len(all_target_cols)}")
    
    print(f"📊 Deep Dive - Target Statistics:")
    for task, cols in TARGET_Y_COLS.items():
        existing_cols = [col for col in cols if col in df_clean.columns]
        if existing_cols:
            print(f"\n{task.upper()} Properties:")
            for col in existing_cols:
                min_val = df_clean[col].min()
                max_val = df_clean[col].max()
                mean_val = df_clean[col].mean()
                std_val = df_clean[col].std()
                print(f"  {col}: [{min_val:.3f}, {max_val:.3f}] μ={mean_val:.3f}, σ={std_val:.3f}")
    
    return df_clean

# For backward compatibility with existing variable name
df = df_clean

# Auto-validate on import
if __name__ == "__main__" or __name__ == "__file__":
    validate_data_preparation()
