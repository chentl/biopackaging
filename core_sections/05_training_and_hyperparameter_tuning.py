# ================================
# SECTION 5: TRAINING UTILITIES AND HYPERPARAMETER TUNING
# ================================
"""
Training utilities aligned with ann_model/trainer.py and ann_model/metrics.py.
Provides training functions and hyperparameter tuning for neural networks.

NOTE: This module depends on imports from section 01 and architecture from section 03.
Load sections 01-03 before loading this section.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import product
from tqdm.auto import tqdm

# Import from section 03 if available
try:
    from core_sections import ConfigurableANN, RNG_SEED
except ImportError:
    try:
        ConfigurableANN  # Check if already defined
        RNG_SEED  # Check if already defined
    except NameError:
        # Fallback - these should be loaded from sections 01 and 03
        RNG_SEED = 0
        ConfigurableANN = None  # Will be defined by loading section 03

def compute_mre_percent(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute Mean Relative Error percentage.
    Aligned with mean_absolute_percentage_error but returns percentage.
    """
    epsilon = 1e-8
    relative_errors = np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))
    return np.mean(relative_errors) * 100.0

def train_model(model, X_train, y_train, X_val, y_val, learning_rate=1e-3, 
                batch_size=32, epochs=150, weight_decay=1e-5):
    """
    Training function aligned with ann_model/trainer.py approach.
    Handles small datasets and various edge cases.
    
    Args:
        model: Neural network model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        learning_rate: Learning rate for Adam optimizer
        batch_size: Batch size for training
        epochs: Number of training epochs
        weight_decay: L2 regularization weight
        
    Returns:
        train_losses, val_losses: Lists of losses per epoch
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Setup optimizer and loss (aligned with ann_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Determine effective batch size
    effective_batch_size = min(batch_size, len(X_train_tensor))
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        
        epoch_train_loss = 0
        n_train_batches = 0
        
        for i in range(0, len(X_train_tensor), effective_batch_size):
            batch_X = X_train_tensor[i:i+effective_batch_size]
            batch_y = y_train_tensor[i:i+effective_batch_size]
            
            optimizer.zero_grad()
            
            # Forward pass
            hidden = model(batch_X)
            output = model.output_layer(hidden)
            if hasattr(model, 'output_activation'):
                output = model.output_activation(output)
            
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            n_train_batches += 1
        
        avg_train_loss = epoch_train_loss / max(n_train_batches, 1)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            hidden = model(X_val_tensor)
            output = model.output_layer(hidden)
            if hasattr(model, 'output_activation'):
                output = model.output_activation(output)
            val_loss = criterion(output, y_val_tensor).item()
            val_losses.append(val_loss)
    
    return train_losses, val_losses


# Backward compatibility alias
def safe_train_model(model, X_train, y_train, X_val, y_val, hyperparams, epochs=150):
    """Backward compatibility wrapper for train_model."""
    return train_model(
        model, X_train, y_train, X_val, y_val,
        learning_rate=hyperparams.get('learning_rate', 1e-3),
        batch_size=hyperparams.get('batch_size', 32),
        epochs=epochs,
        weight_decay=hyperparams.get('weight_decay', 1e-5)
    )


# Hyperparameter search space
HYPERPARAMETER_SPACE = {
    'hidden_layers': [
        (128, 64, 32), (64, 32), (128, 64), (256, 128, 64),
        (64, 64, 32), (128, 128, 64), (32, 16), (96, 48, 24)
    ],
    'activation': ['relu', 'elu', 'tanh', 'leaky_relu'],
    'dropout': [0.0, 0.1, 0.2, 0.3],
    'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
    'batch_size': [16, 32, 64],
    'use_batch_norm': [False, True]
}

def hyperparameter_tuning(task, X_train, y_train, n_folds=5, max_combinations=100):
    """
    Perform hyperparameter tuning using k-fold cross-validation.
    """
    print(f"🔍 Hyperparameter Tuning: {task.upper()}")
    
    # Generate combinations
    keys = list(HYPERPARAMETER_SPACE.keys())
    values = [HYPERPARAMETER_SPACE[k] for k in keys]
    all_combinations = list(product(*values))
    
    # Sample combinations if too many
    if len(all_combinations) > max_combinations:
        np.random.seed(42)
        sampled_indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
        combinations = [all_combinations[i] for i in sampled_indices]
    else:
        combinations = all_combinations
    
    print(f"📊 Testing {len(combinations)} combinations with {n_folds}-fold CV")
    
    results = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Test each combination
    for idx, combo in enumerate(tqdm(combinations, desc=f"Tuning {task}")):
        config = dict(zip(keys, combo))
        fold_scores = []
        
        # Cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr_fold = X_train[train_idx]
            y_tr_fold = y_train[train_idx]
            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]
            
            # Create and configure model
            model = ConfigurableANN(
                input_dim=X_train.shape[1],
                hidden_layers=config['hidden_layers'],
                activation=config['activation'],
                dropout=config['dropout'],
                use_batch_norm=config['use_batch_norm']
            )
            model.output_layer = nn.Linear(model.output_dim, y_train.shape[1])
            
            # Set output activation based on task (aligned with ann_model)
            if task == 'tensile':
                model.output_activation = nn.ReLU()  # Positive tensile properties
            else:  # optical
                model.output_activation = nn.Sigmoid()  # 0-1 range
            
            # Train and validate
            train_losses, val_losses = train_model(
                model, X_tr_fold, y_tr_fold, X_val_fold, y_val_fold,
                learning_rate=config['learning_rate'],
                batch_size=config['batch_size'],
                weight_decay=config.get('weight_decay', 1e-5)
            )
            fold_scores.append(val_losses[-1])
        
        # Store results
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        results.append({
            'config': config,
            'mean_cv_score': mean_score,
            'std_cv_score': std_score,
            'fold_scores': fold_scores
        })
    
    # Sort by performance (lower is better)
    results.sort(key=lambda x: x['mean_cv_score'])
    
    # Print top configurations
    print(f"\n🏆 Top 5 configurations for {task}:")
    print("-" * 70)
    for i, result in enumerate(results[:5]):
        print(f"\nRank {i+1}: CV Score = {result['mean_cv_score']:.4f} ± {result['std_cv_score']:.4f}")
        config = result['config']
        print(f"    Hidden layers: {config['hidden_layers']}")
        print(f"    Activation: {config['activation']}")
        print(f"    Dropout: {config['dropout']}")
        print(f"    Learning rate: {config['learning_rate']}")
        print(f"    Batch size: {config['batch_size']}")
        print(f"    Batch norm: {config['use_batch_norm']}")
    
    return results

def safe_run_ensemble_experiment(X_data, y_data, task_name, n_models=5, n_folds=3, max_configs=100):
    """
    Run a safe ensemble experiment with error handling and robust training.
    
    Args:
        X_data: Input features
        y_data: Target values
        task_name: Name of the task for logging
        n_models: Number of models in ensemble
        n_folds: Number of CV folds
        max_configs: Maximum hyperparameter configurations to test
        
    Returns:
        Dictionary with experiment results
    """
    print(f"\n🎯 Running Ensemble Experiment: {task_name}")
    print(f"📊 Data shape: X={X_data.shape}, y={y_data.shape}")
    
    try:
        # Check data validity
        if len(X_data) == 0 or len(y_data) == 0:
            print("❌ Empty data provided")
            return {'error': 'Empty data'}
        
        if X_data.shape[0] != y_data.shape[0]:
            print(f"❌ Data shape mismatch: X has {X_data.shape[0]} samples, y has {y_data.shape[0]}")
            return {'error': 'Shape mismatch'}
        
        # Determine task type from task name
        task_type = 'tensile' if 'tensile' in task_name else 'optical'
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.2, random_state=RNG_SEED
        )
        
        # Perform hyperparameter tuning
        tuning_results = hyperparameter_tuning(
            task_type, X_train, y_train, 
            n_folds=n_folds, 
            max_combinations=max_configs
        )
        
        if not tuning_results:
            print("❌ No hyperparameter configurations tested")
            return {'error': 'No configurations'}
        
        # Get best configuration
        best_config = tuning_results[0]['config']
        print(f"🏆 Best config: {best_config}")
        
        # Train ensemble models
        models = []
        model_scores = []
        
        kf = KFold(n_splits=n_models, shuffle=True, random_state=RNG_SEED)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr_fold = X_train[train_idx]
            y_tr_fold = y_train[train_idx]
            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]
            
            # Create and configure model
            model = ConfigurableANN(
                input_dim=X_data.shape[1],
                hidden_layers=best_config['hidden_layers'],
                activation=best_config['activation'],
                dropout=best_config['dropout'],
                use_batch_norm=best_config['use_batch_norm']
            )
            model.output_layer = nn.Linear(model.output_dim, y_data.shape[1])
            
            # Set output activation based on task (aligned with ann_model)
            if task_type == 'tensile':
                model.output_activation = nn.ReLU()  # Positive tensile properties
            else:  # optical
                model.output_activation = nn.Sigmoid()  # 0-1 range
            
            # Train model
            train_losses, val_losses = train_model(
                model, X_tr_fold, y_tr_fold, X_val_fold, y_val_fold,
                learning_rate=best_config['learning_rate'],
                batch_size=best_config['batch_size'],
                weight_decay=best_config.get('weight_decay', 1e-5)
            )
            
            models.append(model)
            model_scores.append(val_losses[-1])
        
        # Calculate ensemble predictions on test set
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ensemble_preds = []
        
        for model in models:
            model.eval()
            model = model.to(device)
            
            with torch.no_grad():
                try:
                    X_test_tensor = torch.FloatTensor(X_test).to(device)
                    hidden = model(X_test_tensor)
                    outputs = model.output_layer(hidden)
                    if hasattr(model, 'output_activation'):
                        outputs = model.output_activation(outputs)
                    ensemble_preds.append(outputs.cpu().numpy())
                except Exception as e:
                    print(f"❌ Model prediction error: {e}")
                    # Return zero predictions as fallback
                    ensemble_preds.append(np.zeros_like(y_test))
        
        if not ensemble_preds:
            print("❌ No successful model predictions")
            return {'error': 'No predictions'}
        
        # Average ensemble predictions
        final_predictions = np.mean(ensemble_preds, axis=0)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, final_predictions)
        mae = mean_absolute_error(y_test, final_predictions)
        mre = compute_mre_percent(final_predictions, y_test)
        r2 = r2_score(y_test, final_predictions)
        
        # Prepare results
        results = {
            'task': task_name,
            'val_mse': mse,
            'val_mae': mae,
            'val_mre': mre,
            'val_r2': r2,
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
            'best_config': best_config,
            'model_scores': model_scores,
            'cv_score_best': tuning_results[0]['mean_cv_score'],
            'cv_score_std': tuning_results[0]['std_cv_score']
        }
        
        print(f"✅ Experiment Complete: MRE = {mre:.3f}%, R² = {r2:.3f}")
        return results
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        return {'error': str(e), 'task': task_name}

print("✅ Section 5: Training Utilities Complete (aligned with ann_model/trainer.py)")
print(f"🔧 Hyperparameter space: {len(list(product(*HYPERPARAMETER_SPACE.values())))} total combinations")
print("🔧 Training function: train_model() with Adam optimizer")
print("🔧 Metrics: compute_mre_percent() aligned with ann_model/metrics.py")
print("🔧 Ensemble function: Full pipeline with error handling")