# ================================
# SECTION 7: PARALLEL TRAINING UTILITIES
# ================================
"""
Parallel training utilities aligned with ann_model/metrics.py and ann_model/trainer.py.
Provides functions for training ensemble models in parallel with proper metrics.

NOTE: These functions rely on globals being already loaded from other sections.
Load sections 01-06 before loading this section.
"""

from joblib import Parallel, delayed
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


# Metrics list aligned with ann_model/metrics.py
METRICS = [
    ('mse', mean_squared_error),
    ('mae', mean_absolute_error),
    ('mre', mean_absolute_percentage_error),
    ('r2', r2_score)
]


def compute_mre_percent(y_pred, y_true):
    """
    Compute Mean Relative Error as percentage.
    Aligned with ann_model/metrics.py implementation.
    """
    epsilon = 1e-8
    relative_errors = np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))
    return np.mean(relative_errors) * 100.0

def train_single_model_parallel(args):
    """
    Train a single model - designed to be called in parallel.
    
    Args:
        args: tuple containing all necessary parameters
    
    Returns:
        dict: Model results including predictions and metrics
    """
    (model_idx, hyperparams, arch_params, X_train, y_train, X_test, y_test, 
     task_name, input_dim, output_dim, EnhancedConfigurableANN, train_model) = args
    
    try:
        # Create model with specific hyperparameters
        model = EnhancedConfigurableANN(
            input_dim=input_dim,
            hidden_layers=arch_params['hidden_layers'],
            activation=hyperparams['activation'],
            dropout=hyperparams['dropout'],
            use_batch_norm=hyperparams['use_batch_norm']
        )
        
        # Add output layer
        model.output_layer = nn.Linear(model.output_dim, output_dim)
        
        # Set output activation based on task (aligned with ann_model/config.py)
        # Note: Targets should be scaled before training
        if task_name == 'tensile':
            model.output_activation = nn.ReLU()  # ReLU for positive tensile properties (aligned with ann_model)
        else:  # optical
            model.output_activation = nn.Sigmoid()  # 0-1 range for scaled targets
        
        # Train model (suppress output for parallel execution)
        # Use train_model directly instead of safe_train_model
        train_losses, val_losses = train_model(
            model, X_train, y_train, X_test, y_test,
            learning_rate=hyperparams['learning_rate'],
            batch_size=hyperparams['batch_size'],
            epochs=hyperparams['epochs'],
            weight_decay=hyperparams.get('weight_decay', 1e-5)
        )
        
        # Make predictions
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            hidden = model(X_test_tensor)
            outputs = model.output_layer(hidden)
            predictions = model.output_activation(outputs).cpu().numpy()
        
        # Calculate individual model metrics (aligned with ann_model/metrics.py)
        # Note: These are calculated on the scaled data passed to the function
        # Ensemble metrics will be inverse-transformed to original scale in experiment runners
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mre = compute_mre_percent(predictions, y_test)  # Using our custom function
        r2 = r2_score(y_test, predictions)
        
        return {
            'success': True,
            'model_idx': model_idx + 1,
            'predictions': predictions,
            'mse': mse,
            'mae': mae,
            'mre': mre,
            'r2': r2,
            'hyperparams': hyperparams,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None
        }
        
    except Exception as e:
        import traceback
        return {
            'success': False,
            'model_idx': model_idx + 1,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def train_ensemble_parallel(arch_config, hyperparameter_sets, X_train, y_train, 
                           X_test, y_test, task_name, n_jobs=-1):
    """
    Train multiple models in parallel for ensemble.
    
    Args:
        arch_config: Architecture configuration name ('simple', 'medium', 'complex')
        hyperparameter_sets: List of hyperparameter dictionaries
        X_train, y_train: Training data
        X_test, y_test: Test data
        task_name: 'tensile' or 'optical'
        n_jobs: Number of parallel jobs (-1 for all cores)
    
    Returns:
        tuple: (ensemble_predictions, individual_results)
    """
    # Access globals that should be loaded
    arch_params = ARCHITECTURE_CONFIGS[arch_config]
    
    # Prepare arguments for parallel execution
    args_list = [
        (i, hyperparams, arch_params, X_train, y_train, X_test, y_test,
         task_name, X_train.shape[1], y_train.shape[1], 
         EnhancedConfigurableANN, train_model)
        for i, hyperparams in enumerate(hyperparameter_sets)
    ]
    
    # Train models in parallel
    print(f"   🚀 Training {len(hyperparameter_sets)} models in parallel...")
    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(train_single_model_parallel)(args) for args in args_list
    )
    
    # Separate successful and failed results
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    if failed_results:
        print(f"   ⚠️  {len(failed_results)} model(s) failed to train")
        for failed in failed_results:
            print(f"      ❌ Model {failed['model_idx']}: {failed['error']}")
    
    if not successful_results:
        return None, []
    
    # Extract predictions and individual results
    ensemble_predictions = [r['predictions'] for r in successful_results]
    individual_results = [{
        'model_idx': r['model_idx'],
        'mse': r['mse'],
        'mae': r['mae'],
        'mre': r['mre'],
        'r2': r['r2'],
        'hyperparams': r['hyperparams'],
        'final_train_loss': r['final_train_loss'],
        'final_val_loss': r['final_val_loss']
    } for r in successful_results]
    
    # Print summary
    print(f"   ✅ Successfully trained {len(successful_results)}/{len(hyperparameter_sets)} models")
    for result in individual_results:
        print(f"      🤖 Model {result['model_idx']}: MRE={result['mre']:.3f}%, R²={result['r2']:.3f}")
    
    return ensemble_predictions, individual_results

print("✅ [07] Parallel Training Utilities Loaded")
print("   (Aligned with ann_model/metrics.py and ann_model/trainer.py)")
print(f"🚀 Parallel processing configured for ensemble training")
print(f"🔧 Functions: train_single_model_parallel(), train_ensemble_parallel()")
print(f"📊 Metrics: MSE, MAE, MRE (%), R² - matching ann_model/metrics.py")

