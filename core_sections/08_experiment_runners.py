# ================================
# SECTION 8: EXPERIMENT RUNNERS
# ================================
#
# NOTE: These functions rely on globals being already loaded from sections 01-07.
# Load sections 01-07 before loading this section.
#

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_uip_experiment(task_name, augmentation_ratios=None):
    """
    Run UIP augmentation experiments with 5 models per configuration.
    
    Args:
        task_name: 'tensile' or 'optical'
        augmentation_ratios: List of ratios to test, e.g., [1, 10, 100]
        
    Returns:
        dict: Results for each augmentation ratio
    """
    if augmentation_ratios is None:
        augmentation_ratios = [1, 10, 100]
    
    # Access globals loaded from previous sections
    # These should be available: TARGET_Y_COLS, X_COLS, RNG_SEED, N_JOBS, df,
    # apply_uip_augmentation, ARCHITECTURE_CONFIGS, HYPERPARAMETER_SETS,
    # train_ensemble_parallel, compute_mre_percent
    
    print(f"\n🚀 UIP EXPERIMENT: {task_name.upper()}")
    print("=" * 60)
    
    # Get target columns
    target_cols = TARGET_Y_COLS[task_name]
    target_std_cols = TARGET_Y_STD_COLS.get(task_name, None)
    print(f"🎯 Target properties: {target_cols}")
    
    # Prepare data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RNG_SEED)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    X_train = train_df[X_COLS].values
    y_train = train_df[target_cols].values
    X_test = test_df[X_COLS].values
    y_test = test_df[target_cols].values
    
    # Load experimental standard deviations if available
    y_train_std = None
    if target_std_cols is not None and all(col in train_df.columns for col in target_std_cols):
        y_train_std = train_df[target_std_cols].values
        print(f"📊 Raw Data: Train={X_train.shape}, Test={X_test.shape} (with experimental std)")
    else:
        print(f"📊 Raw Data: Train={X_train.shape}, Test={X_test.shape}")
    
    results = {}
    
    for ratio in augmentation_ratios:
        print(f"\n🔧 Testing UIP Ratio: 1:{ratio}")
        print("-" * 40)
        
        # Determine architecture complexity based on augmentation ratio
        if ratio == 1:
            arch_config = 'simple'
        elif ratio == 10:
            arch_config = 'medium'
        else:  # ratio == 100
            arch_config = 'complex'
        
        print(f"🏗️ Using {arch_config} architecture with 5 ensemble models")
        
        # CRITICAL: Apply UIP augmentation to RAW (unscaled) data with experimental std
        X_augmented, y_augmented = apply_uip_augmentation(
            X_train, y_train, ratio, task_name, y_train_std=y_train_std
        )
        
        # Calculate adaptive noise level for reporting
        noise_scale_factor = 1.0 / np.sqrt(ratio)
        effective_noise_pct = 15.0 * noise_scale_factor  # Base 15% noise (for samples without exp. std)
        
        # Scale features
        feature_scaler = StandardScaler()
        X_augmented_scaled = feature_scaler.fit_transform(X_augmented)
        X_test_scaled = feature_scaler.transform(X_test)
        
        # IMPORTANT: Scale targets based on task (aligned with ann_model approach)
        if task_name == 'optical':
            # Scale optical targets by TARGET_Y_SCALES for proper range [0, 1]
            # TransVis max=275, TransIR max=86, TransUV max=100 → scales [300, 100, 100]
            optical_scales = np.array(TARGET_Y_SCALES['optical'])  # [300, 100, 100]
            y_augmented_scaled = y_augmented / optical_scales
            y_test_scaled = y_test / optical_scales
            target_scaler = optical_scales  # Store for inverse transform
            print(f"📈 Augmented data: {X_augmented_scaled.shape[0]} samples ({X_augmented_scaled.shape[0]/len(train_df):.1f}x)")
            if y_train_std is not None:
                print(f"   Hybrid noise: Experimental std (where available) + {effective_noise_pct:.1f}% fallback")
            else:
                print(f"   Adaptive noise: {effective_noise_pct:.1f}% (scaled by 1/√{ratio})")
            print(f"   Targets scaled by TARGET_Y_SCALES {optical_scales.tolist()} for Sigmoid activation")
        else:  # tensile
            # Scale tensile targets by TARGET_Y_SCALES (aligned with ann_model)
            # This keeps values positive for ReLU activation
            tensile_scales = np.array(TARGET_Y_SCALES['tensile'])  # [150, 150, 3, 20]
            y_augmented_scaled = y_augmented / tensile_scales
            y_test_scaled = y_test / tensile_scales
            target_scaler = tensile_scales  # Store for inverse transform
            print(f"📈 Augmented data: {X_augmented_scaled.shape[0]} samples ({X_augmented_scaled.shape[0]/len(train_df):.1f}x)")
            if y_train_std is not None:
                print(f"   Hybrid noise: Experimental std (where available) + {effective_noise_pct:.1f}% fallback")
            else:
                print(f"   Adaptive noise: {effective_noise_pct:.1f}% (scaled by 1/√{ratio})")
            print(f"   Targets scaled by TARGET_Y_SCALES {tensile_scales.tolist()} for ReLU activation")
        
        # Train ensemble of 5 models IN PARALLEL
        hyperparameter_sets = HYPERPARAMETER_SETS[arch_config]
        
        ensemble_predictions_scaled, individual_results = train_ensemble_parallel(
            arch_config=arch_config,
            hyperparameter_sets=hyperparameter_sets,
            X_train=X_augmented_scaled,
            y_train=y_augmented_scaled,  # Use scaled/normalized targets
            X_test=X_test_scaled,
            y_test=y_test_scaled,  # Use scaled/normalized test targets
            task_name=task_name,
            n_jobs=N_JOBS
        )
        
        if ensemble_predictions_scaled is None or not ensemble_predictions_scaled:
            print(f"❌ All models failed for ratio 1:{ratio}")
            results[f'1:{ratio}'] = {'error': 'All models failed'}
            continue
        
        # Calculate ensemble predictions (average of all models) on scaled data
        ensemble_final_scaled = np.mean(ensemble_predictions_scaled, axis=0)
        
        # Inverse transform predictions back to original scale for MRE calculation
        if task_name == 'optical':
            ensemble_final = ensemble_final_scaled * target_scaler  # Multiply back by scales
        else:  # tensile
            ensemble_final = ensemble_final_scaled * target_scaler  # Multiply back by scales
        
        # Calculate ensemble metrics on ORIGINAL scale for interpretability
        ensemble_mse_scaled = mean_squared_error(y_test_scaled, ensemble_final_scaled)
        ensemble_mae = mean_absolute_error(y_test, ensemble_final)
        ensemble_mre = compute_mre_percent(ensemble_final, y_test)
        ensemble_r2 = r2_score(y_test, ensemble_final)
        
        # Calculate ensemble statistics
        individual_mres = [r['mre'] for r in individual_results]
        individual_r2s = [r['r2'] for r in individual_results]
        individual_train_losses = [r['final_train_loss'] for r in individual_results]
        individual_val_losses = [r['final_val_loss'] for r in individual_results]
        
        # Get architecture params for printing
        arch_params = ARCHITECTURE_CONFIGS[arch_config]
        
        # Store results (Note: MSE is on scaled targets, MAE/MRE/R² on original scale)
        results[f'1:{ratio}'] = {
            'ensemble': {
                'mse_scaled': ensemble_mse_scaled,  # MSE on scaled/normalized targets
                'mae': ensemble_mae,  # MAE on original scale
                'mre': ensemble_mre,  # MRE on original scale
                'r2': ensemble_r2,  # R² on original scale
                'n_models': len(ensemble_predictions_scaled),
                'avg_train_loss': np.mean(individual_train_losses),
                'avg_val_loss': np.mean(individual_val_losses)
            },
            'individual': individual_results,
            'statistics': {
                'mre_mean': np.mean(individual_mres),
                'mre_std': np.std(individual_mres),
                'r2_mean': np.mean(individual_r2s),
                'r2_std': np.std(individual_r2s),
                'mre_improvement': ensemble_mre - np.mean(individual_mres),
                'r2_improvement': ensemble_r2 - np.mean(individual_r2s),
                'train_loss_mean': np.mean(individual_train_losses),
                'train_loss_std': np.std(individual_train_losses),
                'val_loss_mean': np.mean(individual_val_losses),
                'val_loss_std': np.std(individual_val_losses)
            },
            'architecture': arch_config,
            'n_samples': X_augmented_scaled.shape[0],
            'augmentation_factor': X_augmented_scaled.shape[0] / len(train_df)
        }
        
        print(f"\n✅ UIP Ensemble Results for 1:{ratio}:")
        print(f"   📊 Ensemble MRE: {ensemble_mre:.3f}% (original scale)")
        print(f"   📊 Ensemble R²: {ensemble_r2:.3f} (vs individual mean: {np.mean(individual_r2s):.3f})")
        print(f"   📊 Models trained: {len(ensemble_predictions_scaled)}/5")
        print(f"   📊 Architecture: {arch_params['hidden_layers']}")
        print(f"   📈 Samples: {X_augmented_scaled.shape[0]} ({X_augmented_scaled.shape[0]/len(train_df):.1f}x)")
        print(f"\n   📉 Individual Model Losses:")
        for i, result in enumerate(individual_results):
            print(f"      Model {result['model_idx']}: Train Loss={result['final_train_loss']:.6f}, Val Loss={result['final_val_loss']:.6f}")
        print(f"   📉 Ensemble Average Losses:")
        print(f"      Train Loss: {np.mean(individual_train_losses):.6f} ± {np.std(individual_train_losses):.6f}")
        print(f"      Val Loss: {np.mean(individual_val_losses):.6f} ± {np.std(individual_val_losses):.6f}")
    
    return results

def run_smote_experiment(task_name, augmentation_ratios=None):
    """
    Run SMOTE augmentation experiments with 5 models per configuration.
    
    Args:
        task_name: 'tensile' or 'optical'
        augmentation_ratios: List of ratios to test, e.g., [1, 10, 100]
        
    Returns:
        dict: Results for each augmentation ratio
    """
    if augmentation_ratios is None:
        augmentation_ratios = [1, 10, 100]
    
    print(f"\n🚀 SMOTE EXPERIMENT: {task_name.upper()}")
    print("=" * 60)
    
    # Get target columns
    target_cols = TARGET_Y_COLS[task_name]
    print(f"🎯 Target properties: {target_cols}")
    
    # Prepare data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RNG_SEED)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    X_train = train_df[X_COLS].values
    y_train = train_df[target_cols].values
    X_test = test_df[X_COLS].values
    y_test = test_df[target_cols].values
    
    print(f"📊 Raw Data: Train={X_train.shape}, Test={X_test.shape}")
    
    results = {}
    
    for ratio in augmentation_ratios:
        print(f"\n🔧 Testing SMOTE Ratio: 1:{ratio}")
        print("-" * 40)
        
        # Determine architecture complexity based on augmentation ratio
        if ratio == 1:
            arch_config = 'simple'
        elif ratio == 10:
            arch_config = 'medium'
        else:  # ratio == 100
            arch_config = 'complex'
        
        print(f"🏗️ Using {arch_config} architecture with 5 ensemble models")
        
        # Apply SMOTE augmentation to RAW (unscaled) data
        X_augmented, y_augmented = apply_smote_augmentation(
            X_train, y_train, ratio, task_name
        )
        
        # Calculate adaptive noise level for reporting
        noise_scale_factor = 1.0 / np.sqrt(ratio)
        effective_noise_pct = 2.0 * noise_scale_factor  # SMOTE uses 2% base perturbation
        
        # Scale features
        feature_scaler = StandardScaler()
        X_augmented_scaled = feature_scaler.fit_transform(X_augmented)
        X_test_scaled = feature_scaler.transform(X_test)
        
        # Scale targets based on task
        if task_name == 'optical':
            optical_scales = np.array(TARGET_Y_SCALES['optical'])
            y_augmented_scaled = y_augmented / optical_scales
            y_test_scaled = y_test / optical_scales
            target_scaler = optical_scales
            print(f"📈 Augmented data: {X_augmented_scaled.shape[0]} samples ({X_augmented_scaled.shape[0]/len(train_df):.1f}x)")
            print(f"   SMOTE perturbation: {effective_noise_pct:.1f}% (adaptive)")
            print(f"   Targets scaled by TARGET_Y_SCALES {optical_scales.tolist()} for Sigmoid activation")
        else:  # tensile
            tensile_scales = np.array(TARGET_Y_SCALES['tensile'])
            y_augmented_scaled = y_augmented / tensile_scales
            y_test_scaled = y_test / tensile_scales
            target_scaler = tensile_scales
            print(f"📈 Augmented data: {X_augmented_scaled.shape[0]} samples ({X_augmented_scaled.shape[0]/len(train_df):.1f}x)")
            print(f"   SMOTE perturbation: {effective_noise_pct:.1f}% (adaptive)")
            print(f"   Targets scaled by TARGET_Y_SCALES {tensile_scales.tolist()} for ReLU activation")
        
        # Train ensemble of 5 models IN PARALLEL
        hyperparameter_sets = HYPERPARAMETER_SETS[arch_config]
        
        ensemble_predictions_scaled, individual_results = train_ensemble_parallel(
            arch_config=arch_config,
            hyperparameter_sets=hyperparameter_sets,
            X_train=X_augmented_scaled,
            y_train=y_augmented_scaled,
            X_test=X_test_scaled,
            y_test=y_test_scaled,
            task_name=task_name,
            n_jobs=N_JOBS
        )
        
        if ensemble_predictions_scaled is None or not ensemble_predictions_scaled:
            print(f"❌ All models failed for ratio 1:{ratio}")
            results[f'1:{ratio}'] = {'error': 'All models failed'}
            continue
        
        # Calculate ensemble predictions
        ensemble_final_scaled = np.mean(ensemble_predictions_scaled, axis=0)
        
        # Inverse transform predictions back to original scale
        if task_name == 'optical':
            ensemble_final = ensemble_final_scaled * target_scaler
        else:  # tensile
            ensemble_final = ensemble_final_scaled * target_scaler
        
        # Calculate ensemble metrics on ORIGINAL scale
        ensemble_mse_scaled = mean_squared_error(y_test_scaled, ensemble_final_scaled)
        ensemble_mae = mean_absolute_error(y_test, ensemble_final)
        ensemble_mre = compute_mre_percent(ensemble_final, y_test)
        ensemble_r2 = r2_score(y_test, ensemble_final)
        
        # Calculate ensemble statistics
        individual_mres = [r['mre'] for r in individual_results]
        individual_r2s = [r['r2'] for r in individual_results]
        individual_train_losses = [r['final_train_loss'] for r in individual_results]
        individual_val_losses = [r['final_val_loss'] for r in individual_results]
        
        # Get architecture params for printing
        arch_params = ARCHITECTURE_CONFIGS[arch_config]
        
        # Store results
        results[f'1:{ratio}'] = {
            'ensemble': {
                'mse_scaled': ensemble_mse_scaled,
                'mae': ensemble_mae,
                'mre': ensemble_mre,
                'r2': ensemble_r2,
                'n_models': len(ensemble_predictions_scaled),
                'avg_train_loss': np.mean(individual_train_losses),
                'avg_val_loss': np.mean(individual_val_losses)
            },
            'individual': individual_results,
            'statistics': {
                'mre_mean': np.mean(individual_mres),
                'mre_std': np.std(individual_mres),
                'r2_mean': np.mean(individual_r2s),
                'r2_std': np.std(individual_r2s),
                'mre_improvement': ensemble_mre - np.mean(individual_mres),
                'r2_improvement': ensemble_r2 - np.mean(individual_r2s),
                'train_loss_mean': np.mean(individual_train_losses),
                'train_loss_std': np.std(individual_train_losses),
                'val_loss_mean': np.mean(individual_val_losses),
                'val_loss_std': np.std(individual_val_losses)
            },
            'architecture': arch_config,
            'n_samples': X_augmented_scaled.shape[0],
            'augmentation_factor': X_augmented_scaled.shape[0] / len(train_df)
        }
        
        print(f"\n✅ SMOTE Ensemble Results for 1:{ratio}:")
        print(f"   📊 Ensemble MRE: {ensemble_mre:.3f}% (original scale)")
        print(f"   📊 Ensemble R²: {ensemble_r2:.3f} (vs individual mean: {np.mean(individual_r2s):.3f})")
        print(f"   📊 Models trained: {len(ensemble_predictions_scaled)}/5")
        print(f"   📊 Architecture: {arch_params['hidden_layers']}")
        print(f"   📈 Samples: {X_augmented_scaled.shape[0]} ({X_augmented_scaled.shape[0]/len(train_df):.1f}x)")
        print(f"\n   📉 Individual Model Losses:")
        for i, result in enumerate(individual_results):
            print(f"      Model {result['model_idx']}: Train Loss={result['final_train_loss']:.6f}, Val Loss={result['final_val_loss']:.6f}")
        print(f"   📉 Ensemble Average Losses:")
        print(f"      Train Loss: {np.mean(individual_train_losses):.6f} ± {np.std(individual_train_losses):.6f}")
        print(f"      Val Loss: {np.mean(individual_val_losses):.6f} ± {np.std(individual_val_losses):.6f}")
    
    return results

def run_baseline_experiment(task_name, augmentation_ratios=None):
    """
    Run baseline experiments (no augmentation) with 5 models per configuration.
    
    Args:
        task_name: 'tensile' or 'optical'
        augmentation_ratios: List of ratios to test (used for architecture selection)
        
    Returns:
        dict: Results for each configuration
    """
    if augmentation_ratios is None:
        augmentation_ratios = [1, 10, 100]
    
    # Access globals loaded from previous sections
    
    print(f"\n🚀 BASELINE EXPERIMENT: {task_name.upper()}")
    print("=" * 60)
    
    # Get target columns
    target_cols = TARGET_Y_COLS[task_name]
    print(f"🎯 Target properties: {target_cols}")
    
    # Prepare data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RNG_SEED)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    X_train = train_df[X_COLS].values
    y_train = train_df[target_cols].values
    X_test = test_df[X_COLS].values
    y_test = test_df[target_cols].values
    
    # Scale features
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    # IMPORTANT: Scale targets based on task (aligned with ann_model approach)
    if task_name == 'optical':
        # Scale optical targets by TARGET_Y_SCALES for proper range [0, 1]
        optical_scales = np.array(TARGET_Y_SCALES['optical'])  # [300, 100, 100]
        y_train_scaled = y_train / optical_scales
        y_test_scaled = y_test / optical_scales
        target_scaler = optical_scales
        print(f"📊 Data: Train={X_train_scaled.shape}, Test={X_test_scaled.shape}")
        print(f"   Targets scaled by TARGET_Y_SCALES {optical_scales.tolist()} for Sigmoid activation")
    else:  # tensile
        # Scale tensile targets by TARGET_Y_SCALES (aligned with ann_model)
        tensile_scales = np.array(TARGET_Y_SCALES['tensile'])  # [150, 150, 3, 20]
        y_train_scaled = y_train / tensile_scales
        y_test_scaled = y_test / tensile_scales
        target_scaler = tensile_scales
        print(f"📊 Data: Train={X_train_scaled.shape}, Test={X_test_scaled.shape}")
        print(f"   Targets scaled by TARGET_Y_SCALES {tensile_scales.tolist()} for ReLU activation")
    
    results = {}
    
    for ratio in augmentation_ratios:
        print(f"\n🔧 Testing BASELINE (Architecture: 1:{ratio})")
        print("-" * 40)
        
        # Determine architecture complexity based on "ratio" (just for architecture selection)
        if ratio == 1:
            arch_config = 'simple'
        elif ratio == 10:
            arch_config = 'medium'
        else:  # ratio == 100
            arch_config = 'complex'
        
        print(f"🏗️ Using {arch_config} architecture with 5 ensemble models")
        print(f"📈 Data: {X_train_scaled.shape[0]} samples (1.0x - no augmentation)")
        
        # Train ensemble of 5 models IN PARALLEL with scaled targets
        hyperparameter_sets = HYPERPARAMETER_SETS[arch_config]
        
        ensemble_predictions_scaled, individual_results = train_ensemble_parallel(
            arch_config=arch_config,
            hyperparameter_sets=hyperparameter_sets,
            X_train=X_train_scaled,
            y_train=y_train_scaled,  # Use scaled targets
            X_test=X_test_scaled,
            y_test=y_test_scaled,  # Use scaled test targets
            task_name=task_name,
            n_jobs=N_JOBS
        )
        
        if ensemble_predictions_scaled is None or not ensemble_predictions_scaled:
            print(f"❌ All models failed for baseline 1:{ratio}")
            results[f'1:{ratio}'] = {'error': 'All models failed'}
            continue
        
        # Calculate ensemble predictions (average of all models) on scaled data
        ensemble_final_scaled = np.mean(ensemble_predictions_scaled, axis=0)
        
        # Inverse transform predictions back to original scale
        if task_name == 'optical':
            ensemble_final = ensemble_final_scaled * target_scaler  # Multiply back by scales
        else:  # tensile
            ensemble_final = ensemble_final_scaled * target_scaler  # Multiply back by scales
        
        # Calculate ensemble metrics on ORIGINAL scale for interpretability
        ensemble_mse_scaled = mean_squared_error(y_test_scaled, ensemble_final_scaled)
        ensemble_mae = mean_absolute_error(y_test, ensemble_final)
        ensemble_mre = compute_mre_percent(ensemble_final, y_test)
        ensemble_r2 = r2_score(y_test, ensemble_final)
        
        # Calculate ensemble statistics
        individual_mres = [r['mre'] for r in individual_results]
        individual_r2s = [r['r2'] for r in individual_results]
        individual_train_losses = [r['final_train_loss'] for r in individual_results]
        individual_val_losses = [r['final_val_loss'] for r in individual_results]
        
        # Get architecture params for printing
        arch_params = ARCHITECTURE_CONFIGS[arch_config]
        
        # Store results (Note: MSE is on scaled targets, MAE/MRE/R² on original scale)
        results[f'1:{ratio}'] = {
            'ensemble': {
                'mse_scaled': ensemble_mse_scaled,  # MSE on scaled/normalized targets
                'mae': ensemble_mae,  # MAE on original scale
                'mre': ensemble_mre,  # MRE on original scale
                'r2': ensemble_r2,  # R² on original scale
                'n_models': len(ensemble_predictions_scaled),
                'avg_train_loss': np.mean(individual_train_losses),
                'avg_val_loss': np.mean(individual_val_losses)
            },
            'individual': individual_results,
            'statistics': {
                'mre_mean': np.mean(individual_mres),
                'mre_std': np.std(individual_mres),
                'r2_mean': np.mean(individual_r2s),
                'r2_std': np.std(individual_r2s),
                'mre_improvement': ensemble_mre - np.mean(individual_mres),
                'r2_improvement': ensemble_r2 - np.mean(individual_r2s),
                'train_loss_mean': np.mean(individual_train_losses),
                'train_loss_std': np.std(individual_train_losses),
                'val_loss_mean': np.mean(individual_val_losses),
                'val_loss_std': np.std(individual_val_losses)
            },
            'architecture': arch_config,
            'n_samples': X_train_scaled.shape[0],
            'augmentation_factor': 1.0
        }
        
        print(f"\n✅ BASELINE Ensemble Results for 1:{ratio}:")
        print(f"   📊 Ensemble MRE: {ensemble_mre:.3f}% (original scale)")
        print(f"   📊 Ensemble R²: {ensemble_r2:.3f} (vs individual mean: {np.mean(individual_r2s):.3f})")
        print(f"   📊 Models trained: {len(ensemble_predictions_scaled)}/5")
        print(f"   📊 Architecture: {arch_params['hidden_layers']}")
        print(f"   📈 Samples: {X_train_scaled.shape[0]} (1.0x - no augmentation)")
        print(f"\n   📉 Individual Model Losses:")
        for i, result in enumerate(individual_results):
            print(f"      Model {result['model_idx']}: Train Loss={result['final_train_loss']:.6f}, Val Loss={result['final_val_loss']:.6f}")
        print(f"   📉 Ensemble Average Losses:")
        print(f"      Train Loss: {np.mean(individual_train_losses):.6f} ± {np.std(individual_train_losses):.6f}")
        print(f"      Val Loss: {np.mean(individual_val_losses):.6f} ± {np.std(individual_val_losses):.6f}")
    
    return results

print("✅ [08] Experiment Runners Loaded (aligned with ann_model, with data-driven scales)")
print("🔧 Functions: run_uip_experiment(), run_smote_experiment(), run_baseline_experiment()")
print("🎯 Each function runs full ensemble experiments for a specific augmentation method")
print("⚠️  Key Update: Targets are scaled using data-driven TARGET_Y_SCALES:")
print("   • Optical: Scaled by [300, 100, 100] for TransVis/TransIR/TransUV (Sigmoid activation)")
print("   • Tensile: Scaled by [150, 150, 3, 20] (ReLU activation)")
print("   • MRE and R² calculated on ORIGINAL scale for interpretability")

