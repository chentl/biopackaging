# ================================
# SECTION 4: DATA AUGMENTATION METHODS
# ================================
import numpy as np
import pandas as pd
try:
    import smogn
    SMOGN_AVAILABLE = True
except ImportError:
    SMOGN_AVAILABLE = False
    print("⚠️ smogn not available, SMOTE will use fallback method")

# Configuration parameters for augmentation
COMPOSITION_SIGMA = 0.01  # Noise for composition perturbation
PROPERTY_REL_SIGMA = 0.15  # Relative noise for properties (~15%)
AUGMENTATION_SEED = 0   # Random seed

def renormalize_compositions(X_like: np.ndarray) -> np.ndarray:
    """
    Renormalize compositions to sum to 1 with non-negativity constraint.
    Critical for preserving physical constraints in augmentation.
    """
    Xn = np.clip(X_like, 0.0, None)
    row_sums = Xn.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return Xn / row_sums

def apply_uip_augmentation(X_train, y_train, ratio, task_name, rng_seed=0, y_train_std=None):
    """
    Apply UIP augmentation with HYBRID noise (experimental std + adaptive relative noise).
    
    Key improvements:
    - Uses EXPERIMENTAL STANDARD DEVIATION when available (more realistic)
    - Falls back to adaptive relative noise when std is missing/zero
    - Noise scales inversely with sqrt(ratio) to prevent overwhelming model
    - Renormalizes compositions to sum to 1
    - Better preserves physical constraints
    
    Args:
        X_train: Raw (unscaled) training features
        y_train: Training targets
        ratio: Augmentation multiplier (e.g., 1, 10, 100)
        task_name: 'tensile' or 'optical'
        rng_seed: Random seed
        y_train_std: Experimental standard deviations (optional, shape same as y_train)
        
    Returns:
        X_combined: Original + augmented features (still unscaled)
        y_combined: Original + augmented targets
    """
    np.random.seed(rng_seed + ratio)
    
    # Virtual replication
    X_rep = np.repeat(X_train, ratio, axis=0)
    y_rep = np.repeat(y_train, ratio, axis=0)
    
    # ADAPTIVE NOISE SCALING: Scale noise inversely with sqrt(ratio)
    noise_scale_factor = 1.0 / np.sqrt(ratio)
    scaled_comp_sigma = COMPOSITION_SIGMA * noise_scale_factor
    scaled_prop_sigma = PROPERTY_REL_SIGMA * noise_scale_factor
    
    # Add Gaussian noise to compositions and renormalize to sum=1
    X_noise = np.random.normal(loc=0.0, scale=scaled_comp_sigma, size=X_rep.shape)
    X_virt = renormalize_compositions(X_rep + X_noise)
    
    # HYBRID NOISE for labels: Use experimental std when available, else relative noise
    noise_samples = np.random.normal(loc=0.0, scale=1.0, size=y_rep.shape)
    
    if y_train_std is not None:
        # Use experimental standard deviation (BEST approach)
        y_std_rep = np.repeat(y_train_std, ratio, axis=0)
        has_std = (y_std_rep > 0) & ~np.isnan(y_std_rep)
        
        # Initialize with original values
        y_virt = y_rep.copy()
        
        # Apply experimental noise where std is available
        y_virt[has_std] += noise_samples[has_std] * y_std_rep[has_std] * noise_scale_factor
        
        # Apply relative noise where std is missing/zero
        if np.any(~has_std):
            y_virt[~has_std] *= (1.0 + noise_samples[~has_std] * scaled_prop_sigma)
    else:
        # Fallback: Use adaptive relative noise (legacy approach)
        y_virt = y_rep * (1.0 + noise_samples * scaled_prop_sigma)
    
    # Apply constraints
    if task_name == 'optical':
        y_virt = np.clip(y_virt, 0.0, 100.0)  # 0-100% range
    elif task_name == 'tensile':
        # Order: TensileStress, TensileStrain, TensileModulusLog10, TensileToughnessMean100n90
        y_virt[:, 0] = np.clip(y_virt[:, 0], 0.0, None)  # Stress > 0
        y_virt[:, 1] = np.clip(y_virt[:, 1], 0.0, None)  # Strain > 0
        # Modulus log10 and toughness can be negative, no clipping needed
    
    # Combine original + virtual
    X_combined = np.vstack([X_train, X_virt])
    y_combined = np.vstack([y_train, y_virt])
    
    return X_combined, y_combined

def apply_smote_augmentation(X_train, y_train, ratio, task_name, rng_seed=0):
    """
    Apply SMOTE-based augmentation for regression using smogn.
    
    Args:
        X_train: Raw (unscaled) training features
        y_train: Training targets
        ratio: Augmentation multiplier (e.g., 1, 10, 100)
        task_name: 'tensile' or 'optical'
        rng_seed: Random seed
        
    Returns:
        X_combined: Original + augmented features
        y_combined: Original + augmented targets
    """
    np.random.seed(rng_seed + ratio)
    
    if not SMOGN_AVAILABLE:
        print(f"⚠️ smogn not available, using simple oversampling for ratio 1:{ratio}")
        # Fallback to simple oversampling
        n_synthetic = len(X_train) * ratio
        synthetic_indices = np.random.choice(len(X_train), n_synthetic, replace=True)
        
        X_synthetic = X_train[synthetic_indices]
        y_synthetic = y_train[synthetic_indices]
        
        # Add small noise
        noise_scale = 0.01 / np.sqrt(ratio)  # Adaptive noise
        X_noise = np.random.normal(0, noise_scale, X_synthetic.shape)
        y_noise = np.random.normal(0, noise_scale * 0.1, y_synthetic.shape)
        
        X_synthetic = renormalize_compositions(X_synthetic + X_noise)
        y_synthetic = y_synthetic + y_noise
        
        # Apply constraints
        if task_name == 'optical':
            y_synthetic = np.clip(y_synthetic, 0.0, 100.0)
        elif task_name == 'tensile':
            y_synthetic[:, 0] = np.clip(y_synthetic[:, 0], 0.0, None)
            y_synthetic[:, 1] = np.clip(y_synthetic[:, 1], 0.0, None)
        
        X_combined = np.vstack([X_train, X_synthetic])
        y_combined = np.vstack([y_train, y_synthetic])
        return X_combined, y_combined
    
    # Use smogn for proper SMOTE regression
    try:
        # Prepare dataframe for smogn
        df = pd.DataFrame(X_train, columns=[f'X{i}' for i in range(X_train.shape[1])])
        for i in range(y_train.shape[1]):
            df[f'y{i}'] = y_train[:, i]
        
        # Calculate target sample size
        target_samples = len(X_train) * (ratio + 1)
        
        # Apply SMOGN to each target variable
        augmented_dfs = []
        for i in range(y_train.shape[1]):
            try:
                # Configure smogn parameters
                smogn_df = smogn.smoter(
                    data=df,
                    y=f'y{i}',
                    k=min(5, len(X_train) - 1),
                    samp_method='extreme',
                    rel_thres=0.5,
                    rel_method='auto',
                    rel_xtrm_type='both',
                    rel_coef=1.5,
                    pert=0.02 / np.sqrt(ratio)  # Adaptive perturbation
                )
                
                # Limit to target size
                if len(smogn_df) > target_samples:
                    smogn_df = smogn_df.sample(n=int(target_samples), random_state=rng_seed)
                
                augmented_dfs.append(smogn_df)
            except Exception as e:
                print(f"⚠️ SMOGN failed for target {i}: {e}, using original data")
                augmented_dfs.append(df)
        
        # Average predictions across all SMOGN runs (ensemble approach)
        result_df = augmented_dfs[0].copy()
        for i in range(1, len(augmented_dfs)):
            result_df = pd.concat([result_df, augmented_dfs[i]], ignore_index=True)
        
        # Deduplicate and limit size
        result_df = result_df.drop_duplicates().reset_index(drop=True)
        if len(result_df) > target_samples:
            result_df = result_df.sample(n=int(target_samples), random_state=rng_seed)
        
        # Extract features and targets
        X_cols = [c for c in result_df.columns if c.startswith('X')]
        y_cols = [c for c in result_df.columns if c.startswith('y')]
        
        X_combined = result_df[X_cols].values
        y_combined = result_df[y_cols].values
        
        # Renormalize compositions
        X_combined = renormalize_compositions(X_combined)
        
        # Apply task-specific constraints
        if task_name == 'optical':
            y_combined = np.clip(y_combined, 0.0, 100.0)
        elif task_name == 'tensile':
            y_combined[:, 0] = np.clip(y_combined[:, 0], 0.0, None)
            y_combined[:, 1] = np.clip(y_combined[:, 1], 0.0, None)
        
        return X_combined, y_combined
        
    except Exception as e:
        print(f"⚠️ SMOGN augmentation failed: {e}, falling back to simple oversampling")
        # Fallback to simple oversampling
        n_synthetic = len(X_train) * ratio
        synthetic_indices = np.random.choice(len(X_train), n_synthetic, replace=True)
        
        X_synthetic = X_train[synthetic_indices]
        y_synthetic = y_train[synthetic_indices]
        
        noise_scale = 0.01 / np.sqrt(ratio)
        X_noise = np.random.normal(0, noise_scale, X_synthetic.shape)
        y_noise = np.random.normal(0, noise_scale * 0.1, y_synthetic.shape)
        
        X_synthetic = renormalize_compositions(X_synthetic + X_noise)
        y_synthetic = y_synthetic + y_noise
        
        if task_name == 'optical':
            y_synthetic = np.clip(y_synthetic, 0.0, 100.0)
        elif task_name == 'tensile':
            y_synthetic[:, 0] = np.clip(y_synthetic[:, 0], 0.0, None)
            y_synthetic[:, 1] = np.clip(y_synthetic[:, 1], 0.0, None)
        
        X_combined = np.vstack([X_train, X_synthetic])
        y_combined = np.vstack([y_train, y_synthetic])
        return X_combined, y_combined

def build_uip_augmented_splits(df_src, target_map, virtual_multiplier=100,
                              comp_sigma=COMPOSITION_SIGMA, prop_rel_sigma=PROPERTY_REL_SIGMA, seed=AUGMENTATION_SEED):
    """
    Build augmented data splits using UIP (Uncertainty-Informed Perturbation).
    
    UIP Method:
    1. Virtual replication of training samples
    2. Add Gaussian noise to compositions (renormalized)
    3. Add relative Gaussian noise to properties (ADAPTIVE based on multiplier)
    """
    rng = np.random.default_rng(seed)
    X_raw = df_src[X_COLS].values

    X_train_new, X_test_new = {}, {}
    y_train_new, y_test_new = {}, {}
    
    # ADAPTIVE NOISE SCALING: Scale noise inversely with sqrt(multiplier)
    noise_scale_factor = 1.0 / np.sqrt(virtual_multiplier)
    scaled_comp_sigma = comp_sigma * noise_scale_factor
    scaled_prop_sigma = prop_rel_sigma * noise_scale_factor

    for task, cols in target_map.items():
        y_raw = df_src[cols].values
        Xtr_raw, Xte_raw, ytr_raw, yte_raw = train_test_split(
            X_raw, y_raw, test_size=0.2, random_state=0
        )

        # Virtual replication
        X_rep = np.repeat(Xtr_raw, virtual_multiplier, axis=0)
        y_rep = np.repeat(ytr_raw, virtual_multiplier, axis=0)

        # Perturb compositions and renormalize (with adaptive noise)
        X_noise = rng.normal(loc=0.0, scale=scaled_comp_sigma, size=X_rep.shape)
        X_virt = renormalize_compositions(X_rep + X_noise)

        # Perturb properties relatively (with adaptive noise)
        rel_noise = rng.normal(loc=0.0, scale=scaled_prop_sigma, size=y_rep.shape)
        y_virt = y_rep * (1.0 + rel_noise)

        # Apply task-specific constraints
        if task == 'optical':
            y_virt = np.clip(y_virt, 0.0, 100.0)
        if task == 'tensile':
            y_virt[:, 0] = np.clip(y_virt[:, 0], 0.0, None)  # TensileStress
            y_virt[:, 1] = np.clip(y_virt[:, 1], 0.0, None)  # TensileToughness

        # Combine real + virtual samples
        Xtr_aug = np.vstack([Xtr_raw, X_virt])
        ytr_aug = np.vstack([ytr_raw, y_virt])

        # Standardize features
        scaler = StandardScaler()
        Xtr_aug_scaled = scaler.fit_transform(Xtr_aug)
        Xte_scaled = scaler.transform(Xte_raw)

        X_train_new[task] = Xtr_aug_scaled
        y_train_new[task] = ytr_aug
        X_test_new[task] = Xte_scaled
        y_test_new[task] = yte_raw
        print(f"  {task}: {len(Xtr_raw)} → {len(Xtr_aug)} samples")

    return X_train_new, y_train_new, X_test_new, y_test_new

def build_smote_augmented_splits(df_src, target_map, sampling_strategy='auto', 
                                k_neighbors=5, random_state=AUGMENTATION_SEED):
    """
    Build augmented splits using SMOTE (Synthetic Minority Oversampling Technique).
    
    SMOTE Method:
    1. Discretize continuous targets into bins
    2. Use SMOTE to create synthetic samples
    3. Fallback to simple oversampling if SMOTE fails
    """
    X_raw = df_src[X_COLS].values
    
    X_train_new, X_test_new = {}, {}
    y_train_new, y_test_new = {}, {}
    
    for task, cols in target_map.items():
        y_raw = df_src[cols].values
        Xtr_raw, Xte_raw, ytr_raw, yte_raw = train_test_split(
            X_raw, y_raw, test_size=0.2, random_state=0
        )
        
        n_samples_original = len(Xtr_raw)
        
        try:
            # Create bins for target discretization
            target_bins = {}
            for i, col in enumerate(cols):
                target_values = ytr_raw[:, i]
                bins = np.percentile(target_values, [0, 20, 40, 60, 80, 100])
                target_bins[i] = np.digitize(target_values, bins) - 1
            
            # Create combined labels for SMOTE
            combined_labels = np.zeros(n_samples_original)
            for i in range(n_samples_original):
                label_parts = [target_bins[j][i] for j in range(len(cols))]
                combined_labels[i] = sum(label_parts[j] * (5**j) for j in range(len(cols)))
            
            # Apply SMOTE
            smote = SMOTE(sampling_strategy=sampling_strategy, 
                         k_neighbors=min(k_neighbors, n_samples_original-1),
                         random_state=random_state)
            
            combined_data = np.hstack([Xtr_raw, ytr_raw])
            X_synthetic, labels_synthetic = smote.fit_resample(combined_data, combined_labels)
            
            # Extract synthetic features and targets
            X_synthetic_features = X_synthetic[:, :len(X_COLS)]
            y_synthetic_targets = X_synthetic[:, len(X_COLS):]
            
            # Combine original + synthetic
            Xtr_aug = np.vstack([Xtr_raw, X_synthetic_features])
            ytr_aug = np.vstack([ytr_raw, y_synthetic_targets])
            
        except Exception as e:
            print(f"SMOTE failed for {task}: {e}")
            print("Falling back to simple oversampling...")
            
            # Simple oversampling fallback
            n_synthetic = n_samples_original * 2
            synthetic_indices = np.random.choice(n_samples_original, n_synthetic, replace=True)
            
            X_synthetic_features = Xtr_raw[synthetic_indices]
            y_synthetic_targets = ytr_raw[synthetic_indices]
            
            # Add small noise
            noise_scale = 0.01
            X_noise = np.random.normal(0, noise_scale, X_synthetic_features.shape)
            y_noise = np.random.normal(0, noise_scale * 0.1, y_synthetic_targets.shape)
            
            X_synthetic_features = X_synthetic_features + X_noise
            y_synthetic_targets = y_synthetic_targets + y_noise
            
            # Normalize and clip
            X_synthetic_features = renormalize_compositions(X_synthetic_features)
            
            if task == 'optical':
                y_synthetic_targets = np.clip(y_synthetic_targets, 0.0, 100.0)
            if task == 'tensile':
                y_synthetic_targets[:, 0] = np.clip(y_synthetic_targets[:, 0], 0.0, None)
                y_synthetic_targets[:, 1] = np.clip(y_synthetic_targets[:, 1], 0.0, None)
            
            Xtr_aug = np.vstack([Xtr_raw, X_synthetic_features])
            ytr_aug = np.vstack([ytr_raw, y_synthetic_targets])
        
        # Standardize features
        scaler = StandardScaler()
        Xtr_aug_scaled = scaler.fit_transform(Xtr_aug)
        Xte_scaled = scaler.transform(Xte_raw)
        
        X_train_new[task] = Xtr_aug_scaled
        y_train_new[task] = ytr_aug
        X_test_new[task] = Xte_scaled
        y_test_new[task] = yte_raw
        
        print(f"  {task}: {len(Xtr_raw)} → {len(Xtr_aug)} samples")
    
    return X_train_new, y_train_new, X_test_new, y_test_new

def build_uip_augmented_splits_ratio(df_src, x_cols, target_cols, virtual_multiplier=100, n_folds=3):
    """
    Build UIP augmented splits for specific columns with given ratio.
    
    Args:
        df_src: Source dataframe
        x_cols: Input feature columns
        target_cols: Target columns
        virtual_multiplier: UIP augmentation multiplier
        n_folds: Number of folds for splitting
        
    Returns:
        Dictionary with augmented data
    """
    np.random.seed(RNG_SEED)
    X_raw = df_src[x_cols].values
    y_raw = df_src[target_cols].values
    
    # ADAPTIVE NOISE SCALING: Scale noise inversely with sqrt(multiplier)
    noise_scale_factor = 1.0 / np.sqrt(virtual_multiplier)
    scaled_comp_sigma = COMPOSITION_SIGMA * noise_scale_factor
    scaled_prop_sigma = PROPERTY_REL_SIGMA * noise_scale_factor
    
    # Create k-fold splits
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RNG_SEED)
    fold_data = []
    
    for train_idx, val_idx in kf.split(X_raw):
        X_tr = X_raw[train_idx]
        y_tr = y_raw[train_idx]
        X_val = X_raw[val_idx]
        y_val = y_raw[val_idx]
        
        # Apply UIP augmentation to training data
        n_samples = len(X_tr)
        virtual_samples = n_samples * virtual_multiplier
        
        # Virtual replication
        X_rep = np.repeat(X_tr, virtual_multiplier, axis=0)
        y_rep = np.repeat(y_tr, virtual_multiplier, axis=0)
        
        # Perturb compositions and renormalize (with adaptive noise)
        X_noise = np.random.normal(loc=0.0, scale=scaled_comp_sigma, size=X_rep.shape)
        X_virt = renormalize_compositions(X_rep + X_noise)
        
        # Perturb properties relatively (with adaptive noise)
        rel_noise = np.random.normal(loc=0.0, scale=scaled_prop_sigma, size=y_rep.shape)
        y_virt = y_rep * (1.0 + rel_noise)
        
        # Apply constraints
        if 'optical' in target_cols[0].lower():
            y_virt = np.clip(y_virt, 0.0, 100.0)
        if 'tensile' in target_cols[0].lower():
            y_virt[:, 0] = np.clip(y_virt[:, 0], 0.0, None)  # TensileStress
            if y_virt.shape[1] > 1:
                y_virt[:, 1] = np.clip(y_virt[:, 1], 0.0, None)  # TensileToughness
        
        # Combine augmented data
        X_augmented = np.vstack([X_tr, X_virt])
        y_augmented = np.vstack([y_tr, y_virt])
        
        fold_data.append({
            'X_train': X_augmented,
            'y_train': y_augmented,
            'X_val': X_val,
            'y_val': y_val
        })
    
    # Return flattened data for simple ensemble training
    all_X_train = np.vstack([fold['X_train'] for fold in fold_data])
    all_y_train = np.vstack([fold['y_train'] for fold in fold_data])
    
    return {
        'X_augmented': all_X_train,
        'y_augmented': all_y_train,
        'folds': fold_data
    }

def build_smote_augmented_splits_ratio(df_src, x_cols, target_cols, sampling_strategy=1.0, n_folds=3):
    """
    Build SMOTE augmented splits for specific columns with given ratio.
    
    Args:
        df_src: Source dataframe
        x_cols: Input feature columns
        target_cols: Target columns
        sampling_strategy: SMOTE sampling ratio
        n_folds: Number of folds for splitting
        
    Returns:
        Dictionary with augmented data
    """
    np.random.seed(RNG_SEED)
    X_raw = df_src[x_cols].values
    y_raw = df_src[target_cols].values
    
    # Create k-fold splits
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RNG_SEED)
    fold_data = []
    
    for train_idx, val_idx in kf.split(X_raw):
        X_tr = X_raw[train_idx]
        y_tr = y_raw[train_idx]
        X_val = X_raw[val_idx]
        y_val = y_raw[val_idx]
        
        n_samples_original = len(X_tr)
        
        try:
            # Create bins for target discretization
            target_bins = {}
            for i, col in enumerate(target_cols):
                target_values = y_tr[:, i]
                bins = np.percentile(target_values, [0, 20, 40, 60, 80, 100])
                target_bins[i] = np.digitize(target_values, bins) - 1
            
            # Create combined labels for SMOTE
            combined_labels = np.zeros(n_samples_original)
            for i in range(n_samples_original):
                label_parts = [target_bins[j][i] for j in range(len(target_cols))]
                combined_labels[i] = sum(label_parts[j] * (5**j) for j in range(len(target_cols)))
            
            # Apply SMOTE
            smote = SMOTE(sampling_strategy=sampling_strategy, 
                         k_neighbors=min(5, n_samples_original-1),
                         random_state=RNG_SEED)
            
            combined_data = np.hstack([X_tr, y_tr])
            X_synthetic, labels_synthetic = smote.fit_resample(combined_data, combined_labels)
            
            # Extract synthetic features and targets
            X_synthetic_features = X_synthetic[:, :len(x_cols)]
            y_synthetic_targets = X_synthetic[:, len(x_cols):]
            
            # Combine augmented data
            X_augmented = np.vstack([X_tr, X_synthetic_features])
            y_augmented = np.vstack([y_tr, y_synthetic_targets])
            
        except Exception as e:
            print(f"SMOTE failed: {e}, falling back to simple oversampling")
            
            # Simple oversampling fallback
            if sampling_strategy <= 1.0:
                n_synthetic = int(n_samples_original * sampling_strategy)
            else:
                n_synthetic = int(n_samples_original * (sampling_strategy / 100))
            
            synthetic_indices = np.random.choice(n_samples_original, n_synthetic, replace=True)
            
            X_synthetic_features = X_tr[synthetic_indices]
            y_synthetic_targets = y_tr[synthetic_indices]
            
            # Add small noise
            noise_scale = 0.01
            X_noise = np.random.normal(0, noise_scale, X_synthetic_features.shape)
            y_noise = np.random.normal(0, noise_scale * 0.1, y_synthetic_targets.shape)
            
            X_synthetic_features = X_synthetic_features + X_noise
            y_synthetic_targets = y_synthetic_targets + y_noise
            
            # Normalize and clip
            X_synthetic_features = renormalize_compositions(X_synthetic_features)
            
            if 'optical' in target_cols[0].lower():
                y_synthetic_targets = np.clip(y_synthetic_targets, 0.0, 100.0)
            if 'tensile' in target_cols[0].lower():
                y_synthetic_targets[:, 0] = np.clip(y_synthetic_targets[:, 0], 0.0, None)
                if y_synthetic_targets.shape[1] > 1:
                    y_synthetic_targets[:, 1] = np.clip(y_synthetic_targets[:, 1], 0.0, None)
            
            # Combine augmented data
            X_augmented = np.vstack([X_tr, X_synthetic_features])
            y_augmented = np.vstack([y_tr, y_synthetic_targets])
        
        fold_data.append({
            'X_train': X_augmented,
            'y_train': y_augmented,
            'X_val': X_val,
            'y_val': y_val
        })
    
    # Return flattened data for simple ensemble training
    all_X_train = np.vstack([fold['X_train'] for fold in fold_data])
    all_y_train = np.vstack([fold['y_train'] for fold in fold_data])
    
    return {
        'X_augmented': all_X_train,
        'y_augmented': all_y_train,
        'folds': fold_data
    }

print("✅ Section 4: Data Augmentation Methods Complete")
print(f"🔧 UIP parameters (base): comp_sigma={COMPOSITION_SIGMA}, prop_sigma={PROPERTY_REL_SIGMA}")
print("🔧 UIP uses HYBRID noise approach:")
print("   • Experimental std (when available) - most realistic!")
print("   • Adaptive relative noise (fallback) - scales with sqrt(ratio)")
print("   • ratio=1: 15% fallback, ratio=10: 4.7% fallback, ratio=100: 1.5% fallback")
if SMOGN_AVAILABLE:
    print("🔧 SMOTE-Regression (smogn): ENABLED ✅")
    print("   • Uses SMOTER for proper regression augmentation")
    print("   • Adaptive perturbation: scales with sqrt(ratio)")
    print("   • Focuses on extreme/rare values for better coverage")
else:
    print("🔧 SMOTE-Regression: Using fallback (smogn not available)")
    print("   • Install smogn for optimal SMOTE performance: pip install smogn")
print("🔧 Ratio-based functions: UIP and SMOTE ratio variants available")
