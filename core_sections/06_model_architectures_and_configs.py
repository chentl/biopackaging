# ================================
# SECTION 6: MODEL ARCHITECTURES AND HYPERPARAMETER CONFIGURATIONS
# ================================
"""
Model architecture configurations aligned with ann_model/config.py.
Provides ModelConfig and FitConfig classes with defaults for the ensemble pipeline.
"""

import torch
import torch.nn as nn


class BunchWithDefaults(dict):
    """
    Dictionary with defaults and hash support (simplified from ann_model/config.py).
    """
    _DEFAULTS = {}
    _REJECT_NOT_IN_DEFAULT = False
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_all(self):
        """Get all parameters including defaults."""
        content = dict(**self._DEFAULTS)
        content.update(self)
        return content
    
    def __getattr__(self, key):
        if key in self:
            return self[key]
        elif key in self._DEFAULTS:
            return self._DEFAULTS[key]
        else:
            raise AttributeError(key)
    
    def __setitem__(self, key, value):
        if self._REJECT_NOT_IN_DEFAULT and key not in self._DEFAULTS:
            return
        dict.__setitem__(self, key, value)


class ModelConfig(BunchWithDefaults):
    """
    Model configuration aligned with ann_model/config.py ModelConfig.
    Defines architecture parameters for encoder and task-specific heads.
    """
    _DEFAULTS = dict(
        encoder__hidden_layers=12,
        encoder__hidden_base=2.0,
        encoder__hidden_scale='linear',
        encoder__n_output=8,
        encoder__act_hidden='elu',
        encoder__act_output='identity',
        encoder__weight_init='kaiming_uniform_',
        encoder__input_dropout=0.0,
        encoder__output_dropout=0.0,
        encoder__hidden_dropout=0.0,
        
        grade__hidden_layers=4,
        grade__hidden_base=2.0,
        grade__hidden_scale='linear',
        grade__n_output=3,  # len(TARGET_Y_COLS['grade'])
        grade__act_hidden='elu',
        grade__act_output='sigmoid',
        grade__weight_init='kaiming_uniform_',
        grade__input_dropout=0.0,
        grade__output_dropout=0.0,
        grade__hidden_dropout=0.0,
        
        tensile__hidden_layers=4,
        tensile__hidden_base=2.0,
        tensile__hidden_scale='linear',
        tensile__n_output=4,  # len(TARGET_Y_COLS['tensile'])
        tensile__act_hidden='elu',
        tensile__act_output='relu',
        tensile__weight_init='kaiming_uniform_',
        tensile__input_dropout=0.0,
        tensile__output_dropout=0.0,
        tensile__hidden_dropout=0.0,
        
        optical__hidden_layers=4,
        optical__hidden_base=2.0,
        optical__hidden_scale='linear',
        optical__n_output=3,  # len(TARGET_Y_COLS['optical'])
        optical__act_hidden='elu',
        optical__act_output='sigmoid',  # Sigmoid for [0,1] range after scaling by [300,100,100]
        optical__weight_init='kaiming_uniform_',
        optical__input_dropout=0.0,
        optical__output_dropout=0.0,
        optical__hidden_dropout=0.0,
        
        fire__hidden_layers=4,
        fire__hidden_base=2.0,
        fire__hidden_scale='linear',
        fire__n_output=1,  # len(TARGET_Y_COLS['fire'])
        fire__act_hidden='elu',
        fire__act_output='sigmoid',
        fire__weight_init='kaiming_uniform_',
        fire__input_dropout=0.0,
        fire__output_dropout=0.0,
        fire__hidden_dropout=0.0,
        
        random_seed=0,
        dtype_str='float64',
    )
    _REJECT_NOT_IN_DEFAULT = True


class FitConfig(BunchWithDefaults):
    """
    Training configuration aligned with ann_model/config.py FitConfig.
    Defines hyperparameters for model training.
    """
    _DEFAULTS = dict(
        epochs=10,
        lr=1e-5,
        l2=1e-8,
        noise_std=1 / 200,
        ckpt_freq=100000,
        hist_freq=5000,
        split_opt=True,
        split_enc_opt=False,
        restart=True,
        load_weights_from='',
        load_opt_from='',
    )
    _REJECT_NOT_IN_DEFAULT = True

class EnhancedConfigurableANN(nn.Module):
    """
    Enhanced configurable ANN with scalable architectures.
    Simplified design based on head_only_training.ipynb findings.
    """
    
    def __init__(self, input_dim=23, hidden_layers=(64, 32), 
                 activation='elu', dropout=0.1, use_batch_norm=False, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.use_batch_norm = use_batch_norm
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization (optional)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation function
            activation_map = {
                'relu': nn.ReLU(),
                'elu': nn.ELU(),
                'tanh': nn.Tanh(),
                'leaky_relu': nn.LeakyReLU(0.1),
                'selu': nn.SELU(),
                'gelu': nn.GELU(),
                'prelu': nn.PReLU()
            }
            layers.append(activation_map[activation])
            
            # Dropout regularization (optional)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        self.layers = nn.Sequential(*layers)
        self.output_dim = prev_dim
        
    def forward(self, x):
        """Forward pass through the network."""
        return self.layers(x)

# SCALED architecture configurations - complexity increases with augmentation ratio
# More data allows larger models without overfitting
ARCHITECTURE_CONFIGS = {
    'simple': {
        'hidden_layers': (64, 32),
        'dropout': 0.1,
        'use_batch_norm': False,
        'description': 'Compact model for low augmentation (1:1) - 546 samples'
    },
    'medium': {
        'hidden_layers': (128, 64, 32),
        'dropout': 0.15,
        'use_batch_norm': True,
        'description': 'Medium model for moderate augmentation (1:10) - 3K samples'
    },
    'complex': {
        'hidden_layers': (256, 128, 64, 32),
        'dropout': 0.2,
        'use_batch_norm': True,
        'description': 'Large model for high augmentation (1:100) - 27K samples'
    }
}

# SIMPLIFIED hyperparameter sets (matching head_only_training.ipynb)
# Using ELU activation consistently and simpler configs for better generalization
HYPERPARAMETER_SETS = {
    'simple': [
        {
            'learning_rate': 1e-3,
            'batch_size': 32,
            'epochs': 100,
            'weight_decay': 1e-5,
            'activation': 'elu',
            'dropout': 0.1,
            'use_batch_norm': False
        },
        {
            'learning_rate': 8e-4,
            'batch_size': 32,
            'epochs': 120,
            'weight_decay': 1e-5,
            'activation': 'elu',
            'dropout': 0.12,
            'use_batch_norm': False
        },
        {
            'learning_rate': 1.2e-3,
            'batch_size': 32,
            'epochs': 90,
            'weight_decay': 1e-5,
            'activation': 'elu',
            'dropout': 0.08,
            'use_batch_norm': False
        },
        {
            'learning_rate': 9e-4,
            'batch_size': 32,
            'epochs': 110,
            'weight_decay': 1e-5,
            'activation': 'elu',
            'dropout': 0.1,
            'use_batch_norm': False
        },
        {
            'learning_rate': 1.1e-3,
            'batch_size': 32,
            'epochs': 100,
            'weight_decay': 1e-5,
            'activation': 'elu',
            'dropout': 0.1,
            'use_batch_norm': False
        }
    ],
    'medium': [
        {
            'learning_rate': 8e-4,
            'batch_size': 64,
            'epochs': 150,
            'weight_decay': 1e-5,
            'activation': 'elu',
            'dropout': 0.15,
            'use_batch_norm': True
        },
        {
            'learning_rate': 6e-4,
            'batch_size': 64,
            'epochs': 180,
            'weight_decay': 1e-5,
            'activation': 'elu',
            'dropout': 0.18,
            'use_batch_norm': True
        },
        {
            'learning_rate': 1e-3,
            'batch_size': 64,
            'epochs': 120,
            'weight_decay': 1e-5,
            'activation': 'elu',
            'dropout': 0.12,
            'use_batch_norm': True
        },
        {
            'learning_rate': 7e-4,
            'batch_size': 64,
            'epochs': 160,
            'weight_decay': 1e-5,
            'activation': 'elu',
            'dropout': 0.15,
            'use_batch_norm': True
        },
        {
            'learning_rate': 9e-4,
            'batch_size': 64,
            'epochs': 140,
            'weight_decay': 1e-5,
            'activation': 'elu',
            'dropout': 0.15,
            'use_batch_norm': True
        }
    ],
    'complex': [
        {
            'learning_rate': 5e-4,
            'batch_size': 128,
            'epochs': 200,
            'weight_decay': 1e-5,
            'activation': 'elu',
            'dropout': 0.2,
            'use_batch_norm': True
        },
        {
            'learning_rate': 4e-4,
            'batch_size': 128,
            'epochs': 250,
            'weight_decay': 1e-5,
            'activation': 'elu',
            'dropout': 0.22,
            'use_batch_norm': True
        },
        {
            'learning_rate': 7e-4,
            'batch_size': 128,
            'epochs': 180,
            'weight_decay': 1e-5,
            'activation': 'elu',
            'dropout': 0.18,
            'use_batch_norm': True
        },
        {
            'learning_rate': 6e-4,
            'batch_size': 128,
            'epochs': 220,
            'weight_decay': 1e-5,
            'activation': 'elu',
            'dropout': 0.2,
            'use_batch_norm': True
        },
        {
            'learning_rate': 5.5e-4,
            'batch_size': 128,
            'epochs': 200,
            'weight_decay': 1e-5,
            'activation': 'elu',
            'dropout': 0.2,
            'use_batch_norm': True
        }
    ]
}

print("✅ [06] Model Architectures and Hyperparameter Configurations Loaded")
print("   (Aligned with ann_model/config.py)")
print(f"📊 Config Classes:")
print(f"   ✓ ModelConfig: Encoder + task-specific heads (grade, tensile, optical, fire)")
print(f"   ✓ FitConfig: Training hyperparameters")
print(f"📊 Architecture configurations: {len(ARCHITECTURE_CONFIGS)}")
print(f"🔧 Hyperparameter sets per architecture: 5")
print("🎯 SCALED Architecture Strategy:")
print("   ✓ Simple (1:1):   (64, 32) - batch=32, epochs=100, dropout=0.1")
print("   ✓ Medium (1:10):  (128, 64, 32) - batch=64, epochs=150, dropout=0.15 + BatchNorm")
print("   ✓ Complex (1:100): (256, 128, 64, 32) - batch=128, epochs=200, dropout=0.2 + BatchNorm")
print("   ✓ More data → Larger models → Better capacity without overfitting")

