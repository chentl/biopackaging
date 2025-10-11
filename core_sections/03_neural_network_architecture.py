# ================================
# NEURAL NETWORK ARCHITECTURE
# ================================
"""
Configurable Artificial Neural Network architecture aligned with ann_model/base.py.
Supports various activation functions, dropout, and weight initialization.

NOTE: This module depends on X_COLS from section 01.
Load section 01 before loading this section.
"""
import torch
import torch.nn as nn

# X_COLS should be imported from section 01_imports_and_configuration
# If running standalone, define it here:
try:
    from core_sections import X_COLS
except ImportError:
    try:
        X_COLS  # Check if already defined
    except NameError:
        # Fallback definition
        X_COLS = ['LAP', 'MMT', 'CMC', 'CNF', 'SLK',
                  'AGR', 'ALG', 'CAR', 'CHS', 'PEC', 'PUL', 'STA', 'GEL', 'GLU', 'ZIN',
                  'GLY', 'FFA', 'LAC', 'LEV', 'PHA', 'SRB', 'SUA', 'XYL']

# Get available activations from PyTorch
available_activations = torch.nn.modules.activation.__all__
activation_name_lut = {n.lower(): getattr(torch.nn.modules.activation, n) for n in available_activations}

def get_activation(name):
    """Get activation function by name (case-insensitive)."""
    if callable(name):
        return name
    elif name.lower() in activation_name_lut:
        return activation_name_lut[name.lower()]
    elif name == 'identity':
        return nn.Identity
    else:
        raise ValueError(f"Unknown activation: {name}")


class ANNModel(nn.Module):
    """
    Artificial Neural Network model matching ann_model/base.py.
    
    Features:
    - Flexible hidden layer architecture
    - Multiple activation functions (all PyTorch activations)
    - Input, hidden, and output dropout
    - Weight initialization support
    - return_after parameter for intermediate outputs
    """
    
    def __init__(self, hidden_layer_sizes=(100,), input_dim=3,
                 activation='relu', output_activation='relu',
                 name=None, dtype=torch.float64, weight_init=None,
                 input_dropout=0.0, output_dropout=0.0, hidden_dropout=0.0):
        super().__init__()
        self.name = name
        self.dtype = dtype

        _last_dim = input_dim
        self.layers = nn.ModuleDict()

        if input_dropout:
            self.layers['inp_dropout'] = nn.Dropout(input_dropout)

        for i, dim in enumerate(hidden_layer_sizes, 1):
            layer_name = f'fc{i}'
            layer = nn.Linear(_last_dim, dim, dtype=self.dtype)
            if weight_init is not None and weight_init != 'none':
                getattr(torch.nn.init, weight_init)(layer.weight)
                torch.nn.init.zeros_(layer.bias)
            self.layers[layer_name] = layer

            _last_dim = dim

            act_module = None
            if i != len(hidden_layer_sizes):
                if activation and activation != 'none':
                    act_module = get_activation(activation)
            else:
                if output_activation and output_activation != 'none':
                    act_module = get_activation(output_activation)

            if act_module:
                layer_name = f'fc{i}_act'
                layer = act_module()
                self.layers[layer_name] = layer

            if hidden_dropout and (i != len(hidden_layer_sizes)):
                self.layers[f'fc{i}_dropout'] = nn.Dropout(hidden_dropout)

        if output_dropout:
            self.layers[f'out_dropout'] = nn.Dropout(output_dropout)

    def forward(self, x, return_after=None):
        """Forward pass through the network."""
        for name, l in self.layers.items():
            x = l(x)
            if name == return_after:
                break
        return x


# For backward compatibility, keep ConfigurableANN as alias
class ConfigurableANN(ANNModel):
    """Alias for ANNModel for backward compatibility."""
    def __init__(self, input_dim=23, hidden_layers=(64, 32), 
                 activation='relu', dropout=0.1, use_batch_norm=False, dtype=torch.float32):
        # Map old interface to new interface
        super().__init__(
            hidden_layer_sizes=hidden_layers,
            input_dim=input_dim,
            activation=activation,
            output_activation=activation,
            dtype=dtype,
            hidden_dropout=dropout,
            weight_init='kaiming_uniform_'
        )
        self.output_dim = hidden_layers[-1] if hidden_layers else input_dim
        self.use_batch_norm = use_batch_norm

# Test the architecture
test_model = ConfigurableANN(input_dim=len(X_COLS), hidden_layers=(128, 64), dropout=0.1)
total_params = sum(p.numel() for p in test_model.parameters())

print("✅ Neural Network Architecture Created (aligned with ann_model/base.py)")
print(f"📏 Input dimension: {len(X_COLS)}")
print(f"🔗 Hidden output dimension: {test_model.output_dim}")
print(f"⚙️ Total parameters: {total_params:,}")
print(f"💾 Model size: ~{total_params * 4 / 1024:.1f} KB")

def validate_neural_network():
    """Validate neural network architecture and provide summary."""
    print("\n✅ [03] Neural Network Architecture Module Loaded")
    print(f"🏗️ Architecture Summary (ANNModel from ann_model/base.py):")
    print(f"   📏 Input features: {len(X_COLS)}")
    print(f"   🔗 Output dimensions: {test_model.output_dim}")
    print(f"   ⚙️ Total parameters: {total_params:,}")
    print(f"   💾 Memory footprint: ~{total_params * 4 / 1024:.1f} KB")
    
    print(f"🔧 Key Features:")
    print(f"   🏗️ ModuleDict-based layers: {'✅'}")
    has_dropout = any('dropout' in name for name in test_model.layers.keys())
    print(f"   🎲 Dropout support: {'✅' if has_dropout else '⚠️'}")
    print(f"   🎯 Weight initialization: kaiming_uniform_")
    print(f"   🔄 All PyTorch activations supported")
    
    return test_model

# Auto-validate on import
if __name__ == "__main__" or __name__ == "__file__":
    validate_neural_network()
