import pandas as pd
import torch
from torch import nn

available_activations = torch.nn.modules.activation.__all__
activation_name_lut = {n.lower(): getattr(torch.nn.modules.activation, n) for n in available_activations}


def get_activation(name):
    if callable(name):
        return name
    elif name.lower() in activation_name_lut:
        return activation_name_lut[name.lower()]
    elif name == 'identity':
        return nn.Identity
    else:
        raise ValueError()


class ANNModel(nn.Module):
    def __init__(self, hidden_layer_sizes=(100,), input_dim=3,
                 activation='relu', output_activation='relu',
                 name=None, dtype=torch.float64, weight_init=None,
                 input_dropout=0.0, output_dropout=0.0, hidden_dropout=0.0,
                 ):
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
        for name, l in self.layers.items():
            x = l(x)
            if name == return_after:
                break
        return x


class StackedModel(nn.Module):
    def __init__(self, models, stack_rules, name=None):
        super().__init__()
        self.name = name
        self.models = nn.ModuleDict(models)
        self.stack_rules = stack_rules

    def _forward_by_rule(self, x, rule, **kwargs):
        for r in rule:
            x = self.models[r](x, **kwargs)
        return x

    def forward(self, x, output_name, **kwargs):
        return self._forward_by_rule(x, self.stack_rules[output_name], **kwargs)

    def forward_all(self, x, return_df=False, **kwargs):
        output = {}
        for output_name, rule in self.stack_rules.items():
            output[output_name] = self._forward_by_rule(x, rule, **kwargs)

        if return_df:
            return pd.DataFrame.from_records({k: v.detach().view(-1).numpy() for k, v in output.items()})
        else:
            return output

    def predict(self, *args, **kwargs):
        with torch.no_grad():
            return self.forward(*args, **kwargs)

    def predict_all(self, *args, **kwargs):
        with torch.no_grad():
            return self.forward_all(*args, **kwargs)
