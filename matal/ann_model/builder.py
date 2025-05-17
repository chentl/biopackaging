import random

import numpy as np
import torch

from matal.utils import auto_log, Bunch
from .base import ANNModel, StackedModel
from .settings import TARGET_Y_COLS, X_COLS


def build_model(dtype=torch.float64, model_name='model', random_seed=0, **kwargs):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.use_deterministic_algorithms(True)

    models = dict()

    for part in ['encoder'] + list(TARGET_Y_COLS.keys()):

        part_params = Bunch(**{k[len(f'{part}__'):]: v for k, v in kwargs.items() if k.startswith(f'{part}__')})

        if part_params.hidden_scale == 'power':
            hidden_layer_sizes = [round(part_params.hidden_base ** i * part_params.n_output) for i in
                                  reversed(range(part_params.hidden_layers + 1))]
        elif part_params.hidden_scale == 'linear':
            hidden_layer_sizes = [round(part_params.hidden_base * i + part_params.n_output) for i in
                                  reversed(range(part_params.hidden_layers + 1))]
        elif part_params.hidden_scale.startswith('log_'):
            log_base = float(part_params.hidden_scale.split('_')[1])

            if log_base >= 1:
                b = (-log_base + log_base ** (
                        part_params.hidden_base / part_params.n_output)) / part_params.hidden_layers
            else:
                b = (-log_base + log_base ** (part_params.hidden_base / part_params.n_output)) / (
                        part_params.hidden_layers - 1)

            hidden_layer_sizes = [round(part_params.hidden_base)] + [
                round(part_params.n_output * np.emath.logn(log_base, log_base + b * i)) for i in
                reversed(range(part_params.hidden_layers - 1))]
        else:
            raise ValueError(part_params.hidden_scale)
        auto_log(f'hidden_layer_sizes = {hidden_layer_sizes}', level='debug')
        models[part] = ANNModel(hidden_layer_sizes=hidden_layer_sizes,
                                input_dim=len(X_COLS) if part == 'encoder' else kwargs['encoder__n_output'], name=part,
                                activation=part_params.act_hidden,
                                output_activation=part_params.act_output,
                                weight_init=part_params.weight_init,
                                input_dropout=part_params.input_dropout,
                                output_dropout=part_params.output_dropout,
                                hidden_dropout=part_params.hidden_dropout,
                                dtype=dtype)

    rules = dict()
    rules.update({'encoder': ('encoder',)})
    rules.update({p: ('encoder', p) for p in TARGET_Y_COLS.keys()})
    rules.update({'_' + p: (p,) for p in TARGET_Y_COLS.keys()})

    model = StackedModel(models, rules, name=model_name)
    return model
