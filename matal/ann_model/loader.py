import os
import pickle
from pathlib import Path

import lz4.frame
import pandas as pd
import torch

from matal.settings import MODEL_DIR
from .builder import build_model
from .config import ModelConfig


def find_model_path(model_tag):
    matches = list(MODEL_DIR.glob(f'**/{model_tag}'))
    if len(matches) == 1:
        return os.path.relpath(matches[0], MODEL_DIR)
    else:
        return None


def find_all_checkpoints(path, with_opt=False):
    model_cps = sorted(
        [int(os.path.basename(p).split('.')[1].split('_')[1]) for p in Path(path).glob('model.ckpt_*.pt.lz4')])
    if with_opt:
        opt_cps = sorted(
            [int(os.path.basename(p).split('.')[1].split('_')[1]) for p in Path(path).glob('opt.ckpt_*.pt.lz4')])
        return sorted(list(set(model_cps) & set(opt_cps)))
    else:
        return model_cps


def find_last_checkpoint(path, filter_fn=None, with_opt=False):
    cps = find_all_checkpoints(path, with_opt=with_opt)

    if filter_fn:
        cps = [c for c in cps if filter_fn(c, all_cps=cps)]

    if len(cps):
        return cps[-1]
    else:
        return None


def load_history(model_tag, checkpoint='final'):
    path = find_model_path(model_tag)
    if path is None:
        return None

    if checkpoint == 'final':
        fn = f'{model_tag}.hist.pk.lz4'
    elif checkpoint == 'last':
        fn = f'hist.ckpt.pk.lz4'
    elif checkpoint == 'old':
        fn = f'hist.ckpt_old.pk.lz4'
    else:
        fn = f'hist.ckpt.pk.lz4'

    with lz4.frame.open(MODEL_DIR / path / fn, 'rb') as f:
        hist = pickle.load(f)

    if isinstance(checkpoint, int):
        prev_hist_len = len([s for s in hist['step'] if s <= checkpoint])
        hist = {k: h[:prev_hist_len] for k, h in hist.items()}

    hist_df = pd.DataFrame.from_records(hist)
    return hist_df


def load_params(model_tag):
    path = find_model_path(model_tag)
    if path is None:
        return None

    with open(MODEL_DIR / path / f'{model_tag}.param.pk', 'rb') as f:
        param = pickle.load(f)

    return param


def load_model(model_tag, checkpoint='final'):
    path = find_model_path(model_tag)
    if path is None:
        return None

    param = load_params(model_tag)
    model_config = ModelConfig(**param['model_params'])
    model = build_model(model_name=model_tag, **model_config.get_all())

    if checkpoint == 'final':
        fn = f'{model_tag}.model.pt.lz4'
    elif checkpoint == 'last':
        fn = f'model.ckpt_{find_last_checkpoint(MODEL_DIR / path)}.pt.lz4'
    else:
        fn = f'model.ckpt_{checkpoint}.pt.lz4'
    with lz4.frame.open(MODEL_DIR / path / fn, 'rb') as f:
        model.load_state_dict(torch.load(f, weights_only=True))
    model.eval()
    return model


def load_opt(model_tag, checkpoint='final'):
    path = find_model_path(model_tag)
    if path is None:
        return None

    if checkpoint == 'final':
        fn = f'{model_tag}.opt.pt.lz4'
    elif checkpoint == 'last':
        fn = f'opt.ckpt_{find_last_checkpoint(MODEL_DIR / path)}.pt.lz4'
    else:
        fn = f'opt.ckpt_{checkpoint}.pt.lz4'
    with lz4.frame.open(MODEL_DIR / path / fn, 'rb') as f:
        return torch.load(f)
