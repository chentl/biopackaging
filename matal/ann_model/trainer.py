import gc
import pickle

import lz4.frame
import torch
from torch import nn
from tqdm.auto import tqdm

from matal.settings import MODEL_DIR
from matal.utils import auto_log
from .loader import load_model, load_opt, find_last_checkpoint
from .metrics import calc_metrics
from .serializer import save_ckpt
from .settings import TARGET_Y_COLS


def train_model(model, train_batch_data, all_dataset,
                lr=1e-5, l2=1e-8, noise_std=0,
                model_dir=MODEL_DIR,
                split_opt=True,
                split_enc_opt=False,
                epochs=10000, ckpt_freq=1000, hist_freq=1000,
                restart=True,
                load_weights_from='',
                load_opt_from='',
                dtype=torch.float64,
                device=torch.device('cpu'),
                **kwargs):
    if split_enc_opt and (not split_opt):
        raise ValueError('split_enc_opt can not be True while split_opt is False.')

    if split_opt:
        if split_enc_opt:
            optimizer = {t: torch.optim.Adam(model.models[t].parameters(),
                                             lr=lr, weight_decay=l2) for t in TARGET_Y_COLS.keys()}
            optimizer['encoder'] = torch.optim.Adam(model.models['encoder'].parameters(),
                                                    lr=lr, weight_decay=l2)
            enc_opt = optimizer['encoder']
        else:
            optimizer = {t: torch.optim.Adam(
                list(model.models['encoder'].parameters()) + list(model.models[t].parameters()),
                lr=lr, weight_decay=l2) for t in TARGET_Y_COLS.keys()}
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    if load_weights_from:
        if ':' in load_weights_from:
            base_model_name, base_model_ckpt = load_weights_from.split(':')
            base_model = load_model(base_model_name, checkpoint=base_model_ckpt)
            model.load_state_dict(base_model.state_dict())
        else:
            with lz4.frame.open(load_weights_from, 'rb') as f:
                model.load_state_dict(torch.load(f))

    if load_opt_from:
        if ':' in load_opt_from:
            base_model_name, base_model_ckpt = load_opt_from.split(':')
            base_opt = load_opt(base_model_name, checkpoint=base_model_ckpt)
        else:
            with lz4.frame.open(load_weights_from, 'rb') as f:
                base_opt = torch.load(f)

        if split_opt:
            for k, d in base_opt.items():
                auto_log(f'Loading optimizer state from {k} in {load_opt_from}')
                optimizer[k].load_state_dict(d)
        else:
            auto_log(f'Loading optimizer state from {load_opt_from}')
            optimizer.load_state_dict(base_opt)

    last_checkpoint = None
    if restart:
        last_checkpoint = find_last_checkpoint(model_dir, with_opt=True)
        if last_checkpoint is not None:
            auto_log(f'Restart from checkpoint @ step {last_checkpoint}')
            model_fn = f'model.ckpt_{last_checkpoint}.pt.lz4'
            opt_fn = f'opt.ckpt_{last_checkpoint}.pt.lz4'
            hist_fn = f'hist.ckpt.pk.lz4'

            with lz4.frame.open(model_dir / model_fn, 'rb') as f:
                auto_log(f'Loading model state from {model_fn}')
                model.load_state_dict(torch.load(f))

            with lz4.frame.open(model_dir / opt_fn, 'rb') as f:
                saved_opt = torch.load(f)

            if split_opt:
                for k, d in saved_opt.items():
                    auto_log(f'Loading optimizer state from {k} in {opt_fn}')
                    optimizer[k].load_state_dict(d)
            else:
                auto_log(f'Loading optimizer state from {opt_fn}')
                optimizer.load_state_dict(saved_opt)

            with lz4.frame.open(model_dir / hist_fn, 'rb') as f:
                auto_log(f'Loading history from {hist_fn}')
                saved_hist = pickle.load(f)
                prev_hist_len = len([s for s in saved_hist['step'] if s <= last_checkpoint])
                saved_hist = {k: h[:prev_hist_len] for k, h in saved_hist.items()}

    criterion_lut = dict(
        grade=nn.BCEWithLogitsLoss(),
        tensile=nn.MSELoss(),
        optical=nn.MSELoss(),
        fire=nn.MSELoss(),
    )

    targets = list(TARGET_Y_COLS.keys())
    batch_ti_arr = train_batch_data['batch_ti_arr']
    batch_X_arr = [torch.as_tensor(x, device=device, dtype=dtype) for x in tqdm(train_batch_data['batch_X_arr'])]
    del train_batch_data['batch_X_arr']
    gc.collect()
    batch_y_arr = [torch.as_tensor(y, device=device, dtype=dtype) for y in tqdm(train_batch_data['batch_y_arr'])]
    del train_batch_data
    gc.collect()

    n_batches = len(batch_ti_arr)
    n_steps = n_batches * epochs

    def train_pass(step_i):
        batch_i = step_i % n_batches
        target = targets[batch_ti_arr[batch_i]]
        X = batch_X_arr[batch_i]
        y = batch_y_arr[batch_i]

        opt = optimizer[target] if split_opt else optimizer
        opt.zero_grad()
        if split_enc_opt:
            enc_opt.zero_grad()

        X = (X + (torch.rand_like(X) - 0.5) * noise_std)
        outputs = model.forward(X, target)

        loss = criterion_lut[target](outputs, y)
        loss.backward()
        opt.step()
        if split_enc_opt:
            enc_opt.step()

        return target, loss.item()

    step_i = last_checkpoint if (restart and last_checkpoint) else 0
    pbar = tqdm(total=n_steps - step_i, unit='step')
    running_loss = {t: 0.0 for t in TARGET_Y_COLS.keys()}
    running_loss_i = {t: 0 for t in TARGET_Y_COLS.keys()}
    if restart and last_checkpoint:
        history = saved_hist
    else:
        history = {'epoch': [], 'step': [], }

    while step_i < n_steps:
        target, loss = train_pass(step_i)
        running_loss[target] += loss
        running_loss_i[target] += 1

        pbar.update(1)
        step_i += 1

        if (step_i % hist_freq == 0) or (step_i == n_steps):
            history['epoch'].append(step_i // n_batches)
            history['step'].append(step_i)
            metrics = calc_metrics(model, all_dataset, running_loss, running_loss_i, device=device, dtype=dtype, )
            for m, v in metrics.items():
                if m in history:
                    history[m].append(v)
                else:
                    history[m] = [v]
            pbar.set_postfix(**{m: v for m, v in metrics.items() if m.startswith('loss')})

            running_loss = {t: 0.0 for t in TARGET_Y_COLS.keys()}
            running_loss_i = {t: 0 for t in TARGET_Y_COLS.keys()}

        if (step_i % ckpt_freq == 0) or (step_i == n_steps):
            save_ckpt(step_i, model, optimizer, history, model_dir, split_opt=split_opt)

    return optimizer, history
