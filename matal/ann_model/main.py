import argparse
import json
import os
import pickle
import sys

import dill
import lz4.frame
import pandas as pd
import torch

from matal.settings import MODEL_DIR, DATA_DIR
from matal.utils import auto_log, obj_to_hash, Bunch, RobustJSONEncoder
from matal.ann_model.builder import build_model
from matal.ann_model.config import ModelConfig, FitConfig
from matal.ann_model.plot import plot_hist_grade, plot_hist_perf
from matal.ann_model.serializer import DUMP_EXT, DUMP_OPEN, DUMP_KWARGS
from matal.ann_model.trainer import train_model


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(cmd_args=None):
    parser = argparse.ArgumentParser()
    model_config = ModelConfig()
    fit_config = FitConfig()

    for config in [model_config, fit_config]:
        for key, value in config._DEFAULTS.items():
            if type(value) in [str, int, float]:
                parser.add_argument(f'--{key}', type=type(value), nargs='?', default=value)
            elif type(value) in [bool]:
                parser.add_argument(f'--{key}', type=str2bool, nargs='?', default=value)
            else:
                auto_log(f'Ignoring {key} (type {type(value)}) in argparse')

    parser.add_argument(f'--data_name', type=str, nargs='?', default='20230405_t1_B64xN32768')
    if cmd_args is None:
        cmd_args = sys.argv[1:]
    parser.add_argument(f'--device', type=str, nargs='?', default='cpu')
    parser.add_argument(f'--model_prefix', type=str, nargs='?', default='pt_db')

    args = parser.parse_args(cmd_args)

    fit_config.update(vars(args))
    model_config.update(vars(args))
    data_name = args.data_name
    model_prefix = args.model_prefix

    with lz4.frame.open(DATA_DIR / 'batch_cache' / f'{data_name}.batch.pk.lz4', 'rb') as f:
        train_batch_data = dill.load(f)

    with lz4.frame.open(DATA_DIR / 'batch_cache' / f'{data_name}.all_data.pk.lz4', 'rb') as f:
        all_dataset = dill.load(f)

    with open(DATA_DIR / 'batch_cache' / f'{data_name}.config.json', 'r') as f:
        data_config = Bunch(json.load(f))

    model_hash = model_config.get_hash()
    train_hash = fit_config.get_hash()
    instance_hash = obj_to_hash(dict(data_name=data_name, model_hash=model_hash, train_hash=train_hash), size=10)

    model_name = f'{model_prefix}.{data_name}.{instance_hash}'
    model_dir = MODEL_DIR / \
                f'{model_prefix}.{data_name}' / \
                f'model-{model_hash}.bn-{model_config.encoder__hidden_layers}-{model_config.encoder__n_output}' / \
                f'train-{train_hash}' / \
                model_name
    os.makedirs(model_dir, exist_ok=True)

    save_params = Bunch(
        model_name=model_name,
        model_dir=model_dir,
    )

    dump_info = dict(
        cmd_args=cmd_args,
        data_name=data_name,
        model_hash=model_hash,
        model_params=model_config.get_all(),
        train_hash=train_hash,
        fit_config=fit_config.get_all(),
        save_params=save_params,
        parsed_args=vars(args),
        data_config=data_config,
    )
    with open(save_params.model_dir / f'{save_params.model_name}.param.pk', 'wb') as f:
        pickle.dump(dump_info, f)
    with open(save_params.model_dir / f'{save_params.model_name}.param.json', 'w') as f:
        json.dump(dump_info, f, cls=RobustJSONEncoder, indent=2)

    auto_log(f'Building model {save_params.model_name} with '
             f'model_config={model_config.get_all()}, '
             f'fit_config={fit_config.get_all()}.')

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype_str)
    model = build_model(model_name=save_params.model_name, dtype=dtype, **model_config.get_all()).to(device=device)
    print(model)

    optimizer, history = train_model(model, train_batch_data, all_dataset,
                                     dtype=dtype, device=device,
                                     **fit_config.get_all(), **save_params)

    with DUMP_OPEN(save_params.model_dir / f'{save_params.model_name}.model.pt{DUMP_EXT}', **DUMP_KWARGS) as fp:
        torch.save(model.state_dict(), fp)
    with DUMP_OPEN(save_params.model_dir / f'{save_params.model_name}.opt.pt{DUMP_EXT}', **DUMP_KWARGS) as fp:
        torch.save(
            {t: opt.state_dict() for t, opt in optimizer.items()} if fit_config.split_opt else optimizer.state_dict(),
            fp)

    with DUMP_OPEN(save_params.model_dir / f'{save_params.model_name}.hist.pk{DUMP_EXT}', **DUMP_KWARGS) as fp:
        pickle.dump(history, fp)
    hist_df = pd.DataFrame.from_records(history, columns=list(history.keys()))

    hist_df.to_csv(save_params.model_dir / f'{save_params.model_name}.hist.csv', index=False)

    plot_hist_grade(history, save_name=save_params.model_name, save_dir=save_params.model_dir, close=True)
    plot_hist_perf(history, save_name=save_params.model_name, save_dir=save_params.model_dir, close=True)
    return save_params.model_name


if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('Agg')
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    main()
