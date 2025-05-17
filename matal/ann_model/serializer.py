import lz4.frame
import pickle
import os
import torch


DUMP_OPEN = lz4.frame.open
DUMP_EXT = '.lz4'
DUMP_KWARGS = dict(mode='wb')


def save_ckpt(step_i, model, optimizer, history, model_dir, split_opt=False):
    with DUMP_OPEN(model_dir / f'model.ckpt_{step_i}.pt{DUMP_EXT}', **DUMP_KWARGS) as fp:
        torch.save(model.state_dict(), fp)
    with DUMP_OPEN(model_dir / f'opt.ckpt_{step_i}.pt{DUMP_EXT}', **DUMP_KWARGS) as fp:
        torch.save({t: opt.state_dict() for t, opt in optimizer.items()} if split_opt else optimizer.state_dict(),
                   fp)

    new_hist_ckpt_f = model_dir / f'hist.ckpt.pk{DUMP_EXT}'
    old_hist_ckpt_f = model_dir / f'hist.ckpt_old.pk{DUMP_EXT}'
    if os.path.isfile(new_hist_ckpt_f):
        if os.path.isfile(old_hist_ckpt_f):
            os.remove(old_hist_ckpt_f)
        os.rename(new_hist_ckpt_f, old_hist_ckpt_f)
    with DUMP_OPEN(new_hist_ckpt_f, **DUMP_KWARGS) as fp:
        pickle.dump(history, fp)
