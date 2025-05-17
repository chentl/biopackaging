import numpy as np
import pandas as pd
import torch
from matal.ann_model.settings import X_COLS, TARGET_Y_COLS, TARGET_Y_SCALES
from matal.ann_model.loader import load_model


MODEL_TAGS = [
    f"pt_db.v3hp.RE5C",
    f"pt_db.v3hp.VPLV",
    f"pt_db.v3hp.PFFZ",
    f"pt_db.v3hp.CPUT",
    f"pt_db.v3hp.2HNP",
]


MODELS = {tag: load_model(tag) for tag in MODEL_TAGS}


def gen_pred_df(comp, targets=None):
    mat_comp = np.array(comp)
    mat_comp = mat_comp / mat_comp.sum(axis=1).reshape(-1, 1)
    c_ts = torch.as_tensor(mat_comp)
    preds = {}
    targets = targets or TARGET_Y_COLS.keys()
    with torch.no_grad():
        enc = {k: model.models['encoder'](c_ts) for k, model in MODELS.items()}
        for target in targets:
            cols = TARGET_Y_COLS[target]
            std_cols = [f'{c}_STD' for c in cols]
            pred_cols = [f'{c}' for c in cols]
            pred = [model.models[target](enc[k]).numpy().astype(np.float64) for k, model in MODELS.items()]
            pred_mean = np.mean(pred, axis=0) * TARGET_Y_SCALES[target]
            pred_std = np.std(pred, axis=0) * TARGET_Y_SCALES[target]
            preds[target] = pd.DataFrame(pred_mean, columns=pred_cols).join(pd.DataFrame(pred_std, columns=std_cols))

    pred_df = pd.DataFrame(mat_comp, columns=X_COLS)
    for target, df in preds.items():
        pred_df = pred_df.join(df)

    return pred_df
