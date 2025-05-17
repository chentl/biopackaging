
import torch
from .settings import TARGET_Y_SCALES, TARGET_Y_COLS
from sklearn.metrics import confusion_matrix, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from matal.utils import auto_log
import numpy as np


METRICS = [('mse', mean_squared_error),
           ('mae', mean_absolute_error),
           ('mre', mean_absolute_percentage_error),
           ('r2', r2_score)]


def calc_metrics(model, all_dataset, running_loss, running_loss_i, device=None, dtype=None, ):
    train_pred = torch.round(model.predict(all_dataset['grade'].train_X.to(device=device, dtype=dtype), 'grade')).cpu()
    test_pred = torch.round(model.predict(all_dataset['grade'].test_X.to(device=device, dtype=dtype), 'grade')).cpu()

    metrics = {}
    for t in TARGET_Y_COLS.keys():
        metrics[f'loss_{t}'] = (running_loss[t] / running_loss_i[t]) if running_loss_i[t] > 0 else 0

    for gti, grade_target in enumerate(TARGET_Y_COLS['grade']):
        pred_y = train_pred[:, gti].detach().numpy().round().astype(int)
        true_y = all_dataset['grade'].train_y[:, gti].detach().numpy().round().astype(int)
        train_acc = float(np.sum(true_y == pred_y)) / len(true_y)
        metrics[f'acc_{grade_target}'] = train_acc
        try:
            metrics[f'cm_{grade_target}_tn'], \
                metrics[f'cm_{grade_target}_fp'], \
                metrics[f'cm_{grade_target}_fn'], \
                metrics[f'cm_{grade_target}_tp'] = confusion_matrix(true_y, pred_y).ravel()
        except Exception as e:
            auto_log(f'Error calculating confusion matrix for {grade_target}: {e}', level='error')
            metrics[f'cm_{grade_target}_tn'], \
                metrics[f'cm_{grade_target}_fp'], \
                metrics[f'cm_{grade_target}_fn'], \
                metrics[f'cm_{grade_target}_tp'] = np.nan, np.nan, np.nan, np.nan
        n_total_pos = metrics[f'cm_{grade_target}_tp'] + metrics[f'cm_{grade_target}_fn']
        n_total_neg = metrics[f'cm_{grade_target}_tn'] + metrics[f'cm_{grade_target}_fp']
        metrics[f'cm_{grade_target}_tnR'] = metrics[f'cm_{grade_target}_tn'] / n_total_neg if n_total_neg > 0 else np.nan
        metrics[f'cm_{grade_target}_fpR'] = metrics[f'cm_{grade_target}_fp'] / n_total_neg if n_total_neg > 0 else np.nan
        metrics[f'cm_{grade_target}_fnR'] = metrics[f'cm_{grade_target}_fn'] / n_total_pos if n_total_pos > 0 else np.nan
        metrics[f'cm_{grade_target}_tpR'] = metrics[f'cm_{grade_target}_tp'] / n_total_pos if n_total_pos > 0 else np.nan

        pred_y = test_pred[:, gti].detach().numpy().round().astype(int)
        true_y = all_dataset['grade'].test_y[:, gti].detach().numpy().round().astype(int)
        test_acc = float(np.sum(true_y == pred_y)) / len(true_y)
        metrics[f'acc_{grade_target}_test'] = test_acc
        try:
            metrics[f'cm_{grade_target}_tn_test'], \
                metrics[f'cm_{grade_target}_fp_test'], \
                metrics[f'cm_{grade_target}_fn_test'], \
                metrics[f'cm_{grade_target}_tp_test'] = confusion_matrix(true_y, pred_y).ravel()
        except Exception as e:
            auto_log(f'Error calculating confusion matrix for {grade_target}: {e}', level='error')
            metrics[f'cm_{grade_target}_tn_test'], \
                metrics[f'cm_{grade_target}_fp_test'], \
                metrics[f'cm_{grade_target}_fn_test'], \
                metrics[f'cm_{grade_target}_tp_test'] = np.nan, np.nan, np.nan, np.nan
        n_total_pos = metrics[f'cm_{grade_target}_tp_test'] + metrics[f'cm_{grade_target}_fn_test']
        n_total_neg = metrics[f'cm_{grade_target}_tn_test'] + metrics[f'cm_{grade_target}_fp_test']
        metrics[f'cm_{grade_target}_tnR_test'] = metrics[f'cm_{grade_target}_tn_test'] / n_total_neg if n_total_neg > 0 else np.nan
        metrics[f'cm_{grade_target}_fpR_test'] = metrics[f'cm_{grade_target}_fp_test'] / n_total_neg if n_total_neg > 0 else np.nan
        metrics[f'cm_{grade_target}_fnR_test'] = metrics[f'cm_{grade_target}_fn_test'] / n_total_pos if n_total_pos > 0 else np.nan
        metrics[f'cm_{grade_target}_tpR_test'] = metrics[f'cm_{grade_target}_tp_test'] / n_total_pos if n_total_pos > 0 else np.nan

    for metric, metric_func in METRICS:
        for ti, (mtarget, cols) in enumerate(TARGET_Y_COLS.items()):
            if mtarget == 'grade': continue
            train_pred = model.predict(all_dataset[mtarget].train_X.to(device=device, dtype=dtype),
                                       mtarget).cpu() if mtarget in all_dataset else None
            # test_pred = model.predict(all_dataset[mtarget].test_X, mtarget) if mtarget in all_dataset else None

            for ci, col in enumerate(cols):
                try:
                    _y_true = all_dataset[mtarget].train_y[:, ci] * TARGET_Y_SCALES[mtarget][ci]
                    _y_pred = train_pred[:, ci] * TARGET_Y_SCALES[mtarget][ci]
                    _y_true = _y_true.detach().numpy()
                    _y_pred = _y_pred.detach().numpy()
                    m_val = metric_func(_y_true, _y_pred)
                except Exception as e:
                    auto_log(f'Error evaluating {metric} on {col}: {e}', level='error')
                    m_val = np.nan
                metrics[f'{metric}_{col}'] = m_val
    return metrics
