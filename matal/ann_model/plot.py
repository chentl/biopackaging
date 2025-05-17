import matplotlib.pyplot as plt

from .metrics import METRICS
from .settings import TARGET_Y_COLS


def plot_hist_perf(hist, save_name='model', save_dir=None, close=True):
    plot_targets = [f'{target}_{y_col}' for target, y_cols in TARGET_Y_COLS.items() if target != 'grade' for y_col in
                    y_cols]
    targets = list(TARGET_Y_COLS.keys())

    fig, axs = plt.subplots(5, len(plot_targets),
                            figsize=(3 * len(plot_targets), 16),
                            dpi=300, layout='constrained',
                            gridspec_kw=dict(wspace=0.1, hspace=0.1))

    for row_i, ax_row in enumerate(axs):
        for col_i, ax in enumerate(ax_row):
            if row_i == 0:
                m = 'Loss'
                if col_i >= len(targets):
                    continue
                ax.plot(hist['step'], hist[f'{m.lower()}_{targets[col_i]}'])
                ax.set_ylabel(f'Loss - {targets[col_i]}')
            else:
                m = METRICS[row_i - 1][0].upper()
                plot_target = plot_targets[col_i]
                if '_' not in plot_target:
                    for y_col in TARGET_Y_COLS[plot_target]:
                        ax.plot(hist['step'], hist[f'{m.lower()}_{y_col}'], label=f'{m} - {y_col}')
                    ax.set_ylabel(f'{m} - {plot_target}')
                    ax.legend()
                else:
                    target, y_col = plot_target.split('_')
                    ax.plot(hist['step'], hist[f'{m.lower()}_{y_col}'], label=f'{m} - {y_col}')
                    ax.set_ylabel(f'{m} - {y_col}')
                if m == 'R2':
                    ax.set_yscale('symlog')
                    ax.set_ylim(None, 1)
            ax.set_xlabel('Step')

    if save_dir:
        fig.savefig(save_dir / f'{save_name}.hist_perf.pdf')
    if close:
        plt.close(fig)
    else:
        return fig


def plot_hist_grade(hist, save_name='model', save_dir=None, close=True):
    fig, axs = plt.subplots(3, 3, figsize=(12, 12), dpi=300, layout='constrained',
                            gridspec_kw=dict(wspace=0.1, hspace=0.1))

    for row_i, ax_row in enumerate(axs):
        for col_i, ax in enumerate(ax_row):
            if (row_i == 0) and (col_i == 0):
                ax.plot(hist['step'], hist['loss_grade'])
                ax.set_ylabel('Loss')
                ax.set_title('Train Loss')
            elif (row_i == 0) and (col_i == 1):
                for y_col in TARGET_Y_COLS['grade']:
                    ax.plot(hist['step'], hist[f'acc_{y_col}'], label=y_col)
                ax.set_ylim(0, 1)
                ax.set_ylabel('Accuracy')
                ax.set_title('Train Accuracy')
            elif (row_i == 0) and (col_i == 2):
                for y_col in TARGET_Y_COLS['grade']:
                    ax.plot(hist['step'], hist[f'acc_{y_col}_test'], label=y_col)
                ax.set_ylim(0, 1)
                ax.set_ylabel('Accuracy')
                ax.set_title('Test Accuracy')
            else:
                y_col = TARGET_Y_COLS['grade'][col_i]
                suffix, data_label = [('', 'Train'), ('_test', 'Test'), ][row_i - 1]
                for cm, c in zip(['tn', 'fp', 'fn', 'tp'], ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']):
                    ax.plot(hist['step'], hist[f'cm_{y_col}_{cm}R{suffix}'],
                            label=f'{cm.upper()}', color=c,
                            linestyle='-' if cm[0] == 't' else ':')
                ax.set_ylabel('Sample Count')
                ax.set_title(f'{data_label} Confusion Matrix @ {y_col}')
                ax.set_ylim(0, 1)

            ax.set_xlabel('Step')
            ax.legend()

    if save_dir:
        fig.savefig(save_dir / f'{save_name}.hist_grade.pdf')

    if close:
        plt.close(fig)
    else:
        return fig
