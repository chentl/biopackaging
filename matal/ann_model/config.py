from matal.utils import auto_log, obj_to_hash, Bunch, RobustJSONEncoder
from .settings import TARGET_Y_COLS


def sort_dict(d, **kwargs):
    return {k: d[k] for k in sorted(d.keys(), **kwargs)}


class BunchWithDefaults(Bunch):
    _DEFAULTS = {}
    _EXCLUDE_FROM_HASH = []
    _REJECT_NOT_IN_DEFAULT = False

    def __init__(self, *args,
                 defaults=None,
                 exclude_from_hash=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if defaults:
            self._DEFAULTS.update(defaults)
        if exclude_from_hash:
            self._EXCLUDE_FROM_HASH.extend(exclude_from_hash)

    def get_content(self, reduce_default=True, sort_keys=True, for_hash=False):
        content = Bunch()
        for k, v in self.items():
            if k.startswith('_'): continue
            if reduce_default and k in self._DEFAULTS and self._DEFAULTS.get(k) == v:
                continue
            if for_hash and k in self._EXCLUDE_FROM_HASH:
                continue
            content[k] = v

        if sort_keys:
            content = Bunch(**sort_dict(content))

        return content

    def get_all(self):
        content = Bunch(**self._DEFAULTS)
        content.update(self.get_content())
        return content

    def get_hash(self, reduce_default=True, sort_keys=True, **kwargs):
        content = self.get_content(reduce_default=reduce_default, sort_keys=sort_keys, for_hash=True)
        return obj_to_hash(content, **kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        elif key in self._DEFAULTS:
            return self._DEFAULTS[key]
        else:
            raise AttributeError(key)

    def __setitem__(self, key, value):
        if self._REJECT_NOT_IN_DEFAULT and key not in self._DEFAULTS:
            return
        dict.__setitem__(self, key, value)

    def update(self, *args, **kwargs) -> None:
        for k, v in dict(*args, **kwargs).items():
            if self._REJECT_NOT_IN_DEFAULT and k not in self._DEFAULTS:
                continue
            self[k] = v


class ModelConfig(BunchWithDefaults):
    _DEFAULTS = dict(
        encoder__hidden_layers=12,
        encoder__hidden_base=2.0,
        encoder__hidden_scale='linear',
        encoder__n_output=8,
        encoder__act_hidden='elu',
        encoder__act_output='identity',
        encoder__weight_init='kaiming_uniform_',
        encoder__input_dropout=0.0,
        encoder__output_dropout=0.0,
        encoder__hidden_dropout=0.0,

        grade__hidden_layers=4,
        grade__hidden_base=2.0,
        grade__hidden_scale='linear',
        grade__n_output=len(TARGET_Y_COLS['grade']),
        grade__act_hidden='elu',
        grade__act_output='sigmoid',
        grade__weight_init='kaiming_uniform_',
        grade__input_dropout=0.0,
        grade__output_dropout=0.0,
        grade__hidden_dropout=0.0,

        tensile__hidden_layers=4,
        tensile__hidden_base=2.0,
        tensile__hidden_scale='linear',
        tensile__n_output=len(TARGET_Y_COLS['tensile']),
        tensile__act_hidden='elu',
        tensile__act_output='relu',
        tensile__weight_init='kaiming_uniform_',
        tensile__input_dropout=0.0,
        tensile__output_dropout=0.0,
        tensile__hidden_dropout=0.0,

        optical__hidden_layers=4,
        optical__hidden_base=2.0,
        optical__hidden_scale='linear',
        optical__n_output=len(TARGET_Y_COLS['optical']),
        optical__act_hidden='elu',
        optical__act_output='sigmoid',
        optical__weight_init='kaiming_uniform_',
        optical__input_dropout=0.0,
        optical__output_dropout=0.0,
        optical__hidden_dropout=0.0,

        fire__hidden_layers=4,
        fire__hidden_base=2.0,
        fire__hidden_scale='linear',
        fire__n_output=len(TARGET_Y_COLS['fire']),
        fire__act_hidden='elu',
        fire__act_output='sigmoid',
        fire__weight_init='kaiming_uniform_',
        fire__input_dropout=0.0,
        fire__output_dropout=0.0,
        fire__hidden_dropout=0.0,
        random_seed=0,
        dtype_str='float64',
    )
    _REJECT_NOT_IN_DEFAULT = True


class FitConfig(BunchWithDefaults):
    _DEFAULTS = dict(
        epochs=10,
        lr=1e-5,
        l2=1e-8,
        noise_std=1 / 200,
        ckpt_freq=100000,
        hist_freq=5000,
        split_opt=True,
        split_enc_opt=False,
        restart=True,
        load_weights_from='',
        load_opt_from='',
    )
    _EXCLUDE_FROM_HASH = ['epochs', 'ckpt_freq', 'hist_freq', 'restart']
    _REJECT_NOT_IN_DEFAULT = True
