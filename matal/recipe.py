import copy
import pickle
import warnings
import numpy as np
import pandas as pd
import inspect
from collections import defaultdict
from matal.utils import cache_return, obj_to_hash, load_cache, save_cache, auto_log

import lz4.frame
from scipy.spatial.distance import cdist, pdist
from tqdm.auto import tqdm


def generate_random_compositions(materials, n_comps=30, n_sources_per_comp=None,
                                 guaranteed_materials=None, n_least_guaranteed_materials=0,
                                 ignored_materials=None, fixed_mask=None,
                                 random_state=None, return_df=True, min_frac=0,
                                 feas_model=None, feas_func=None, feas_cutoff=1.0, feas_mask_func=None, 
                                 batch_factor=1.0, clip_n_comps=True,
                                 filt_func=None, method='random'):
    # global comps
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    warnings.filterwarnings("ignore")

    if n_sources_per_comp is not None:
        mask_arr = [0, ] * (len(materials) - n_sources_per_comp) + [1, ] * n_sources_per_comp
    else:
        mask_arr = [1, ] * len(materials)

    if guaranteed_materials is not None and n_least_guaranteed_materials > 0:
        guaranteed_material_indexes = [materials.index(m) for m in guaranteed_materials]
    else:
        guaranteed_material_indexes = None
    
    if ignored_materials is not None:
        ignored_material_indexes = [materials.index(m) for m in ignored_materials]
    else:
        ignored_material_indexes = None
        
    if fixed_mask is None:
        def get_mask():
            while True:
                mask = np.array(sorted(mask_arr, key=lambda _: rng.random()), dtype=np.uint8)
                if (guaranteed_material_indexes is None) or (
                        sum(mask[guaranteed_material_indexes]) >= n_least_guaranteed_materials):
                    if (ignored_material_indexes is None) or (
                        sum(mask[ignored_material_indexes]) == 0):
                        break
            return mask
    else:
        get_mask = lambda *args: np.array(fixed_mask, dtype=np.uint8)
    
    if feas_mask_func is None:
        if feas_func is None:
            if feas_model is not None:
                feas_mask_func = lambda c: feas_model.predict([c]) >= feas_cutoff
        else:
            feas_mask_func = lambda c: feas_func([c]) >= feas_cutoff
    
    valid_comps = []
    batch_size = round(n_comps * batch_factor)
    round_used = 0
    while len(valid_comps) < n_comps:
        if method in ['random', 'dirichlet']:
            _method = method
        else:
            _method = 'dirichlet' if rng.integers(0, 3) else 'random'

        if _method == 'random':
            comps = (rng.random((batch_size, len(materials))) + min_frac) / (1 + min_frac)
            masks = np.array([get_mask() for _ in range(batch_size)])
            comps *= np.array(masks, dtype=np.float64)
        elif _method == 'dirichlet':
            comps = np.array([
                (rng.dirichlet(get_mask().astype(np.float64), size=None) + min_frac) / (1 + min_frac)
                for _ in range(batch_size)
            ])
        else:
            raise ValueError(_method)
            
        comps /= comps.sum(axis=1).reshape(-1, 1)
        
        if min_frac > 0:
            for _ in range(100):
                min_mask = (comps != 0) & (comps < min_frac)
                if not min_mask.any(): break
                comps[min_mask] *= (1.5 * (min_frac / np.mean(comps[min_mask])))
                comps /= comps.sum(axis=1).reshape(-1, 1)
            else:
                continue

        if feas_mask_func is not None:
            feas_mask = feas_mask_func(comps)
            comps = comps[feas_mask]
        
        if filt_func is not None:
            comps = comps[filt_func(comps)]
        round_used += 1
        
        valid_comps.extend(comps)
        
    auto_log(f'round_used = {round_used}, '
             f'len(valid_comps) = {len(valid_comps)}, '
             f'ef = {n_comps/len(valid_comps):.3%}, '
             f'fvol/tvol = {len(valid_comps)/(round_used*batch_size):.3%}', 
             level='debug')
    if clip_n_comps:
        valid_comps = valid_comps[:n_comps]

    if return_df:
        return pd.DataFrame(valid_comps, columns=materials)
    else:
        return np.array(valid_comps)

    
class UniformCompositionGenerator:
    """
    Generating spaced-out compositions within design space while maximizing
    predicted performance using Monte Carlo method.
    """

    def __init__(self, materials=None, n_comps=30, n_iters=500000,
                 random_state=None, perf_model=None, perf_func=None, existing_comps=None,
                 boltzmann_const=1e5, checkpoint_freq=10, perf_coeff=1.0, dist_coeff=1.0, 
                 intra_dist_coeff=None, inter_dist_coeff=None, 
                 cache_batch_factor=16, feas_comp_files=None, feas_comp_file_perf_col=None,
                 pbar=True, 
                 **kwargs):
        self.materials = materials
        self.n_comps = n_comps
        self.n_iters = n_iters
        self.random_state = random_state
        self.pbar = pbar
        
        if perf_func:
            self.perf_func = perf_func
        elif perf_model:
            self.perf_func = perf_model.predict
        else:
            self.perf_func = None
            
        self.existing_comps = existing_comps
        self.boltzmann_const = boltzmann_const
        self.checkpoint_freq = checkpoint_freq
        self.perf_coeff = perf_coeff
        self.dist_coeff = dist_coeff
        self.intra_dist_coeff = intra_dist_coeff if intra_dist_coeff is not None else dist_coeff
        self.inter_dist_coeff = inter_dist_coeff if inter_dist_coeff is not None else dist_coeff
        self.cache_batch_factor = cache_batch_factor
        self.feas_comp_files = list(feas_comp_files) if feas_comp_files is not None else list()
        self.feas_comp_file_perf_col = feas_comp_file_perf_col
        self.kwargs = kwargs

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            # TODO(1.4): remove specific handling of "base_estimator".
            # The "base_estimator" key is special. It was deprecated and
            # renamed to "estimator" for several estimators. This means we
            # need to translate it here and set sub-parameters on "estimator",
            # but only if the user did not explicitly set a value for
            # "base_estimator".
            if (
                    key == "base_estimator"
                    and valid_params[key] == "deprecated"
                    and self.__module__.startswith("sklearn.")
            ):
                warnings.warn(
                    f"Parameter 'base_estimator' of {self.__class__.__name__} is"
                    " deprecated in favor of 'estimator'. See"
                    f" {self.__class__.__name__}'s docstring for more details.",
                    FutureWarning,
                    stacklevel=2,
                )
                key = "estimator"
            valid_params[key].set_params(**sub_params)

        return self

    def _get_rand_comp(self, with_pred=False):
        ''' return a random composition '''

        # If the cache does not exist, or the cache is empty, rebuild cache
        comp_cache = getattr(self, '_comp_cache', [])
        if len(comp_cache) == 0:
            self._build_comp_cache()

        # pop one composition, and with its prediction if existed, from cache
        comp = self._comp_cache.pop()
        pred = self._pred_cache.pop() if self.perf_func else None
        if with_pred:
            return comp, pred
        else:
            return comp

    def _build_comp_cache(self):
        ''' build cache containing a list of random composition '''
        
        comps, preds = None, None
        if len(self.feas_comp_files) > 0:
            comp_file = self.feas_comp_files.pop(0)
            try:
                with lz4.frame.open(comp_file, 'rb') as f:
                    comp_df = pickle.load(f)
                comps = comp_df[self.materials].values
                
                auto_log(f'Load feas_comps from {comp_file}', level='debug')
                if self.perf_func and self.feas_comp_file_perf_col:
                    preds = comp_df[self.feas_comp_file_perf_col].values
                    auto_log(f'Load feas_comps preds from {self.feas_comp_file_perf_col} @ {comp_file}', level='debug')
            except Exception as e:
                auto_log(f'Error while loading feas_comps: {e}')
        
        if comps is None:
            seed = self.rng_.integers(0, np.iinfo(np.int64).max)
            comps = generate_random_compositions(self.materials, n_comps=round(self.n_comps * self.cache_batch_factor),
                                                 random_state=seed, return_df=False, clip_n_comps=False, **self.kwargs)
            preds = None
        self._comp_cache = [c for c in comps]
        
        if self.perf_func:
            if preds is None:
                auto_log(f'Calculating preds using {self.perf_func}', level='debug')
                self._pred_cache = [c for c in self.perf_func(self._comp_cache)]
            else:
                self._pred_cache = [c for c in preds]

    def _get_cache_key(self):
        params = self.get_params()
        del params['n_iters']
        del params['checkpoint_freq']

        return obj_to_hash(params)

    def load_checkpoint(self):
        cache_key = self._get_cache_key()
        try:
            checkpoints = load_cache(self.__class__.__name__, cache_key)
            assert isinstance(checkpoints, dict)

            max_iter = max([i for i in checkpoints.keys() if i <= self.n_iters])
            cp = checkpoints[max_iter]
            assert isinstance(cp, dict)

            for key, value in cp.items():
                setattr(self, key, value)

            return cp['iter_']
        except:
            return None

    def save_checkpoint(self):
        params = self.get_params()
        del params['n_iters']

        cache_key = obj_to_hash(params)

        try:
            checkpoints = load_cache(self.__class__.__name__ + '__cps', version=cache_key, log=False)
        except:
            checkpoints = dict()

        stats_vars = [v for v in vars(self) if (v[-1] == '_') and (v[-2] != '_') and (v[0] != '_')]
        cp = {v: getattr(self, v) for v in stats_vars}

        checkpoints[cp['iter_']] = cp
        save_cache(checkpoints, self.__class__.__name__ + '__cps', version=cache_key, log=True)

    def optimize(self):
        if self.pbar:
            pbar = tqdm(total=self.n_iters)

        # Initialize comps and calculate its score
        if not hasattr(self, 'iter_'):
            self.best_score_ = -np.inf
            self.opt_history_ = []
            self.iter_ = 0
            self.rng_ = np.random.default_rng(self.random_state)
            [self._get_rand_comp(with_pred=True) for _ in range(self.n_comps)]
            
            init_camps_preds = [self._get_rand_comp(with_pred=True) for _ in range(self.n_comps)]
            self.current_comps_ = [c for c, _ in init_camps_preds]
            self.current_predictions_ = [p for _, p in init_camps_preds]
            self.current_score_ = self.score_comps(self.current_comps_, 
                                                   perf_func=self.perf_func,
                                                   predictions=None if (self.perf_func is None) else self.current_predictions_)
            self.best_comps_ = self.current_comps_
        else:
            if self.pbar:
                pbar.update(self.iter_)

        if self.perf_func and (not hasattr(self, 'current_predictions_')):
            # Keep a list of prediction, so we don't need to re-predict all of them
            # each time we changed one composition
            self.current_predictions_ = self.perf_func(self.current_comps_)

        checkpoint_i = 0
        while self.iter_ < self.n_iters:
            self.iter_ += 1
            if self.pbar:
                pbar.update(1)

            if self.perf_func:
                # Randomly select a composition, compositions with lower
                # predicted performance are more likely to be selected.
                sorted_indices = np.argsort(self.current_predictions_)
                p = np.e ** (-1.0 * np.arange(len(sorted_indices)))
                pivot = self.rng_.choice(sorted_indices, p=p / p.sum())
            else:
                # Randomly select a composition
                pivot = self.rng_.integers(0, len(self.current_comps_))

            # Replace the composition with a new random one
            comps = copy.deepcopy(self.current_comps_)
            new_comp, new_pred = self._get_rand_comp(with_pred=True)
            comps[pivot] = new_comp

            # Update predictions
            if self.perf_func:
                predictions = copy.deepcopy(self.current_predictions_)
                predictions[pivot] = new_pred
            else:
                predictions = None

            # Update score
            score = self.score_comps(comps, predictions=predictions)

            # if self._rng.random() > np.log(score / best_score):

            # if np.exp(np.clip(self.boltzmann_const * (score - self.current_score_), -np.inf, 100)) > self.rng_.random():
            if (self.boltzmann_const * (score - self.current_score_)) > np.log(self.rng_.random()):
            # if score > self.best_score_:
                self.current_score_ = score
                self.current_comps_ = comps
                if self.perf_func:
                    self.current_predictions_ = predictions
                self.opt_history_.append([self.iter_, pivot, score])

                if score > self.best_score_:
                    self.best_comps_ = comps
                    self.best_score_ = score
                    if self.perf_func:
                        self.best_predictions_ = predictions
                    checkpoint_i += 1
                    # if checkpoint_i % self.checkpoint_freq == 0:
                    #     self.save_checkpoint()

        # self.save_checkpoint()
        if self.pbar:
            pbar.close()
        return pd.DataFrame(self.best_comps_, columns=self.materials)

    def score_comps(self, comps, perf_func=None, predictions=None, ):
        if predictions is None:
            if perf_func is None:
                predictions = [1.0]
            else:
                predictions = perf_func(comps)
        perf_score = np.mean(predictions)
        
        if (self.intra_dist_coeff > 0.0):
            min_intra_distance = pdist(comps).min()
        else:
            min_intra_distance = 1.0
            
        if (self.inter_dist_coeff > 0.0) and (self.existing_comps is not None):
            min_inter_distance = cdist(self.existing_comps, comps).min()
        else:
            min_inter_distance = 1.0

        # score of a list of comps = (min paired-distance) * mean(predicted performance)
        score = (min_intra_distance ** self.intra_dist_coeff) * \
                (min_inter_distance ** self.inter_dist_coeff) * \
                (perf_score ** self.perf_coeff)
        
        return score
        