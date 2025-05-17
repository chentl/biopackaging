import os
import pickle
import dill
import shutil
import hashlib
import base64
import json

import numpy as np
import pandas as pd

from .log import auto_log
from ..settings import CACHE_DIR

import lz4.frame


class CacheLoadError(Exception):
    pass


class RobustJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, np.generic):
            return obj.tolist() 
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif hasattr(obj, 'serialize'):
            return f'*** unserializable object {repr(obj)} ***'
            # TODO: serialization for tensorflow.python.keras.engine.node.Node object
            # return obj.serialize()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return f'*** unserializable object {repr(obj)} ***'


def save_cache(obj: object, tag: str, version: str = None, meta: object = None, log=True) -> None:
    if version:
        cache_dir = os.path.join(CACHE_DIR, tag, version[:2])
    else:
        cache_dir = os.path.join(CACHE_DIR, tag)
    os.makedirs(cache_dir, exist_ok=True)
    
    if meta:
        json_name = f'cache_{version}.meta.json' if version else 'cache.meta.json'
        with open(os.path.join(cache_dir, json_name), 'w') as f:
            json.dump(meta, f, cls=RobustJSONEncoder, indent=2)


    cache_name = f'cache_{version}.pkl' if version else 'cache.pkl'
    with open(os.path.join(cache_dir, cache_name), 'wb') as f:
        dill.dump(obj, f)
    
    if log: auto_log(f'Cache saved to {cache_name} @ {tag}', level='debug')

        
def load_cache(tag: str, version: str = None, log=True) -> object:
    if version:
        cache_dir = os.path.join(CACHE_DIR, tag, version[:2])
    else:
        cache_dir = os.path.join(CACHE_DIR, tag)
    cache_name = f'cache_{version}.pkl' if version else 'cache.pkl'
    
    try:
        with open(os.path.join(cache_dir, cache_name), 'rb') as f:
            obj = dill.load(f)
            if log: auto_log(f'Cache loaded from {cache_name} @ {tag}', level='debug')
    except Exception:
        raise CacheLoadError()
    
    return obj


def load_cache_lz4(tag: str, version: str = None, log=True, with_meta=False) -> object:
    if version:
        cache_dir = os.path.join(CACHE_DIR, tag, version[:2])
    else:
        cache_dir = os.path.join(CACHE_DIR, tag)
    cache_name = f'cache_{version}.pkl.lz4' if version else 'cache.pkl.lz4'
    json_name = f'cache_{version}.meta.json.lz4' if version else 'cache.meta.json.lz4'

    try:
        with lz4.frame.open(os.path.join(cache_dir, cache_name), 'rb') as f:
            obj = dill.load(f)
            if log: auto_log(f'Cache loaded from {cache_name} @ {tag}', level='debug')
    except (OSError, pickle.UnpicklingError) as e:
        raise CacheLoadError(e)

    if with_meta:
        try:
            with lz4.frame.open(os.path.join(cache_dir, json_name), 'rb') as f:
                meta = json.loads(f.read().decode())
        except (OSError, ValueError) as e:
            raise CacheLoadError(e)
        return obj, meta
    else:
        return obj



def obj_to_hash(obj: object, size:int = 5) -> str:
    obj_pkl = dill.dumps(obj)
    h = hashlib.blake2b(digest_size=size)
    h.update(obj_pkl)
    version = base64.b32encode(h.digest()).decode('utf-8').replace('=', '0')
    return version


def cache_return(func, save_meta=True, log=False):
    def wrapper(*args, **kwargs):
        tag = f'func__{func.__name__}'
        
        params = {'arg': args, 'kwargs': kwargs}
        version = obj_to_hash(params, size=40)
        try:
            if log: auto_log(f'Try to load function cache for {tag} @ {version}', level='debug')
            obj = load_cache(tag, version=version)
            if log: auto_log(f'Loaded function cache from {tag} @ {version}')
        except CacheLoadError:
            obj = func(*args, **kwargs)
            save_cache(obj, tag, version=version, meta=params if save_meta else None)
            if log: auto_log(f'Saved function cache to {tag} @ {version}')
            
        return obj
    
    return wrapper
    
