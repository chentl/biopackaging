import collections.abc

# from https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
def merge_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = merge_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
