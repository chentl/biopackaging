def str_to_list(s: str, func: callable = str, sep: str = ','):
    return [func(i.strip()) for i in s.split(sep) if i.strip()]

