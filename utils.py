import random
from functools import wraps
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def now():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def cartesian_product(df1, df2):
    df1["key"] = 1
    df2["key"] = 1
    return pd.merge(df1, df2, on="key").drop("key", axis=1)


def urand_given_range(start, end):
    r = np.random.random()
    return (end - start) * r + start


def rand_decorator(keyvar, start, end):
    def _rand_decorator(func):
        @wraps(func)
        def wrapper(*args, **kargs):
            r = urand_given_range(start, end)
            kargs[keyvar] = r
            res = func(*args, **kargs)
            return res
        return wrapper
    return _rand_decorator
