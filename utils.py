import random
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
