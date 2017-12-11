import random
from datetime import datetime
import numpy as np
import tensorflow as tf


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def now():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
