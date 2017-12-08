from pathlib import Path
import pandas as pd
import numpy as np
import config
import generator
import learner
import model
import utils

"""
Experiment and Cross Validation Script for local data.

Todo:
1. Validation silence data (completed)
2. Cross Vaidation script with respect of uid split
"""


def data_load(silence_data_version, silence_train_size=2000):
    file_df = pd.read_csv(config.TRAIN_FILE_META_INFO)
    file_df = file_df[["path", "uid", "possible_label", "plnum", "is_valid"]]
    bg_paths = file_df[file_df["possible_label"] == "_background_noise_"]
    file_df = file_df[file_df["possible_label"] != "_background_noise_"]

    train_df = file_df[file_df.is_valid != True]
    valid_df = file_df[file_df.is_valid]

    silence_data_path = Path(config.SILECE_DATA_PATH)/silence_data_version
    silence_df = pd.read_csv(silence_data_path/"file_info.csv")
    silence_train = silence_df.iloc[:silence_train_size]
    silence_valid = silence_df.iloc[silence_train_size:]
    print(train_df.shape, valid_df.shape)
    train_df = pd.concat([train_df, silence_train])
    valid_df = pd.concat([valid_df, silence_valid])

    return train_df, valid_df, bg_paths


if __name__ == "__main__":
    utils.set_seed(2017)
    
    train, valid, bg_paths = data_load("2017_12_08_15_41_26")
    print(train.shape, valid.shape)
    sample_size = 2000
    
    batch_size = 64
    label_num = len(config.POSSIBLE_LABELS)
    train_generator = generator.batch_generator(train,
                                                batch_size,
                                                label_num,
                                                bg_paths)
    valid_generator = generator.batch_generator(valid,
                                                batch_size,
                                                label_num,
                                                bg_paths,
                                                mode='valid')
    valid_steps = int(np.ceil(valid.shape[0]/batch_size))
    steps_per_epoch = int(np.ceil(sample_size*label_num/batch_size))

    cnn = model.STFTCNN()
    learner = learner.Learner(cnn)
    result = learner.learn(train_generator,
                           valid_generator,
                           valid_steps,
                           steps_per_epoch=steps_per_epoch)
