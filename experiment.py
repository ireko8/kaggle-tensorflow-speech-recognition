from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
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


def data_load(silence_data_version):
    file_df = pd.read_csv(config.TRAIN_FILE_META_INFO)
    file_df = file_df[["path", "uid", "possible_label", "plnum", "is_valid"]]
    bg_paths = file_df[file_df["possible_label"] == "_background_noise_"]
    file_df = file_df[file_df["possible_label"] != "_background_noise_"]

    silence_data_path = Path(config.SILECE_DATA_PATH)/silence_data_version
    silence_df = pd.read_csv(silence_data_path/"file_info.csv")

    return file_df, bg_paths, silence_df


def experiment(estimator,
               train_df,
               valid_df,
               bg_paths,
               batch_size,
               sample_size,
               version_path=None,
               csv_log_path=None):
    
    label_num = len(config.POSSIBLE_LABELS)
    train_generator = generator.batch_generator(train_df,
                                                batch_size,
                                                label_num,
                                                bg_paths,
                                                sampling_size=sample_size)
    valid_generator = generator.batch_generator(valid_df,
                                                batch_size,
                                                label_num,
                                                bg_paths,
                                                mode='valid',
                                                sampling_size=sample_size)
    valid_steps = int(np.ceil(valid_df.shape[0]/batch_size))
    steps_per_epoch = int(np.ceil(sample_size*label_num/batch_size))

    learn = learner.Learner(estimator, version_path, csv_log_path)
    result = learn.learn(train_generator,
                         valid_generator,
                         valid_steps,
                         steps_per_epoch=steps_per_epoch)
    return result


def validation(silence_data_version,
               estimator,
               sample_size=2000,
               batch_size=64,
               silence_train_size=2000):
    file_df, bg_paths, silence_df = data_load(silence_data_version)
    train_df = file_df[~file_df.is_valid]
    valid_df = file_df[file_df.is_valid]

    silence_train = silence_df.iloc[:silence_train_size]
    silence_valid = silence_df.iloc[silence_train_size:]
    train_df = pd.concat([train_df, silence_train])
    valid_df = pd.concat([valid_df, silence_valid])

    estimator.model_init()
    result = experiment(estimator, train_df, valid_df, bg_paths,
                        batch_size, sample_size)
    return result


def cross_validation(estimator,
                     silence_data_version,
                     cv_version,
                     n_splits=5,
                     sample_size=1800,
                     batch_size=64,
                     silence_train_size=1800):
    
    """cross_validation func with silence_data
    """

    version_path = Path("cv/")/estimator.name/cv_version
    version_path.mkdir(parents=True, exist_ok=True)
    file_df, bg_paths, silence_data = data_load(silence_data_version)
    file_df = file_df.drop(["is_valid"], axis=1)

    uid_list = file_df.uid.unique()
    kfold_data = KFold(n_splits=n_splits, shuffle=True).split(uid_list)
    kfold_silence = KFold(n_splits=n_splits, shuffle=True).split(silence_data)
    kfold = zip(kfold_data, kfold_silence)
    result = list()

    for i, ((train, test), (train_silence, test_silence)) in enumerate(kfold):
        train_uid = uid_list[train]
        test_uid = uid_list[test]

        train = file_df[file_df.uid.isin(train_uid)]
        test = file_df[file_df.uid.isin(test_uid)]
        train = pd.concat([train, silence_data.iloc[train_silence]])
        test = pd.concat([test, silence_data.iloc[test_silence]])
        print(i, len(train), len(test))

        fold_dump_path = str(version_path / "fold_{}.hdf5".format(i))
        csv_log_path = str(version_path / "fold_{}_log.csv".format(i))

        estimator.model_init()  # initialize model

        res_fold = experiment(estimator, train, test, bg_paths,
                              batch_size, sample_size,
                              version_path=fold_dump_path,
                              csv_log_path=csv_log_path)
        result.append(res_fold)

    return result


if __name__ == "__main__":
    utils.set_seed(2017)

    cnn = model.STFTCNN()
    silence_data_version = "2017_12_08_15_41_26"
    cv_version = utils.now()
    res = cross_validation(cnn, silence_data_version, cv_version)
