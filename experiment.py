from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import config
import generator
import learner
import model
import utils
import submit

"""
Experiment and Cross Validation Script for local data.

Todo:
1. Validation silence data (completed)
2. Cross Vaidation script with respect of uid split
"""


def extract_fname(path):
    return Path(path).parts[-1]


def augment_data_load(paths, augs, version):
    flist = paths.path.apply(extract_fname)
    for aug in augs:
        print("load file info of {}".format(aug))
        aug_path = "{}_file_info_version_{}.csv".format(aug,
                                                        version)
        augment_file_info = pd.read_csv(Path("data/")/aug_path)
        augment_file_info = augment_file_info[["path",
                                               "uid",
                                               "possible_label",
                                               "plnum",
                                               "is_valid"]]
        augment_file_info["fn"] = augment_file_info.path.apply(extract_fname)
        augment_file_info = augment_file_info[augment_file_info.fn.isin(flist)]
        augment_file_info = augment_file_info.drop("fn", axis=1)
        paths = pd.concat([paths, augment_file_info])
        print("done")

    print(paths.shape)
    return paths


def data_load(silence_data_version):
    print("base data load")
    file_df = pd.read_csv(config.TRAIN_FILE_META_INFO)
    file_df = file_df[["path", "uid", "possible_label", "plnum", "is_valid"]]
    bg_paths = file_df[file_df["possible_label"] == "_background_noise_"]
    file_df = file_df[file_df["possible_label"] != "_background_noise_"]
    file_df['plnum'] = file_df.plnum.astype('int64')

    silence_data_path = Path(config.SILECE_DATA_PATH)/silence_data_version
    silence_df = pd.read_csv(silence_data_path/"file_info.csv")
    print("done")

    return file_df, bg_paths, silence_df


def experiment(estimator,
               train_df,
               valid_df,
               bg_paths,
               batch_size,
               sample_size,
               augment_list,
               version_path=None,
               csv_log_path=None):
    
    label_num = len(config.POSSIBLE_LABELS)
    train_generator = generator.batch_generator(train_df,
                                                batch_size,
                                                label_num,
                                                bg_paths,
                                                aug_processes=augment_list,
                                                sampling_size=sample_size)
    valid_generator = generator.batch_generator(valid_df,
                                                batch_size,
                                                label_num,
                                                bg_paths,
                                                aug_processes=augment_list,
                                                mode='valid',
                                                sampling_size=sample_size)
    augmented_data_size = sample_size*label_num*(len(augment_list)+1)
    valid_steps = int(np.ceil(valid_df.shape[0]/batch_size))
    steps_per_epoch = int(np.ceil(augmented_data_size/batch_size))

    learn = learner.Learner(estimator, version_path, csv_log_path)
    result = learn.learn(train_generator,
                         valid_generator,
                         valid_steps,
                         steps_per_epoch=steps_per_epoch)
    return result


def validation(silence_data_version,
               estimator,
               augment_list,
               aug_version,
               sample_size=2000,
               batch_size=config.BATCH_SIZE,
               silence_train_size=2000):
    print("experiment(validation type) start")
    file_df, bg_paths, silence_df = data_load(silence_data_version)

    train_df = file_df[~file_df.is_valid]
    valid_df = file_df[file_df.is_valid]

    silence_train = silence_df.iloc[:silence_train_size]
    silence_valid = silence_df.iloc[silence_train_size:]

    print("load augmentation")
    print("train")
    train_df = augment_data_load(train_df, augment_list, aug_version)
    print("valid")
    # valid_df = augment_data_load(valid_df, augment_list, aug_version)
    print("silence_data")
    silence_train = augment_data_load(silence_train, augment_list, aug_version)
    # silence_valid = augment_data_load(silence_valid, augment_list, aug_version)
    print("done")

    train_df = pd.concat([train_df, silence_train])
    valid_df = pd.concat([valid_df, silence_valid])

    assert(len(train_df.plnum.unique()) == len(config.POSSIBLE_LABELS))

    print("data load done")
    estimator.model_init()
    result = experiment(estimator, train_df, valid_df, bg_paths,
                        batch_size, sample_size, augment_list)
    return result


def cross_validation(estimator,
                     silence_data_version,
                     cv_version,
                     aug_version,
                     n_splits=5,
                     sample_size=1800,
                     batch_size=64,
                     silence_train_size=1800):
    
    """cross_validation func with silence_data
    
    Todo: uid label list encoding -> stratified kfold
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

    for i, ((train_id, other_id), (train_sid, other_sid)) in enumerate(kfold):
        train_uid = uid_list[train_id]
        valid_uid = uid_list[train_id[:len(other_id)/2]]
        test_uid = uid_list[other_id[len(other_id)/2:]]

        train = file_df[file_df.uid.isin(train_uid)]
        valid = file_df[file_df.uid.isin(valid_uid)]
        test = file_df[file_df.uid.isin(test_uid)]
        assert(set(train.uid) & set(test.uid) & set(valid.uid) == set())

        train = augment_data_load(train, config.AUG_LIST, aug_version)
        silence_train = silence_data.iloc[train_sid]
        silence_train = augment_data_load(silence_train,
                                          config.AUG_LIST,
                                          aug_version)
        train = pd.concat([train, silence_train])

        valid = augment_data_load(valid, config.AUG_LIST, aug_version)
        silence_valid_id = other_sid[:len(other_sid)/2]
        silence_valid = silence_data.iloc[silence_valid_id]
        silence_valid = augment_data_load(silence_valid,
                                          config.AUG_LIST,
                                          aug_version)
        valid = pd.concat([valid, silence_valid])

        test_silence = other_sid[len(other_sid)/2:]
        test = pd.concat([test, silence_data.iloc[test_silence]])
        print(i, len(train), len(valid), len(test))

        label_dist = train.possible_label.value_counts()
        label_dist.to_csv(version_path/"fold_{}_train_ldist".format(i))

        label_dist = test.possible_label.value_counts()
        label_dist.to_csv(version_path/"fold_{}_test_ldist".format(i))
        
        fold_dump_path = str(version_path / "fold_{}.hdf5".format(i))
        csv_log_path = str(version_path / "fold_{}_log.csv".format(i))

        estimator = model.VGG1D()
        estimator.model_init()  # initialize model

        res_fold = experiment(estimator, train, test, bg_paths,
                              batch_size, sample_size,
                              version_path=fold_dump_path,
                              csv_log_path=csv_log_path)

        predict_test_probs = submit.predict(test,
                                            bg_paths,
                                            estimator)
        fold_plobs = pd.DataFrame(predict_test_probs,
                                  columns=config.POSSIBLE_LABELS)
        fold_plobs_df = pd.concat([test, fold_plobs], axis=1)
        fold_plobs_df.to_csv(version_path/"fold_{}_test.csv".format(i),
                             index=False)
        predict_id = np.argmax(predict_test_probs, axis=1)
        print("test accuracy:", accuracy_score(test.plnum, predict_id))
              
        result.append(res_fold)

    return result


if __name__ == "__main__":
    utils.set_seed(2017)

    cnn = model.VGG1D()
    silence_data_version = "2017_12_08_15_41_26"
    cv_version = utils.now()
    validation(silence_data_version,
               cnn,
               config.AUG_LIST,
               config.AUG_VERSION,
               sample_size=2000)
    # res = cross_validation(cnn, silence_data_version, cv_version)
