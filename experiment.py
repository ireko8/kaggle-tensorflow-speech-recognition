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

    silence_data_path = Path(config.SILENCE_DATA_PATH)/silence_data_version
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
                                                sampling_size=sample_size)
    valid_generator = generator.batch_generator(valid_df,
                                                batch_size,
                                                label_num,
                                                bg_paths,
                                                mode='valid',
                                                sampling_size=sample_size)
    data_size = sample_size*label_num
    valid_steps = int(np.ceil(valid_df.shape[0]/batch_size))
    steps_per_epoch = int(np.ceil(data_size/batch_size))

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
    sample_size = sample_size*(len(augment_list) + 1)

    assert(len(train_df.plnum.unique()) == len(config.POSSIBLE_LABELS))
    assert(set(train_df.uid) & set(valid_df.uid) == set())

    print("data load done")
    estimator.model_init()
    result = experiment(estimator, train_df, valid_df, bg_paths,
                        batch_size, sample_size, augment_list)
    return result


def cross_validation(estimator_name,
                     silence_data_version,
                     cv_version,
                     aug_version,
                     aug_list,
                     n_splits=5,
                     base_sample_size=1800,
                     batch_size=64,
                     silence_train_size=1800):
    
    """cross_validation func with silence_data
    
    Todo: uid label list encoding -> stratified kfold
    """

    version_path = Path("cv/")/estimator_name/cv_version
    version_path.mkdir(parents=True, exist_ok=True)
    file_df, bg_paths, silence_data = data_load(silence_data_version)
    file_df = file_df.drop(["is_valid"], axis=1)

    uid_list = file_df.uid.unique()
    kfold_data = KFold(n_splits=n_splits, shuffle=True).split(uid_list)
    kfold_silence = KFold(n_splits=n_splits, shuffle=True).split(silence_data)
    kfold = zip(kfold_data, kfold_silence)
    result = list()
    cv_acc = list()

    for i, ((train_id, other_id), (train_sid, other_sid)) in enumerate(kfold):
        print("fold {} start".format(i))
        print("-"*80)
        train_uid = uid_list[train_id]
        id_valid_len = int(len(other_id)/2)
        valid_uid = uid_list[other_id[:id_valid_len]]
        test_uid = uid_list[other_id[id_valid_len:]]

        train = file_df[file_df.uid.isin(train_uid)]
        valid = file_df[file_df.uid.isin(valid_uid)]
        test = file_df[file_df.uid.isin(test_uid)]

        # quick check for proper validation
        assert(set(train.uid) & set(valid.uid) == set())
        assert(set(train.uid) & set(test.uid) == set())
        assert(set(valid.uid) & set(test.uid) == set())

        train = augment_data_load(train, config.AUG_LIST, aug_version)
        silence_train = silence_data.iloc[train_sid]
        silence_train = augment_data_load(silence_train,
                                          config.AUG_LIST,
                                          aug_version)
        train = pd.concat([train, silence_train])

        valid = augment_data_load(valid, config.AUG_LIST, aug_version)
        sid_valid_len = int(len(other_sid)/2)
        silence_valid_id = other_sid[:sid_valid_len]
        silence_valid = silence_data.iloc[silence_valid_id]
        silence_valid = augment_data_load(silence_valid,
                                          config.AUG_LIST,
                                          aug_version)
        valid = pd.concat([valid, silence_valid])

        test = augment_data_load(test, config.AUG_LIST, aug_version)
        test_silence_id = other_sid[sid_valid_len:]
        silence_test = silence_data.iloc[test_silence_id]
        silence_test = augment_data_load(silence_test,
                                         config.AUG_LIST,
                                         aug_version)
        test = pd.concat([test, silence_test])

        # info of dataset
        print('{:>10},{:>10},{:>10},{:>10}'.format("type",
                                                   "train_size",
                                                   "valid_size",
                                                   "test_size"))
        print('train', len(train), len(valid), len(test))
        print('silence',
              len(silence_train),
              len(silence_valid),
              len(silence_test))
        print("train label dist")
        print(train.possible_label.value_counts())
        print("valid label dist")
        print(valid.possible_label.value_counts())
        print("test label dist")
        print(test.possible_label.value_counts())

        sample_size = base_sample_size*(len(aug_list) + 1)
        print('augmentation types', len(aug_list), sample_size)

        label_dist = train.possible_label.value_counts()
        label_dist.to_csv(version_path/"fold_{}_train_ldist".format(i))

        label_dist = test.possible_label.value_counts()
        label_dist.to_csv(version_path/"fold_{}_test_ldist".format(i))
        
        fold_dump_path = str(version_path / "fold_{}.hdf5".format(i))
        csv_log_path = str(version_path / "fold_{}_log.csv".format(i))

        # TODO: refactor architecture of model module (VGG1D, STFTCNN)
        if estimator_name == "VGG1D":
            estimator = model.VGG1D()
            estimator.model_init()

        print("learning start")
        print("-"*40)
        res_fold = experiment(estimator, train, test, bg_paths,
                              batch_size, sample_size, aug_list,
                              version_path=fold_dump_path,
                              csv_log_path=csv_log_path)
        print("-"*40)
        print("done")

        predict_test_probs = submit.predict(test,
                                            bg_paths,
                                            estimator)
        fold_plobs = pd.DataFrame(predict_test_probs,
                                  columns=config.POSSIBLE_LABELS)
        test.index = range(len(test))
        fold_plobs_df = pd.concat([test, fold_plobs], axis=1)
        fold_plobs_df.to_csv(version_path/"fold_{}_test.csv".format(i),
                             index=False)
        predict_id = np.argmax(predict_test_probs, axis=1)
        acc = accuracy_score(test.plnum, predict_id)
        print("test accuracy:", acc)

        result.append(res_fold)
        cv_acc.append(acc)

    cv_acc = pd.Series(cv_acc)
    print("cv accuracy mean: {}, std:{}".format(cv_acc.mean(),
                                                cv_acc.std()))
    cv_acc.to_csv(version_path/"cv_test.csv")
    return result


if __name__ == "__main__":
    utils.set_seed(2017)

    cv_version = "{time}_{model}_augmented".format(**{'time': utils.now(),
                                                      'model': "VGG1D"})
    # validation(config.SILECE_DATA_VERSION,
    #            cnn,
    #            config.AUG_LIST,
    #            config.AUG_VERSION,
    #            sample_size=2000)
    res = cross_validation("VGG1D",
                           config.SILENCE_DATA_VERSION,
                           cv_version,
                           config.AUG_VERSION,
                           config.AUG_LIST)
