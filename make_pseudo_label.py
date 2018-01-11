from pathlib import Path
import pandas as pd
import numpy as np
import utils
import config


def convert_aug_path(row, aug, test_aug_version):
    fname = Path(row.path).parts[-1]
    aug_path = Path(config.AUG_PATH)
    aug_path = aug_path/test_aug_version/aug/"audio"/fname
    return [aug_path, row.plnum, row.possible_label]


def make_pseudo_labeling(cv_version, fold,
                         threshold=None,
                         max_probs_drop=True,
                         sampling=None,
                         sampling_threshold=False,
                         num_fold=5):
    pseudo_dir = Path('data/pseudo_label/{}'.format(cv_version))
    pseudo_dir.mkdir(exist_ok=True, parents=True)
    
    test_paths = sorted(Path(config.TEST_AUDIO_PATH).glob("*wav"))
    test_paths = pd.DataFrame(test_paths, columns=["path"])
    test_flist = test_paths.path.apply(lambda x: x.parts[-1])

    cv_res = []
    
    for i in range(num_fold):
        print("fold {}".format(i))
        if i != fold:
            res = pd.read_csv("sub/{}/{}_probs.csv".format(cv_version,
                                                           i))
            res = res.sort_values(by='path')
            assert(all(test_flist.values == res.path.values))
            cv_res.append(res.drop("path", axis=1).values)

    cv_probs = np.array(cv_res)
    print(cv_probs.shape)
    cv_mean = np.mean(cv_probs, axis=0)
    pseudo_plnum = pd.Series(np.argmax(cv_mean, axis=1), name="plnum")
    max_probs = pd.Series(np.max(cv_mean, axis=1), name="max_probs")
    pseudo_label = pd.concat([test_paths, pseudo_plnum, max_probs], axis=1)
    pseudo_label["possible_label"] = utils.id_to_label(pseudo_label.plnum)
    if threshold:
        pseudo_label = pseudo_label[pseudo_label.max_probs >= threshold]
    if sampling:
        pseudo_label = pseudo_label.groupby("possible_label")

        def pseudo_sampling(label):
            sl = label.sort_values(by="max_probs",
                                   ascending=False)
            return sl[:sampling]
        
        pseudo_label = pseudo_label.apply(pseudo_sampling)
    if max_probs_drop:
        pseudo_label = pseudo_label.drop("max_probs", axis=1)
    print(pseudo_label.possible_label.value_counts())
    return pseudo_label


def make_pseudo_augment(pseudo_label, aug_list, test_aug_version):

    augmented_df = [pseudo_label]
    for aug in aug_list:

        def convert_path(row):
            return convert_aug_path(row, aug, test_aug_version)
        
        pseudo_aug = pseudo_label.apply(convert_path, axis=1)
        pseudo_aug = pd.DataFrame(pseudo_aug, columns=["path",
                                                       "plnum",
                                                       "possible_label"])
        augmented_df.append(pseudo_aug)

    return pd.concat(augmented_df)

