from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import config
import utils


def ensembler(sub_list, sub_name, submission=True, weights=None):

    test_paths = Path(config.TEST_AUDIO_PATH).glob("*wav")
    test_paths = sorted([Path(path).parts[-1] for path in test_paths])
    test_paths = pd.Series(test_paths, name='fname')
    
    probs = []
    
    for sub in tqdm(sub_list):
        sub_cv_probs = []
        prob_paths = sorted(Path(sub).glob("*.csv"))
        for fold_probs in tqdm(prob_paths):
            fold_prob = pd.read_csv(fold_probs)
            fold_prob = fold_prob.sort_values(by="path")
            assert(all(test_paths.values == fold_prob.path.values))
            sub_cv_probs.append(fold_prob.drop("path", axis=1).values)
        probs.append(np.array(sub_cv_probs).mean(0))

    if weights is not None:
        ensembled_probs = np.array(probs) * weights[:, np.newaxis, np.newaxis]
    
    ensembled_probs = np.array(probs).mean(0)

    sub_probs = pd.DataFrame(ensembled_probs, columns=config.POSSIBLE_LABELS)
    sub_probs = pd.concat([test_paths, sub_probs], axis=1)
    sub_probs.to_csv("result/probs/{}.csv".format(sub_name),
                     index=False)

    if submission:
        ensemble_plnum = np.argmax(ensembled_probs, axis=1)
        ensemble_plnum = pd.Series(ensemble_plnum, name='label')
        ensemble_label = utils.id_to_label(ensemble_plnum)

        submission = pd.concat([test_paths, ensemble_label], axis=1)
        submission.to_csv("submit/{}.csv".format(sub_name),
                          index=False)


if __name__ == "__main__":

    sub_list = ["sub/VGG1Dv2/2018_01_08_19_37_23_VGG1Dv3_4017_2018_01_09_01_37_49",
                "sub/VGG1Dv2/2018_01_10_01_27_39_VGG1Dv2_4017_2018_01_10_14_47_56",
                "sub/VGG1Dv2/2018_01_09_19_22_27_VGG1Dv3_2017_2018_01_09_22_59_16",
                "sub/VGG1Dv2/2018_01_12_19_45_41_VGG1Dv2_5017_pseudo_STFTCNNv2_2018_01_12_01_39_43_STFTCNNv2_5017_2018_01_12_18_06_39_2018_01_13_12_29_16",
                "sub/STFTCNNv2/2018_01_12_01_39_43_STFTCNNv2_5017_2018_01_12_18_06_39",
                "sub/STFTCNN/2018_01_07_05_16_53",
                "sub/STFTCNNv2/2018_01_11_09_54_57_STFTCNNv2_3018_online_2018_01_12_02_48_47",
                "sub/STFTCNNv2/2018_01_13_13_31_43_VGG1Dv2_5017_pseudo_VGG1Dv2_2018_01_12_19_45_41_VGG1Dv2_5017_pseudo_STFTCNNv2_2018_01_12_01_39_43_STFTCNNv2_5017_2018_01_12_18_06_39_2018_01_13_12_29_16_2018_01_15_02_19_00",
                "sub/STFTCNNv2/2018_01_14_18_20_00_cv_2018_01_14_06_41_34_STFTCNNv2_3018_online_pseudo_VGG1Dv2_2018_01_10_22_50_50_VGG1Dv2_3018_online_2018_01_11_09_26_40_pseudosize_3000"]
    sub_name = utils.now()
    # sub_name = "material/STFTCNNv2_oversampling"
    # weights = [1, 1, 1, 1, 1, 1, 1, 1]
    ensembler(sub_list, sub_name)
    sub_name += ".csv"
    pd.Series(sub_list, name="material").to_csv(Path("ensemble")/sub_name,
                                                index=False)
