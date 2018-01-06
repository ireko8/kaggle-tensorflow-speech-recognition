from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import config
import utils


def ensembler(sub_list, sub_name):
    probs = []
    for sub in tqdm(sub_list):
        prob_paths = Path(sub).glob("*.csv")
        for fold_probs in tqdm(prob_paths):
            fold_prob = pd.read_csv(fold_probs)
            probs.append(fold_prob.drop("path", axis=1).values)

    ensembled_probs = np.array(probs).mean(0)
    ensemble_plnum = np.argmax(ensembled_probs, axis=1)
    ensemble_plnum = pd.Series(ensemble_plnum, name='label')
    ensemble_label = utils.id_to_label(ensemble_plnum)

    test_paths = Path(config.TEST_AUDIO_PATH).glob("*wav")
    test_paths = [Path(path).parts[-1] for path in test_paths]
    test_paths = pd.Series(test_paths, name='fname')
    submission = pd.concat([test_paths, ensemble_label], axis=1)
    submission.to_csv("submit/{}.csv".format(sub_name),
                      index=False)


if __name__ == "__main__":

    sub_list = ["sub/VGG1Dv2/2018_01_06_17_40_23",
                "sub/STFTCNN/2018_01_02_13_39_20"]
    sub_name = utils.now()
    ensembler(sub_list, sub_name)
