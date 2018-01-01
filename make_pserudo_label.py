import pandas as pd
import numpy as np


def make_pseudo_labeling(cv_version, fold):
    test_paths = Path(config.TEST_AUDIO_PATH).glob("*wav")
    test_paths = pd.DataFrame(test_paths, columns=["path"])
    cv_res = []
    for i in range(5):
        if i != fold:
            res = pd.read_csv("cv/{}/fold_{}_test.csv".format(cv_version,
                                                              i))
            cv_res.append(res.iloc[-12:])
    cv_probs = np.array(cv_res)
    cv_mean = np.mean(cv_probs, axis=0)
    pseudo_plnum = np.argmax(, axis=0)
    pseudo_label = pd.concat([test_paths, pseudo_plnum])
