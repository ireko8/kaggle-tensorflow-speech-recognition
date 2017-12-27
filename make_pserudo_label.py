import pandas as pd


def make_pseudo_label(cv_version, fold):
    cv_res = []
    for i in range(5):
        if i != fold:
            res = pd.read_csv("cv/VGG1D/{}/fold_{}_test.csv".format(cv_version,
                                                                    i))
            cv_res.append(res)
    
