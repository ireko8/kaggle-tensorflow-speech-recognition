from pathlib import Path
import numpy as np
import pandas as pd
import generator
import config
import model


def test_data_load():
    file_df = pd.read_csv(config.TRAIN_FILE_META_INFO)
    df = file_df[["path", "uid", "possible_label", "plnum"]]
    silence_paths = df[df["possible_label"] == "silence"]
    
    test_paths = Path(config.TEST_AUDIO_PATH).glob("*wav")
    test_paths = pd.DataFrame(test_paths, columns=["path"])
    return test_paths, silence_paths


if __name__ == '__main__':

    batch_size = 64
    test_paths, silence_paths = test_data_load()
    test_generator = generator.batch_generator(test_paths,
                                               batch_size,
                                               len(config.POSSIBLE_LABELS),
                                               silence_paths,
                                               mode='test')
    id2name = dict(zip(range(len(config.POSSIBLE_LABELS)),
                       config.POSSIBLE_LABELS))

    cnn = model.STFTCNN()
    cnn.model.load_weights('./model/SpectCNN/2017:12:06:23:30:17.hdf5')
    steps = int(np.ceil(len(test_paths)/batch_size))
    predict_probs = cnn.model.predict_generator(test_generator, steps)

    predict_cls = np.argmax(predict_probs, axis=1)

    submission = dict()
    import ipdb; ipdb.set_trace()

    for i in range(len(test_paths)):
        fname = Path(test_paths.iloc[i][0]).parts[-1]
        label = id2name[predict_cls[i]]
        submission[fname] = label

    with open('submit/starter_submission.csv', 'w') as fout:
        fout.write('fname,label\n')
        for fname, label in submission.items():
            fout.write('{},{}\n'.format(fname, label))
