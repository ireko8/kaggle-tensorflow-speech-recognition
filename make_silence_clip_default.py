import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import librosa
import soundfile as sf
import config
import utils


def silence_data_load():
    file_df = pd.read_csv(config.TRAIN_FILE_META_INFO)
    silence_df = file_df[file_df.possible_label == "_background_noise_"]
    return silence_df


if __name__ == "__main__":
    utils.set_seed(2017)
    
    silence_df = silence_data_load()
    
    silence_data = [librosa.core.load(x)[0] for x in silence_df.path]
    print([len(x) for x in silence_data])
    silence_data = np.concatenate(silence_data)
    length = config.SAMPLE_RATE
    
    version = utils.now()
    
    size = 20000
    dir_path = Path("data/silence/{}".format(version))
    dir_path.mkdir(exist_ok=True, parents=True)
    
    path_list = [dir_path/"simple_slice_{}.wav".format(i) for i in range(size)]
    uid_list = ["Nothing" for _ in range(size)]
    possible_label_list = ["silence" for _ in range(size)]
    plnum_list = [10 for _ in range(size)]
    
    for i in tqdm(range(size)):
        if i % 1000 == 0:
            wav = np.zeros(length)
        else:
            start = np.random.randint(0, len(silence_data) - length)
            if i % 10 == 0:
                volume = 1
            else:
                volume = np.random.random()
            wav = volume*silence_data[start:start+length]
        sf.write(str(path_list[i]),
                 wav,
                 config.SAMPLE_RATE,
                 subtype='PCM_16')

    file_info = {"path": path_list,
                 "possible_label": possible_label_list,
                 "uid": uid_list,
                 "plnum": plnum_list}
    pd.DataFrame(file_info).to_csv(dir_path/"file_info.csv",
                                   index=False)
