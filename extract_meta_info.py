import re
from pathlib import Path
import pandas as pd
import config


def extract_uid_and_nohash(wav_path):
    uid_pattern = r'^(\w+)_nohash_(\d+).wav'
    label = wav_path.parts[-2]
    
    if label == "_background_noise_":
        uid = 'No User'
        nohash = -1
    else:
        x = re.match(uid_pattern, wav_path.parts[-1])
        uid = x.group(1)
        nohash = x.group(2)
    
    return pd.Series({"absolute path": wav_path.absolute(),
                      "label": label,
                      "uid": uid,
                      "nohash": nohash})


def is_list(wav_path, file_list):
    return '/'.join(wav_path.parts[-2:]) in file_list


def possible_labeling(label, possible):
    if label == "_background_noise_":
        return "_background_noise_"
    elif label not in possible:
        return "unknown"
    else:
        return label


if __name__ == '__main__':
    audio_path = Path(config.TRAIN_AUDIO_PATH)
    train_path = Path(config.TRAIN_PATH)
    label2n = dict(zip(config.POSSIBLE_LABELS,
                       range(len(config.POSSIBLE_LABELS))))

    train_wavs = pd.Series(audio_path.glob('**/*.wav'), name="path")
    v = train_wavs.apply(extract_uid_and_nohash)
    train_file_info = pd.concat([train_wavs, v], axis=1)
    
    with open(Path.joinpath(train_path,
                            "validation_list.txt"), "r") as valid_list:
        valid_list = valid_list.readlines()
        
    valid_list = [fname.replace('\n', '') for fname in valid_list]

    def is_valid(path):
        return is_list(path, valid_list)

    def possible(label):
        return possible_labeling(label, config.POSSIBLE_LABELS)

    train_file_info["is_valid"] = train_file_info['path'].apply(is_valid)
    train_file_info["possible_label"] = train_file_info.label.apply(possible)
    train_file_info["plnum"] = train_file_info.possible_label.replace(label2n)
    
    train_file_info.to_csv("data/train_file_info.csv")
