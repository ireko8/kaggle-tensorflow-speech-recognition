from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
import scipy.signal as signal
import librosa
from tqdm import tqdm
import soundfile as sf
import config
import utils
import generator
import experiment
import noise
import submit


def change_volume(wav, rate):
    return wav*rate


def roll(wav, shift):
    return np.roll(wav, shift)


def pitch_shift(wav, pitch):
    wav = librosa.effects.pitch_shift(wav,
                                      config.SAMPLE_RATE,
                                      n_steps=pitch)
    return wav


def zero_padding_random(wav, sample_rate):
    rem_len = sample_rate - len(wav)
    j = np.random.randint(0, rem_len)

    wav = np.pad(wav,
                 (j, rem_len - j),
                 "constant")
    return wav


def shift(wav, shift):
    shifted = np.zeros_like(wav)
    while shift == 0:
        shift = np.random.randint(config.SHIFT_MIN,
                                  config.SHIFT_MAX)
        
    if shift >= 0:
        shifted[shift:] += wav[:-shift]
    else:
        shifted[:shift] += wav[-shift:]
    return shifted
    

def strech(wav, rate=1):
    if len(wav) != config.SAMPLE_RATE:
        raise("wav length is not 1s")

    input_length = 16000
    wav = librosa.effects.time_stretch(wav, rate)
    if len(wav) > input_length:
        start = np.random.randint(len(wav) - input_length)
        wav = wav[start:input_length+start]
    else:
        wav = zero_padding_random(wav, config.SAMPLE_RATE)

    assert(len(wav) == config.SAMPLE_RATE)
        
    return wav


def add_whitenoise(wav, rate=0.005):
    wn = np.random.randn(len(wav))
    wav = (1-rate)*wav + rate*wn
    return np.clip(wav, -1, 1).astype(np.float32)


def add_noise(nt):
    def _add_noise(wav, rate=0.005):
        wav = (1-rate)*wav + noise.gen_noise(nt, rate)
        return np.clip(wav, -1, 1).astype(np.float32)
    return _add_noise


def distortion(wav, threshold, level):
    wav = np.clip(level*wav, -threshold, threshold)
    return wav


def mix_two_wav(wav1, wav2, mix_rate=0.5):
    wav = (1-mix_rate)*wav1 + mix_rate*wav2
    return wav


def mix_bgn_wav(wav, bgn, mix_rate=0.005):
    i = np.random.randint(0, len(bgn) - len(wav))
    bgn_cut = bgn[i:i+len(wav)]
    wav = (1-mix_rate)*wav + mix_rate*bgn_cut
    return wav


def lowpass_filter(samples, cutoff,
                   sample_rate=config.SAMPLE_RATE,
                   numtaps=255):
    nyq_freq = sample_rate/2
    cutoff_normalized = cutoff/nyq_freq
    fir_filter = signal.firwin(numtaps, cutoff_normalized)
    filterred_samples = signal.lfilter(fir_filter, 1, samples)
    return filterred_samples


def patch_bg_random(wav, sample_rate, bgn):
    rem_len = sample_rate - len(wav)
    i = np.random.randint(0, len(bgn) - rem_len)
    silence_part = bgn[i:(i+rem_len)]
    j = np.random.randint(0, rem_len)
    silence_part_left = silence_part[0:j]
    silence_part_right = silence_part[j:rem_len]
    wav = np.concatenate([silence_part_left,
                          wav,
                          silence_part_right])
    return wav


def zero_padding(wav, sample_rate):
    if len(wav) >= sample_rate:
        raise("wav length is longer than one seconds")
    
    wav = np.pad(wav,
                 (0, sample_rate - len(wav)),
                 "constant")

    return wav


def clip_random(wav, sample_rate):
    if len(wav) <= sample_rate:
        raise("wav length is shorter than one sec.")

    rand = np.random.randint(0, len(wav) - sample_rate)
    return wav[rand:(rand+sample_rate)]


class Augment():

    def __init__(self, bgn, augment_list):
        self.bgn = bgn

        self.augment_list = augment_list
        self.augment_df = pd.DataFrame(augment_list, columns=["aug_name"])

        vol_up = partial(change_volume, rate=config.VOLUME_UP)
        vol_down = partial(change_volume, rate=config.VOLUME_DOWN)
        shift_forward = utils.rand_decorator("shift",
                                             start=config.SHIFT_MIN,
                                             end=config.SHIFT_MAX,
                                             integer=True)(shift)
        shift_backward = utils.rand_decorator("shift",
                                              start=-config.SHIFT_MAX,
                                              end=-config.SHIFT_MIN,
                                              integer=True)(shift)

        # speed_up = utils.rand_decorator("rate",
        #                                 start=config.SPEED_UP_MIN,
        #                                 end=config.SPEED_UP_MAX)(strech)

        # speed_down = utils.rand_decorator("rate",
        #                                   start=config.SPEED_DOWN_MIN,
        #                                   end=config.SPEED_DOWN_MAX)(strech)
        speed_up = partial(strech, rate=config.SPEED_DOWN_MAX)
        speed_down = partial(strech, rate=config.SPEED_DOWN_MIN)

        # pitch_up = utils.rand_decorator("pitch",
        #                                 start=config.PITCH_MIN,
        #                                 end=config.PITCH_MAX)(pitch_shift)
        pitch_up = partial(pitch_shift, pitch=config.PITCH_MIN)
        pitch_down = partial(pitch_shift, pitch=config.PITCH_MAX)
        
        add_wn = utils.rand_decorator("rate",
                                      start=config.ADD_WN_MIN,
                                      end=config.ADD_WN_MAX)(add_whitenoise)
        add_wn2 = utils.rand_decorator("rate",
                                       start=0.1,
                                       end=0.5)(add_whitenoise)

        add_bn = utils.rand_decorator("rate",
                                      start=config.ADD_WN_MIN,
                                      end=config.ADD_WN_MAX)(add_noise("blue"))
        add_brn = utils.rand_decorator("rate",
                                       start=config.ADD_WN_MIN,
                                       end=config.ADD_WN_MAX)(add_noise("brown"))
        add_vn = utils.rand_decorator("rate",
                                      start=config.ADD_WN_MIN,
                                      end=config.ADD_WN_MAX)(add_noise("violet"))
        add_rn = utils.rand_decorator("rate",
                                      start=config.ADD_WN_MIN,
                                      end=config.ADD_WN_MAX)(add_noise("red"))

        patch_bg = partial(patch_bg_random, sample_rate=config.SAMPLE_RATE,
                           bgn=bgn)
        mix_bgn = partial(mix_bgn_wav, bgn=bgn, mix_rate=config.MIX_BGN_RATE)
        mix_random = partial(utils.rand_decorator("mix_rate",
                                                  start=config.MIX_BGN_RATE,
                                                  end=config.MIX_BGN_MAX)
                             (mix_bgn_wav),
                             bgn=bgn)
        
        lp_2000 = partial(lowpass_filter,
                          cutoff=2000,
                          sample_rate=config.SAMPLE_RATE)
        lp_4000 = partial(lowpass_filter,
                          cutoff=4000,
                          sample_rate=config.SAMPLE_RATE)
        lp_6000 = partial(lowpass_filter,
                          cutoff=6000,
                          sample_rate=config.SAMPLE_RATE)
        lp_random = utils.rand_decorator("cutoff",
                                         start=config.LP_MIN,
                                         end=config.LP_MAX,
                                         integer=True)(lowpass_filter)

        abbrev_func_map = {"id": (lambda x: x),
                           "vol_up": vol_up,
                           "vol_down": vol_down,
                           "shift_forward": shift_forward,
                           "shift_backward": shift_backward,
                           "speed_up": speed_up,
                           "speed_down": speed_down,
                           "pitch_up": pitch_up,
                           "pitch_down": pitch_down,
                           "add_wn": add_wn,
                           "add_wn2": add_wn2,
                           "add_bn": add_bn,
                           "add_brn": add_brn,
                           "add_rn": add_rn,
                           "add_vn": add_vn,
                           "patch_bg": patch_bg,
                           "mix_bgn": mix_bgn,
                           "mix_random": mix_random,
                           "lp_2000": lp_2000,
                           "lp_4000": lp_4000,
                           "lp_6000": lp_6000,
                           "lp_random": lp_random}
        
        self.abbrev_func_map = abbrev_func_map

    def dump(self, paths, directory):
        dir_path = Path("./data/augment/" + directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        def aug_file(path, aug, out_path):
            wav = generator.process_wav_file(path)
            wav = self.abbrev_func_map[aug](wav)
            
            fname = Path(path).parts[-1]
            label = Path(path).parts[-2]
            Path(out_path/aug/label).mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path/aug/label/fname),
                     wav,
                     config.SAMPLE_RATE,
                     subtype='PCM_16')

        tqdm.pandas(desc='augment')
        for aug in self.augment_list:
            print(aug)
            paths.path.progress_apply(lambda x: aug_file(x, aug, dir_path))


if __name__ == "__main__":
    utils.set_seed(2017)
    
    sdata = config.SILENCE_DATA_VERSION
    train_paths, bgn_paths, silence_paths = experiment.data_load(sdata)
    # test_paths, bgn_paths = submit.test_data_load()

    # bgn_paths = bgn_paths[~bgn_paths.path.str.contains("white")]
    bgn_data = [generator.read_wav_file(x)[1] for x in bgn_paths.path]
    bgn_data = np.concatenate(bgn_data)

    directory = "{}_param_fixed".format(utils.now())

    aug_class = Augment(bgn_data, ["pitch_up",
                                   "pitch_down",
                                   "speed_up",
                                   "speed_down"])

    # print('test augmentation')
    # aug_class.dump(test_paths, directory)
    
    print('train augmentation')
    aug_class.dump(train_paths, directory)
    print('silence augmentation')
    aug_class.dump(silence_paths, directory)
