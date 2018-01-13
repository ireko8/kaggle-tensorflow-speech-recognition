import random
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import librosa
from tensorflow.python.keras.utils import to_categorical
import augment
import config
import utils
import noise


def read_wav_file(fname):
    sample_rate, wav = wavfile.read(fname)
    wav = wav.astype(np.float32)/np.iinfo(np.int16).max
    return sample_rate, wav


def process_wav_file(fname,
                     aug_name=None,
                     aug_class=None,
                     online=False,
                     bgn=None):
    sample_rate, wav = read_wav_file(fname)

    if len(wav) > sample_rate:
        wav = augment.clip_random(wav, sample_rate)
    elif len(wav) < sample_rate:
        wav = augment.zero_padding_random(wav, sample_rate)

    if online:
        if random.random() < 0.5:
            shift = np.random.randint(config.SHIFT_MIN,
                                      config.SHIFT_MAX)
            wav = augment.shift(wav, shift)
            
        if random.random() < 0.5:
            wav *= -1

        if random.random() < 0.5:
            noise_vol = np.random.rand() * config.MIX_BGN_MAX
            wav = augment.mix_bgn_wav(wav, bgn, mix_rate=noise_vol)

    if aug_name:
        wav = aug_class.abbrev_func_map[aug_name](wav)
        wav = np.clip(wav, -1, 1)

    return np.clip(wav, -1, 1)


def wav_to_spct(wav, sample_rate=config.SAMPLE_RATE):
    specgram = signal.stft(wav, sample_rate,
                           nperseg=400,
                           noverlap=240,
                           nfft=512,
                           padded=False,
                           boundary=None)

    phase = np.angle(specgram[2]) / np.pi
    amp = np.log1p(np.abs(specgram[2]))

    assert(np.stack([phase, amp], axis=2).dtype == np.float32)
    
    return [np.stack([phase, amp], axis=2)]


def wav_to_melspct(wav, sample_rate=config.SAMPLE_RATE):
    melspct = librosa.feature.melspectrogram(y=wav, sr=config.SAMPLE_RATE)
    assert(melspct.shape == (128, 32))
    return melspct


def batch_generator(input_df, batch_size, category_num,
                    online=False,
                    bgn_paths=None,
                    oversampling=False,
                    sampling_size=None,
                    mode='train'):
    
    if online:
        print("online")
        # bgn_paths = bgn_paths[~bgn_paths.path.str.contains("white")]
        bgn_data = [read_wav_file(x)[1] for x in bgn_paths.path]
        for nt in config.NOISE_TYPE:
            nt_noise = []
            for _ in range(120):
                nt_noise.append(noise.gen_noise(nt, 1))
        bgn_data = np.concatenate(bgn_data)

    def online_aug_preprocess(row):
        wav = process_wav_file(row.path, bgn=bgn_data, online=True)
        assert(len(wav) == config.SAMPLE_RATE)
        # return wav_to_spct(wav)
        return [np.array(wav).reshape((config.SAMPLE_RATE, 1))]

    def preprocess(path):
        wav = process_wav_file(path)
        # return wav_to_spct(wav)
        return [np.array(wav).reshape((config.SAMPLE_RATE, 1))]

    while True:
        base_df = input_df
        if mode == "train":
            if oversampling:
                grouped = input_df.groupby('plnum')
                base_df = grouped.apply(lambda x: x.sample(n=sampling_size))
            base_df_id = random.sample(range(base_df.shape[0]),
                                       base_df.shape[0])
        else:
            base_df_id = list(range(base_df.shape[0]))

        for start in range(0, len(base_df), batch_size):
            end = min(start + batch_size, len(base_df))
            batch_df_id = base_df_id[start:end]
            batch_df = base_df.iloc[batch_df_id]

            if online:
                x_batch_df = batch_df.apply(online_aug_preprocess,
                                            axis=1).values
            else:
                x_batch_df = batch_df.path.apply(preprocess).values

            x_batch = np.concatenate(x_batch_df)
                
            if len(x_batch.shape) != 3:
                x_batch = np.stack(x_batch_df[:, 0])

            while len(x_batch.shape) != 3:
                import ipdb; ipdb.set_trace()
                if online:
                    x_batch = batch_df.apply(online_aug_preprocess,
                                             axis=1).values
                else:
                    x_batch = batch_df.path.apply(preprocess).values
            
                x_batch = np.concatenate(x_batch)

            if mode != 'test':
                y_batch = batch_df.plnum.values
                y_batch = to_categorical(y_batch, num_classes=category_num)
                yield x_batch, y_batch
            else:
                yield x_batch
