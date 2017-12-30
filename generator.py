import random
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
from tensorflow.python.keras.utils import to_categorical
import augment
import config


def read_wav_file(fname):
    sample_rate, wav = wavfile.read(fname)
    wav = wav.astype(np.float32)/np.iinfo(np.int16).max
    return sample_rate, wav


def process_wav_file(fname):
    sample_rate, wav = read_wav_file(fname)

    if len(wav) > sample_rate:
        wav = augment.clip_random(wav, sample_rate)
    elif len(wav) < sample_rate:
        wav = augment.zero_padding_random(wav, sample_rate)

    return wav


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


def batch_generator(input_df, batch_size, category_num,
                    mode='train'):

    def preprocess(path):
        wav = process_wav_file(path)
        # return wav_to_spct(wav)
        return [np.array(wav).reshape((config.SAMPLE_RATE, 1))]

    while True:
        base_df = input_df
        if mode == "train":
            base_df_id = random.sample(range(base_df.shape[0]),
                                       base_df.shape[0])
        else:
            base_df_id = list(range(base_df.shape[0]))

        for start in range(0, len(base_df), batch_size):
            end = min(start + batch_size, len(base_df))
            batch_df_id = base_df_id[start:end]
            batch_df = base_df.iloc[batch_df_id]

            x_batch = batch_df.path.apply(preprocess).values
            x_batch = np.concatenate(x_batch)

            if mode != 'test':
                y_batch = batch_df.plnum.values
                y_batch = to_categorical(y_batch, num_classes=category_num)
                yield x_batch, y_batch
            else:
                yield x_batch
