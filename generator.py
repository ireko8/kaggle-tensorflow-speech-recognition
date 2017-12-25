import random
import numpy as np
import scipy.signal as signal
from tensorflow.python.keras.utils import to_categorical
import librosa
import augment
import config
import utils


def read_wav_file(fname):
    wav, sample_rate = librosa.core.load(fname,
                                         sr=config.SAMPLE_RATE)
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
                    bgn_paths,
                    mode='train',
                    sampling_size=2000):

    def preprocess(path):
        wav = process_wav_file(path)
        return [np.array(wav).reshape((config.SAMPLE_RATE, 1))]
        
    while True:
        if mode == 'train':
            grouped = input_df.groupby('plnum')
            base_df = grouped.apply(lambda x: x.sample(n=sampling_size))
            base_df_id = random.sample(range(base_df.shape[0]),
                                       base_df.shape[0])
            # print("base_df shape: ", base_df.shape)
        else:
            base_df = input_df
            base_df_id = list(range(input_df.shape[0]))
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
