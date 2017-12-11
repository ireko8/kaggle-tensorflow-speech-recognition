import random
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
from tensorflow.python.keras.utils import to_categorical
import augment


def read_wav_file(fname):
    sample_rate, wav = wavfile.read(fname)
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    return wav, sample_rate


def process_wav_file(fname, bgn_data):
    wav, sample_rate = read_wav_file(fname)

    if len(wav) > sample_rate:
        wav = augment.clip_random(wav, sample_rate)
    elif len(wav) < sample_rate:
        wav = augment.zero_padding_random(wav, sample_rate)
        
    specgram = signal.stft(wav, sample_rate,
                           nperseg=400,
                           noverlap=240,
                           nfft=512,
                           padded=False,
                           boundary=None)

    phase = np.angle(specgram[2]) / np.pi
    amp = np.log1p(np.abs(specgram[2]))
    
    return np.stack([phase, amp], axis=2)


def batch_generator(input_df, batch_size, category_num, bgn_paths,
                    mode='train',
                    sampling_size=2000):
    
    bgn_data = [read_wav_file(x)[0] for x in bgn_paths.path]
    bgn_data = np.concatenate(bgn_data)
    
    def preprocess(wav_file):
        return process_wav_file(wav_file, bgn_data)
    
    while True:
        if mode == 'train':
            grouped = input_df.groupby('plnum')
            base_df = grouped.apply(lambda x: x.sample(n=sampling_size))
            base_df_id = random.sample(range(base_df.shape[0]),
                                       base_df.shape[0])
        else:
            base_df = input_df
            base_df_id = list(range(input_df.shape[0]))
        for start in range(0, len(base_df), batch_size):
            end = min(start + batch_size, len(base_df))
            batch_df_id = base_df_id[start:end]
            batch_df = base_df.iloc[batch_df_id]

            x_batch = np.stack(batch_df.path.apply(preprocess).values)
            if mode != 'test':
                y_batch = batch_df.plnum.values
                y_batch = to_categorical(y_batch, num_classes=category_num)
                yield x_batch, y_batch
            else:
                yield x_batch
