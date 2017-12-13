import numpy as np
import scipy.signal as signal
import librosa


def change_volume(wav, rate):
    return wav*rate


def roll(wav, shift):
    return np.roll(wav, shift)


def shift(wav, shift):
    shifted = np.zeros_like(wav)
    if shift >= 0:
        shifted[shift:] += wav[:shift]
    else:
        shifted[:shift] += wav[shift:]
    return shifted
    

def strech(wav, rate=1):
    if len(wav) == 16000:
        raise("wav length is not 16000")

    input_length = 16000
    wav = librosa.effects.time_stretch(wav, rate)
    if len(wav) > input_length:
        wav = wav[:input_length]
    else:
        wav = np.pad(wav, (0, max(0, input_length - len(wav))), "constant")
        
    return wav


def add_whitenoise(wav, rate=0.005):
    wn = np.random.randn(len(wav))
    wav += rate*wn
    return wav


def add_pinknoise(wav, rate=0.005):
    NotImplemented


def distortion(wav, threshold, level):
    wav = np.clip(level*wav, -threshold, threshold)
    return wav


def mix_two_wav(wav1, wav2, mix_rate=0.5):
    wav = mix_rate*wav1 + (1-mix_rate)*wav2
    return wav


def lowpass_filter(cutoff, samples, sample_rate, numtaps=255):
    nyq_freq = sample_rate/2
    cutoff_normalized = cutoff/nyq_freq
    fir_filter = signal.firwin(numtaps, cutoff_normalized)
    filterred_samples = signal.lfilter(fir_filter, 1, samples)
    return filterred_samples


def patch_bg(wav, sample_rate, bgn):
    rem_len = sample_rate - len(wav)
    i = np.random.randint(0, len(bgn) - rem_len)
    silence_part = bgn[i:(i+sample_rate)]
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


def zero_padding_random(wav, sample_rate):
    """zero padding start and end with random split
    >>> wav = [1, 2, 3]
    >>> wav_padded = zero_padding_random(wav, 5)
    >>> wav_padded = [0, 1, 2, 3, 0]
    """
    rem_len = sample_rate - len(wav)
    j = np.random.randint(0, rem_len)

    wav = np.pad(wav,
                 (j, rem_len - j),
                 "constant")
    return wav


def clip_random(wav, sample_rate):
    if len(wav) <= sample_rate:
        raise("wav length is shorter than one sec.")

    rand = np.random.randint(0, len(wav) - sample_rate)
    return wav[rand:(rand+sample_rate)]


class Augmentation():

    def __init__(self, wav, sample_rate):
        self.wav = wav
        self.sample_rate = sample_rate
