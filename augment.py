import numpy as np
import scipy.signal as signal
import librosa


def roll(wav, shift):
    return np.roll(wav, shift)


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
    fir_filter = signal.firwin(numtaps, cutoff_normalized, pass_zero=False)
    filterred_samples = signal.lfilter(fir_filter, 1, samples)
    return filterred_samples


def patch_bg(wav, samplerate, bgn):
    rem_len = sample_rate - len(wav)
    i = np.random.randint(0, len(silence_data) - rem_len)
    silence_part = bgn[i:(i+sample_rate)]
    j = np.random.randint(0, rem_len)
    silence_part_left = bgn[0:j]
    silence_part_right = bgn[j:rem_len]
    wav = np.concatenate([silence_part_left,
                          wav,
                          silence_part_right])
    return wav
