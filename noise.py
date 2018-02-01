from pathlib import Path
import numpy as np
import config
import utils
import soundfile as sf
from tqdm import tqdm


"""
from kernel script below
https://www.kaggle.com/holzner/generating-different-colors-of-noise
"""


def normalize(samples):
    """normalizes a sample to unit standard 
    deviation (assuming the mean is zero)
    """
    std = samples.std()
    if std > 0:
        return samples / std
    else:
        return samples


def _gen_colored_noise(spectral_shape):
        # helper function generating a noise spectrum
        # and applying a shape to it
        flat_spectrum = np.random.normal(size=config.SAMPLE_RATE // 2 + 1) + \
                        1j * np.random.normal(size=config.SAMPLE_RATE // 2 + 1)

        return normalize(np.fft.irfft(flat_spectrum * spectral_shape).real)


def gen_noise(color, volume):
    
    assert(config.SAMPLE_RATE % 2 == 0)
    assert((volume >= 0) & (volume <= 1))
    spectrum_len = config.SAMPLE_RATE // 2 + 1
    
    if color == 'white':
        # flat in frequency
        
        # note that this needs to be normalized because
        # with std = 1 many samples will be outside +1/-1
        noise = np.random.normal(size=config.SAMPLE_RATE)

    elif color == 'pink':
        noise = _gen_colored_noise(1./(np.sqrt(np.arange(spectrum_len)+1.)))
    
    elif color == 'blue':
        noise = _gen_colored_noise(np.sqrt(np.arange(spectrum_len)))
    
    elif color == 'brown' or color == 'red':
        noise = _gen_colored_noise(1. / (np.arange(spectrum_len) + 1))
    
    elif color == 'violet' or color == 'purple':
        noise = _gen_colored_noise(np.arange(spectrum_len))
    
    else:
        raise Exception("unsupported noise color %s" % color)

    return noise * volume


if __name__ == "__main__":
    version = "noise_{}".format(utils.now())
    size = 3000
    
    noise_dir = Path(config.SILENCE_DATA_PATH)/version
    noise_dir.mkdir(parents=True, exist_ok=True)

    noise_types = ['white', 'blue', 'brown', 'red', 'violet']
    for nt in noise_types:
        print(nt)
        for i in tqdm(range(size)):
            fname = "{}_{}.wav".format(nt, i)
            vol = np.random.rand()
            wav = gen_noise(nt, vol)
            wav = np.clip(wav, -1, 1)
            sf.write(str(noise_dir/fname),
                     wav,
                     config.SAMPLE_RATE,
                     subtype='PCM_16')
