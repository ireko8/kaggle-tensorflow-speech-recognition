from pathlib import Path
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Input
from tensorflow.python.keras.layers import BatchNormalization, UpSampling1D
from tensorflow.python.keras.layers import LeakyReLU
from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
import config
import generator


class AutoEncoder1D():
    """1d AutoEncoder
    """
    def __init__(self,
                 name="AutoEncoder1D",
                 encoded_dim=100):
        self.name = name

    def model_init(self, input_shape=(config.SAMPLE_RATE, 1)):
        x_in = Input(shape=input_shape)
        x = BatchNormalization()(x_in)
        for i in range(3):
            x = Conv1D(8*(2 ** (3-i)), 4,
                       dilation_rate=2**i,
                       padding="causal")(x)
            x = LeakyReLU()(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(4)(x)

        encoded = x

        for i in range(3):
            x = Conv1D(8*(2 ** i), 4,
                       dilation_rate=2**(3-i),
                       padding='causal')(x)
            x = LeakyReLU()(x)
            x = BatchNormalization()(x)
            x = UpSampling1D(2)(x)
        model = Model(inputs=x_in, outputs=x)
        model.compile(optimizer='rmsprop',
                      loss='mean_squared_error')
        self.autoencoder = model
        self.encoder = Model(x_in, encoded)


def batch_generator_for_ae(input_df, batch_size, mode='train'):

    def add_noise(path):
        fname = Path(path).parts[:-2]
        aug_path = config.AUG_PATH/config.AUG_VERSION
        noised_path = aug_path/"add_wn"/fname
        wav = generator.process_wav_file(noised_path)
        return wav

    while(True):
        if mode == 'train':
            input_df = input_df.sample(frac=1).reset_index(drop=True)

        for start in range(0, len(input_df), batch_size):
            end = min(start + batch_size, len(input_df))
            batch = input_df[start:end]
            x_batch = batch.path.apply(generator.process_wav_file)
            if mode == 'train':
                y_batch = batch.path.apply(add_noise)
                yield x_batch, y_batch
            else:
                yield x_batch


def train_autoencoder(X, train_percent, batch_size):
    ae = AutoEncoder1D()
    ae.model_init()
    train_size = int(len(X) * train_percent)
    X_train, X_valid = X[:train_size], X[train_size:]
    train_generator = batch_generator_for_ae(X_train, batch_size)
    valid_generator = generator.batch_generator(X_valid, batch_size,
                                                mode="valid")

    valid_steps = int(np.ceil(X_valid.shape[0]/batch_size))
    steps_per_epoch = int(np.ceil(X_train/batch_size))

    learn = learner.Learner(ae)
    result = learn.learn(train_generator,
                         valid_generator,
                         valid_steps,
                         steps_per_epoch=steps_per_epoch)
    return result, ae

        
def tsne_mapping(X):
    tsne = TSNE()
    y = tsne.fit_transform(X)
    return y
