from pathlib import Path
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.callbacks import CSVLogger
from keras_tqdm import TQDMCallback
import utils
import config


class Learner():

    def __init__(self, model,
                 dump_path=None,
                 csv_log_path=None):

        version = utils.now()
        if dump_path is None:
            Path('model/{}'.format(model.name)).mkdir(exist_ok=True)
            self.dump_path = 'model/{}/{}.hdf5'.format(model.name, version)
        else:
            self.dump_path = dump_path

        if csv_log_path is None:
            Path('logs/{}'.format(model.name)).mkdir(exist_ok=True)
            self.csv_log_path = 'logs/{}/{}_log.csv'.format(model.name,
                                                            version)
        else:
            self.csv_log_path = csv_log_path

        self.model = model.model
        self.callbacks = [EarlyStopping(monitor='val_loss',
                                        patience=7,
                                        verbose=1,
                                        min_delta=0.00001,
                                        mode='min'),
                          ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.1,
                                            patience=4,
                                            verbose=1,
                                            epsilon=0.0001,
                                            mode='min'),
                          ModelCheckpoint(monitor='val_loss',
                                          filepath=self.dump_path,
                                          save_best_only=True,
                                          save_weights_only=True,
                                          mode='min'),
                          CSVLogger(self.csv_log_path),
                          TQDMCallback()]

    def learn(self, train_generator, valid_generator, validation_steps,
              steps_per_epoch,
              epochs=config.EPOCHS):
        history = self.model.fit_generator(generator=train_generator,
                                           steps_per_epoch=steps_per_epoch,
                                           epochs=epochs,
                                           verbose=0,
                                           callbacks=self.callbacks,
                                           validation_data=valid_generator,
                                           validation_steps=validation_steps)
        return history

    def predict(self, test_generator, steps):
        return self.model.predict_generator(test_generator, steps)
