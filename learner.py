from pathlib import Path
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.callbacks import CSVLogger
import utils


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
                                        patience=5,
                                        verbose=1,
                                        min_delta=0.01,
                                        mode='min'),
                          ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.1,
                                            patience=3,
                                            verbose=1,
                                            epsilon=0.01,
                                            mode='min'),
                          ModelCheckpoint(monitor='val_loss',
                                          filepath=self.dump_path,
                                          save_best_only=True,
                                          save_weights_only=True,
                                          mode='min'),
                          CSVLogger(self.csv_log_path)]

    def learn(self, train_generator, valid_generator, validation_steps,
              steps_per_epoch=344,
              epochs=20):
        history = self.model.fit_generator(generator=train_generator,
                                           steps_per_epoch=steps_per_epoch,
                                           epochs=epochs,
                                           callbacks=self.callbacks,
                                           validation_data=valid_generator,
                                           validation_steps=validation_steps)
        return history

    def predict(self, test_generator, steps):
        return self.model.predict_generator(test_generator, steps)
