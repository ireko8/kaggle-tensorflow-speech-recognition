from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D
from tensorflow.python.keras.layers import Activation, BatchNormalization
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from tensorflow.python.keras.layers import GlobalMaxPool2D, GlobalMaxPool1D
from tensorflow.python.keras.layers import concatenate, Dense, Dropout
import config


class STFTCNN():

    def __init__(self,
                 name="STFTCNN"):

        self.name = name

    def model_init(self, input_shape=(257, 98, 2)):
        
        x_in = Input(shape=input_shape)
        x = BatchNormalization()(x_in)
        for i in range(4):
            x = Conv2D(16*(2 ** i), (3, 3))(x)
            x = Activation('elu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (1, 1))(x)
        x_branch_1 = GlobalAveragePooling2D()(x)
        x_branch_2 = GlobalMaxPool2D()(x)
        x = concatenate([x_branch_1, x_branch_2])
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(len(config.POSSIBLE_LABELS), activation='softmax')(x)
        model = Model(inputs=x_in, outputs=x)
        model.compile(optimizer='nadam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model


class VGG1D():
    """1d VGG16 like convolution model
    """
    def __init__(self, name="VGG1D"):
        self.name = name

    def model_init(self, input_shape=(config.SAMPLE_RATE, 1)):
        x_in = Input(shape=input_shape)
        x = BatchNormalization()(x_in)
        for i in range(6):
            x = Conv1D(8*(2 ** i), 16,
                       strides=2,
                       padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(2, padding="same")(x)

        x = Conv1D(8*(2 ** 5), 4,
                   strides=1,
                   padding="same")(x)

        x_branch_1 = GlobalAveragePooling1D()(x)
        x_branch_2 = GlobalMaxPool1D()(x)
        x = concatenate([x_branch_1, x_branch_2])
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(len(config.POSSIBLE_LABELS), activation='softmax')(x)
        model = Model(inputs=x_in, outputs=x)
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model


class VGG1Dv2():
    """1d VGG16 like convolution model
    """
    def __init__(self, name="VGG1Dv2"):
        self.name = name

    def model_init(self, input_shape=(config.SAMPLE_RATE, 1)):
        x_in = Input(shape=input_shape)
        x = BatchNormalization()(x_in)
        for i in range(6):
            if i <= 3:
                ks = 16
                ks2 = 8
                st = 2
                st2 = 1
            else:
                ks = 8
                ks2 = 8
                st = 1
                st2 = 1
            x = Conv1D(2**(i+3), ks,
                       strides=st,
                       padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization()(x)
            x = Conv1D(2**(i+3), ks2,
                       strides=st2,
                       padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization()(x)

            if i >= 4:
                x = Conv1D(2**(i+3), ks2,
                           strides=st2,
                           padding="same")(x)
                x = Activation("relu")(x)
                x = BatchNormalization()(x)
                x = Conv1D(2**(i+3), ks2,
                           strides=st2,
                           padding="same")(x)
                x = Activation("relu")(x)
                x = BatchNormalization()(x)
                
            x = MaxPooling1D(2, padding="same")(x)

        x = Conv1D(2**9, 4,
                   strides=1,
                   padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)

        x_branch_1 = GlobalAveragePooling1D()(x)
        x_branch_2 = GlobalMaxPool1D()(x)
        x = concatenate([x_branch_1, x_branch_2])
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(len(config.POSSIBLE_LABELS), activation='softmax')(x)
        model = Model(inputs=x_in, outputs=x)
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model


class VGG1Dv3():
    """1d VGG16 like convolution model
    """
    def __init__(self, name="VGG1Dv3"):
        self.name = name

    def model_init(self, input_shape=(config.SAMPLE_RATE, 1)):

        def Conv1Dbn(x, conv_count, ks, ch, st=1, dropout=None):
            if dropout:
                x = Dropout(dropout)(x)
            for c in range(conv_count):
                x = Conv1D(ch, ks,
                           strides=st,
                           padding="same")(x)
                x = Activation("relu")(x)
                x = BatchNormalization()(x)

            x = MaxPooling1D(2, padding="same")(x)
            return x
        
        x_in = Input(shape=input_shape)
        x = Conv1Dbn(x_in, 1, 16, 8, st=2)
        x = Conv1Dbn(x, 2, 3, 16)
        x = Conv1Dbn(x, 2, 3, 32)
        x = Conv1Dbn(x, 2, 3, 64, dropout=0.1)
        x = Conv1Dbn(x, 4, 3, 128, dropout=0.1)
        x = Conv1Dbn(x, 4, 3, 256, dropout=0.1)
        x = Conv1Dbn(x, 4, 3, 512, dropout=0.1)
        x = Conv1Dbn(x, 2, 3, 1024, dropout=0.1)

        x = GlobalAveragePooling1D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(len(config.POSSIBLE_LABELS), activation='softmax')(x)
        
        model = Model(inputs=x_in, outputs=x)
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model
        

class MelSpectCNN():

    def __init__(self,
                 name="MelSpectCNN"):

        self.name = name

    def model_init(self, input_shape=(128, 32, 1)):
        
        x_in = Input(shape=input_shape)
        x = BatchNormalization()(x_in)
        for i in range(4):
            x = Conv2D(16*(2 ** i), (3, 3))(x)
            x = Activation('elu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (1, 1))(x)
        x_branch_1 = GlobalAveragePooling2D()(x)
        x_branch_2 = GlobalMaxPool2D()(x)
        x = concatenate([x_branch_1, x_branch_2])
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(len(config.POSSIBLE_LABELS), activation='softmax')(x)
        model = Model(inputs=x_in, outputs=x)
        model.compile(optimizer='nadam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model
