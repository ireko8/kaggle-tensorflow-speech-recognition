from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, AvgPool1D
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.python.keras.layers import Activation, BatchNormalization
from tensorflow.python.keras.layers import GlobalAveragePooling2D, add
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
                for _ in range(3):
                    sc = x
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

        init_filter_num = 8
        x_in_1d = Input(shape = (16000,1))
        x_1d = BatchNormalization(name = 'batchnormal_1d_in')(x_in_1d)
        for i in range(9):
            name = 'step'+str(i)
            x_1d = Conv1D(init_filter_num*(2 ** i), (3),padding = 'same', name = 'conv'+name+'_1')(x_1d)
            x_1d = BatchNormalization(name = 'batch'+name+'_1')(x_1d)
            if i !=0:
                x_1d = add([x_1d_concate, x_1d])
            x_1d = Activation('relu')(x_1d)
            for j in range(2,14):
                short_cut = x_1d
                x_1d = Conv1D(init_filter_num*(2 ** i), (3),padding = 'same', name = 'conv'+name+'_'+str(j))(x_1d)
                x_1d = BatchNormalization(name = 'batch'+name+'_'+str(j))(x_1d)
                x_1d = add([short_cut,x_1d])
                x_1d = Activation('relu')(x_1d)
            x_1d_max = MaxPooling1D((2), padding='same')(x_1d)
            x_1d_avg = AvgPool1D((2), padding='same')(x_1d)
            x_1d = add([x_1d_max, x_1d_avg])
            x_1d = BatchNormalization(name = 'batch'+name+'_avgmax_add')(x_1d)
            x_1d = Activation('relu')(x_1d)
            if i != 8:
                x_1d_concate = concatenate([x_1d_max, x_1d_avg])
                x_1d_concate = BatchNormalization(name = 'batch'+name+'_avgmax_concate')(x_1d_concate)
        x_1d = Conv1D(1024, (1),name='last1024')(x_1d)
        x_1d = GlobalMaxPool1D()(x_1d) #only g max
        x_1d = Dense(1024, activation = 'relu', name= 'dense1024_onlygmax')(x_1d)
        x_1d = Dropout(0.2)(x_1d)
        x_1d = Dense(len(config.POSSIBLE_LABELS), activation = 'softmax',name='cls_1d')(x_1d)
        model = Model(inputs=x_in_1d, outputs=x_1d)
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


class STFTCNNv2():

    def __init__(self,
                 name="STFTCNNv2"):

        self.name = name

    def model_init(self, input_shape=(257, 98, 2)):
        
        x_in = Input(shape=input_shape)
        x = Conv2D(16, (3, 3), padding="same")(x_in)        
        x = Activation('relu')(x)
        x = BatchNormalization()(x_in)
        x = MaxPooling2D((2, 2))(x)
        for i in range(4):
            kernel_size = 32*(2 ** i)
            x = Conv2D(kernel_size, (3, 3), padding="same")(x)
            x = Activation('relu')(x)
            x = BatchNormalization()(x)
            x = Conv2D(kernel_size, (3, 3), padding="same")(x)
            x = Activation('relu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
        x = Conv2D(256, (1, 1))(x)
        x_branch_1 = GlobalAveragePooling2D()(x)
        x_branch_2 = GlobalMaxPool2D()(x)
        x = concatenate([x_branch_1, x_branch_2])
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(len(config.POSSIBLE_LABELS), activation='softmax')(x)
        model = Model(inputs=x_in, outputs=x)
        model.compile(optimizer='nadam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model


class MFCCNN():

    def __init__(self,
                 name="MFCCNN"):
        
        self.name = name
        
    def model_init(self, input_shape=(1, 40, 101)):
        
        x_in = Input(shape=input_shape)
        x = x_in
        for i in range(3):
            x = Conv2D(8*(2 ** i), (3, 3),
                       data_format="channels_first",
                       padding="same")(x)
            x = Activation('relu')(x)
            x = BatchNormalization()(x)

            x = Conv2D(8*(2 ** i), (3, 3),
                       data_format="channels_first",
                       padding="same")(x)
            x = Activation('relu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2), data_format="channels_first")(x)
            print(x.shape)

        x = Flatten()(x)
        x = Dropout(0.2)(x)        
        x = Dense(512, activation='relu')(x)
        x = Dense(len(config.POSSIBLE_LABELS), activation='softmax')(x)
        model = Model(inputs=x_in, outputs=x)
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        self.model = model
