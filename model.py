from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, BatchNormalization
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import GlobalMaxPool2D
from tensorflow.python.keras.layers import concatenate, Dense, Dropout
import config


class STFTCNN():

    def __init__(self,
                 input_shape=(257, 98, 2),
                 name="STFTCNN"):

        self.name = name
        
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
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model
