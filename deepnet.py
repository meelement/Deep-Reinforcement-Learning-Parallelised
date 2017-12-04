from keras.callbacks import TensorBoard
from keras.layers import Activation, Dense, Flatten, Cropping2D, Conv2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Lambda
from keras.backend import tf as ktf

from time import time
from time import strftime
import numpy as np


class DQN():

    def __init__(self, batch_size, input_shape, optimizer = "Adam"):
        self.train_datagen = ImageDataGenerator()
        self.batch_size = batch_size
        self.input_shape = input_shape

        # Visualize training
        self.tensorboard = self.tensorboard_object()

        self.model = self.build_model()
        # Compile the model
        self.model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['mae'])

    @staticmethod
    def tensorboard_object():
        now = strftime("%c")
        log_dir = "logs/" + now.format(time())
        return TensorBoard(log_dir=log_dir)

    def build_model(self):
        # Model
        model = Sequential()

        # Convolutional
        model.add(Cropping2D(cropping=((45, 5), (0, 0)),
                             input_shape=self.input_shape))
        Lambda(lambda image: ktf.image.resize_images(image, (80, 200)))
        model.add(Lambda(lambda x: (x / 255.0) - 0.5))

        model.add(Conv2D(filters=6,
                         kernel_size=(5, 5), strides=(2, 2),
                         padding='valid',
                         dilation_rate=(1, 1),
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='zeros'))
        model.add(Activation('relu'))

        model.add(Conv2D(filters=16,
                         kernel_size=(5, 5), strides=(2, 2),
                         padding='valid',
                         dilation_rate=(1, 1),
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='zeros'))
        model.add(Activation('relu'))

        model.add(Conv2D(filters=32,
                         kernel_size=(5, 5), strides=(2, 2),
                         padding='valid',
                         dilation_rate=(1, 1),
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='zeros'))
        model.add(Activation('relu'))

        model.add(Conv2D(filters=32,
                         kernel_size=(3, 3), strides=(1, 1),
                         padding='valid',
                         dilation_rate=(1, 1),
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='zeros'))
        model.add(Activation('relu'))

        model.add(Conv2D(filters=32,
                         kernel_size=(3, 3), strides=(1, 1),
                         padding='valid',
                         dilation_rate=(1, 1),
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='zeros'))
        model.add(Activation('relu'))

        # Fully connected
        model.add(Flatten())
        # model.add(Dropout(0.35))
        model.add(Dense(units=1164))
        model.add(Activation('relu'))
        model.add(Dense(units=100))
        model.add(Activation('relu'))
        model.add(Dense(units=50))
        model.add(Activation('relu'))
        model.add(Dense(units=10))
        model.add(Activation('relu'))
        model.add(Dense(units=1))
        return model

    def learn(self, x_train, y_train) :

        train_size = x_train.shape[0]

        train_generator = self.train_datagen.flow(x_train,
                                                  y_train,
                                                  batch_size=self.batch_size)
        
        # Train the model
        self.model.fit_generator(train_generator,
                            steps_per_epoch=int(np.ceil(train_size / float(self.batch_size))),
                            epochs=self.epochs,
                            workers=4,
                            callbacks=[self.tensorboard],
                            verbose=1)

    def predict(self, state, action):
        self.model.predict(state)

    def save_model(self, log_dir, model_name):
        # Saving the model
        model_json = self.model.to_json()
        with open(log_dir + "/model.json", "w") as json_file:
            json_file.write(model_json)

        self.model.save(log_dir + '/' + model_name + '.h5')