from keras.callbacks import TensorBoard
from keras.layers import Activation, Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications.xception import Xception

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
        base_model = Xception(input_shape=self.input_shape,
                              weights='imagenet',
                              include_top=False)

        # Top Model Block
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(1, activation='tanh')(x)

        # add your top layer block to your base model
        model = Model(base_model.input, predictions)

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
        return self.model.predict(np.expand_dims(state, axis=0))

    def save_model(self, log_dir, model_name):
        model_json = self.model.to_json()
        with open(log_dir + "/model.json", "w") as json_file:
            json_file.write(model_json)

        self.model.save(log_dir + '/' + model_name + '.h5')