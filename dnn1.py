from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from time import strftime
from time import time
from keras.callbacks import TensorBoard
from keras.layers import *
from keras.models import Model


class model_1():

    def __init__(self, batch_size, epochs, optimizer="Adam"):
        self.train_datagen = ImageDataGenerator()
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = self.build_model()
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

    def build_model(self):
        base_model = Xception(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

        # Top Model Block
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(176, activation='softmax')(x)

        # add your top layer block to your base model
        model = Model(base_model.input, predictions)

        return model

        
    def learn(self, x_train, y_train):
        train_size = x_train.shape[0]

        train_generator = self.train_datagen.flow(x_train, y_train, batch_size=self.batch_size)

        # Train the model
        self.model.fit_generator(train_generator,
                                 steps_per_epoch=int(np.ceil(train_size / float(self.batch_size))),
                                 workers=4,
                                 epochs=self.epochs,
                                 verbose=1)

    def predict(self, state):
        return self.model.predict(state)

    def save_model(self, log_dir):
        # Saving the model
        model_json = self.model.to_json()
        with open(log_dir + "/model.json", "w") as json_file:
            json_file.write(model_json)

        self.model.save(log_dir + '/my_model.h5')


