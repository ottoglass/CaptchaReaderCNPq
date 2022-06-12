import json
import os

from keras import optimizers
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          InputLayer, MaxPooling2D)
from keras.models import Sequential, load_model
from keras.utils import to_categorical

import letters as lt


class Network:
    LabelsDict = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
                  10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I", 19: "J", 20: "K",
                  21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T", 30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z", }

    def __init__(self):
        self.model = Sequential([
            Conv2D(40, (5, 5), strides=2, padding="same",
                   input_shape=(46, 46, 1)),
            Activation('relu'),
            Conv2D(40, (3, 3), strides=2, padding="same",
                   input_shape=(46, 46, 1)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(80, (3, 3), strides=2, padding="same"),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(80, (2, 2), strides=2, padding="same"),
            Activation('relu'),
            Flatten(),
            Dense(2000),
            Activation('relu'),
            Dropout(0.5),
            Dense(1000),
            Activation('relu'),
            Dense(len(Network.LabelsDict)),
            Activation('softmax'),
        ])

    def train(self, data, labels):
        learning_rate = 0.001
        decay_rate = learning_rate/100
        sgd = optimizers.SGD(
            lr=learning_rate, decay=decay_rate, nesterov=True, momentum=0.8)
        self.model.compile(optimizer=sgd,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        one_hot_labels = to_categorical(
            labels, num_classes=len(Network.LabelsDict))
        self.model.fit(data, one_hot_labels, epochs=120, batch_size=200)

    def save(self, name):
        self.model.save(name+".h5")
        with open(name+"_labels.json", "w") as json_file:
            json_labels = json.dumps(Network.LabelsDict)
            json_file.write(json_labels)
        self.saveWeights(name)

    def loadWeights(self, name):
        self.model.load_weights(name+"_weights.h5")

    def saveWeights(self, name):
        self.model.save_weights(name+"_weights.h5")

    def loadDictionary(self, dictionary):
        Network.LabelsDict = dictionary

    def load(self, name):
        self.model = load_model(name+".h5")
        with open(name+"_labels.json") as json_file:
            labels = json.loads(json_file.read())
            Network.LabelsDict = labels

    def predictCaptcha(self, captcha):
        letters = lt.find(captcha)
        result = ""
        for letter in letters[0:4]:
            (x, y, w, h) = letter
            image = lt.resize(captcha[y:(y+h), x:(x+w)])
            image = image.reshape(1, 46, 46, 1)
            prediction = self.model.predict_classes(image)
            result += Network.LabelsDict[str(prediction[0])]
        return result
