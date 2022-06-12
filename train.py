import os

import numpy as np
from matplotlib import pyplot as plt

import cv2
from kerasNN import Network

MODEL_PATH = "Models"
MODEL_NAME = "Model-Arch1-3"
model_location = os.path.join(MODEL_PATH, MODEL_NAME)


def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    dictionary = {}
    for label, d in enumerate(directories):
        dictionary[label] = d
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".png")]
        for f in file_names:
            images.append(cv2.imread(f, 0))
            labels.append(label)

    return images, labels, dictionary


ROOT_PATH = "D:\\AI_DATA\\captcha2"

train_data, train_labels, dictionary = load_data(
    os.path.join(ROOT_PATH, "training"))
train_data = np.array(train_data, dtype=np.float32)
train_data = train_data.reshape(train_data.shape[0], 46, 46, 1)
train_labels = np.array(train_labels, dtype=np.int32)
Network.LabelsDict = dictionary
NN = Network()
# NN.load(model_location)
NN.train(train_data, train_labels)
NN.save(model_location)
