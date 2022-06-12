import os

import numpy as np

import cv2
import letters as lt
from kerasNN import Network

MODEL_PATH = "Models"
MODEL_NAME = "Model-Arch1-4"
model_location = os.path.join(MODEL_PATH, MODEL_NAME)


def load_data(data_directory):
    files = os.listdir(data_directory)
    labels = []
    captchas = []
    for f in files:
        captcha = cv2.imread(os.path.join(data_directory, f), 0)
        labels.append(f[0:4])
        captchas.append(captcha)
    return captchas, labels


DATA_PATH = "D:\\AI_DATA\\captcha2\\eval"

eval_data, eval_labels = load_data(DATA_PATH)
NN = Network()
NN.load(model_location)


ACC = 0
for c_idx, captcha in enumerate(eval_data):
    prediction = NN.predictCaptcha(captcha)
    print("prediction: "+prediction+" label: "+eval_labels[c_idx], end="\r")
    if eval_labels[c_idx] == prediction:
        ACC += 1

ACC = ACC/len(eval_data)

print("Accuracy: "+str(ACC))
