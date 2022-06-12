import os

import cv2
from Pages import CNPqLattes

MODEL_PATH = "Models"
DATA_FOLDER = "DATA"
NUM_SAMPLES = 200
MODEL = "Model-Arch1-4"
model_location = os.path.join(MODEL_PATH, MODEL)

success_folder = os.path.join(DATA_FOLDER, "success")
failure_folder = os.path.join(DATA_FOLDER, "failure")

if not os.path.isdir(success_folder):
    os.makedirs(success_folder)
if not os.path.isdir(failure_folder):
    os.makedirs(failure_folder)


hits = 0
lattes = CNPqLattes()
lattes.NN.load(model_location)
for i in range(NUM_SAMPLES):
    captcha = lattes.requestCaptcha()
    prediction = lattes.NN.predictCaptcha(captcha)
    ACC = hits/(i+1)
    print("num: "+str(i+1)+" prediction: " +
          prediction, "Accuracy: "+str(ACC), end="\r")
    if lattes.validCaptcha("K4793551T3", prediction):
        cv2.imwrite(success_folder+"\\"+prediction+"-"+str(i) +
                    ".png", cv2.cvtColor(captcha, cv2.COLOR_GRAY2BGR))
        hits += 1
    else:
        cv2.imwrite(failure_folder+"\\"+prediction+"-"+str(i) +
                    ".png", cv2.cvtColor(captcha, cv2.COLOR_GRAY2BGR))
    print("num: "+str(i+1)+" prediction: " +
          prediction, "Accuracy: "+str(ACC), end="\r")
