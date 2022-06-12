import os

import cv2
import letters as lt

DATA_PATH = "D:\\AI_DATA\\captcha2\\CAPTCHAS"
FOLDER_PATH = os.path.join(DATA_PATH, "training")
files = os.listdir(DATA_PATH)
os.mkdir(FOLDER_PATH)
for f_idx, f in enumerate(files):
    captcha = cv2.imread(os.path.join(DATA_PATH, f), 0)
    letters = lt.find(captcha)
    if len(letters) != 4:
        continue
    for i in range(4):
        (x, y, w, h) = letters[i]
        image = captcha[y:(y+h), x:(x+w)]
        if not(cv2.imwrite(os.path.join(FOLDER_PATH, f[i], str(f_idx))+".png", lt.resize(image))):
            os.mkdir(os.path.join(FOLDER_PATH, f[i]))
            cv2.imwrite(os.path.join(FOLDER_PATH, f[i], str(
                f_idx))+".png", lt.resize(image))
