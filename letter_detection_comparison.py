import numpy as numpy
from matplotlib import pyplot as plt

import cv2
import letters

for i in range(21):
    img = cv2.imread(
        "D:\\Users\\Otto Glass\\Documents\\CaptchaReader\\Examplo Captcha\\"+str(i+1)+".png", 0)
    letters_bounds = letters.find(img)
    if len(letters_bounds) == 4:
        for letter in letters_bounds:
            (x, y, w, h) = letter
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
    plt.subplot(3, 7, i+1)
    plt.imshow(img, 'gray')
    plt.xticks([]), plt.yticks([])
plt.show()
