import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_image(srcpath):
    img=cv2.imdecode(np.fromfile(srcpath, dtype=np.uint8), -1)

    if hasattr(img, 'ndim')==False:
        print("Failed to read " + srcpath)

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = img / 255
    # img = cv2.imread(srcpath, 0)
    return img

def show_image(img):
    plt.figure()
    plt.imshow(img, cmap='gray')