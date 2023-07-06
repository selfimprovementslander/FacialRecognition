import cv2
import numpy as np

classifiers = [cv2.CascadeClassifier("Classifiers/haarcascade_frontalface_alt2.xml"),
               cv2.CascadeClassifier("Classifiers/haarcascade_profileface.xml")]


def find_faces(img):
    # Gray-scaling the image improves performance
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    output1 = classifiers[0].detectMultiScale(
        grayscale,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
    )

    output2 = classifiers[1].detectMultiScale(
        grayscale,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(30, 30),
    )

    return output1, output2


