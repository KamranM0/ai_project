import cv2
import numpy as np
import random

def augment_image(image):
    # Random flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # Random rotation
    angle = random.randint(-15, 15)
    M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return image
