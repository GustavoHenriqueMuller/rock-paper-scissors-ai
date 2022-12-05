import os
from glob import glob
import time

import numpy as np

from skimage.io import imread
from skimage import color
from skimage import feature
from skimage import filters

from rpscv import utils

import cv2

def fastRotate(img):
    """Rotates the image clockwise 90 deg."""
    return np.transpose(img, axes = (1, 0, 2))[:, ::-1, :].copy()

def generateGrayFeatures(imageShape = (200, 300, 3), verbose = False, randomSeed = 42):
    imageSize = imageShape[0] * imageShape[1]
    gestures = [utils.ROCK, utils.PAPER, utils.SCISSORS]

    # Create a list of image files for each gesture
    files = []

    for i, gesture in enumerate(gestures):
        path = os.path.join(utils.imagePathsRaw[gesture], '*.png')
        files.append(glob(path))
        files[i].sort(key = str.lower)

    totalImagesAmount = sum([len(i) for i in files])

    # Create empty numpy arays for features and labels
    features = np.empty((totalImagesAmount, imageSize), dtype = np.float32)
    labels = np.empty(totalImagesAmount, dtype = np.int)

    # Generate grayscale images
    counter = 0

    for i, gesture in enumerate(gestures):
        for imagePath in files[i]:
            if verbose:
                print('Processing image {}'.format(imagePath))

            # Load image as a numpy array
            image = imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if image.shape == imageShape:
                # Generate and store image features in features array
                features[counter] = getGray(image, threshold = 17)

                # Store image label in labels array
                labels[counter] = gesture

                counter += 1
            else:
                print('Image {} has invalid shape: {}, {} expected, skipping image.'.format( \
                    imagePath, image.shape, imageShape))

    print('Completed processing {} images'.format(counter))

    return features[:counter], labels[:counter]


def getGray(image, hueValue = 63, threshold = 0):
    image = removeBackground(image, hueValue, threshold)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255

    return image.ravel()


def hueDistance(image, hueValue):
    # Convert image to HSV colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hChannel = hsv[:, :, 0].astype(int)

    dist = np.minimum(np.abs(hChannel - hueValue),
                      360 - np.abs(hChannel - hueValue))

    return dist


def removeBackground(image, hueValue, threshold = 0):
    # Get an image corresponding to the hue distance from the background hue value
    dist = hueDistance(image, hueValue)

    # Create a copy of the source image to use as masked image
    masked = image.copy()

    # Select background pixels using thresholding and set value to zero (black)
    if threshold == 0:
        masked[dist < filters.threshold_mean(dist)] = 0
    else:
        masked[dist < threshold] = 0

    return masked
