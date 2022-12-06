import pickle
import random
import sys
import time

from os import system

import tkinter
import tkinter.filedialog
import pygame as pg
import pygame.locals

import numpy as np
import cv2

from rpscv import utils
from rpscv import imgproc as imp
from rpscv.gui import RPSGUI

def promptFile():
    # Create a Tk file dialog and cleanup when finished
    top = tkinter.Tk()
    top.withdraw() # Hide window

    fileName = tkinter.filedialog.askopenfilename(parent = top)
    top.destroy()

    return fileName

def saveImage(img, gesture, notify=False):
    # Define image path and filename
    folder = utils.imagePathsRaw[gesture]
    name = utils.gestureTexts[gesture] + '-' + time.strftime('%Y%m%d-%H%M%S')
    extension = '.png'

    if notify:
        print('Saving {}'.format(folder + name + extension))

    # Save image
    cv2.imwrite(folder + name + extension, img)

if __name__ == '__main__':
    system('cls')

    try:
        # Initialize game mode variables
        loop = False
        maxScore = 5
        timeBetweenRounds = 3000 # ms

        # Read command line arguments
        if len(sys.argv) > 1:
            for arg in sys.argv[1:]:
                if arg == 'loop':
                    loop = True
                else:
                    print('{} is not a recognized argument'.format(arg))

        # Load classifier from pickle file
        classifierFilename = 'classifier.pkl'

        with open(classifierFilename, 'rb') as f:
            classifier = pickle.load(f)

        # Initialize GUI
        gui = RPSGUI(loop = loop)

        # Load static images for computer gestures
        computerImages = {}

        inputImage = cv2.imread('img/gui/rock.png', cv2.IMREAD_COLOR)
        computerImages[utils.ROCK] = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
        
        inputImage = cv2.imread('img/gui/paper.png', cv2.IMREAD_COLOR)
        computerImages[utils.PAPER] = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)

        inputImage = cv2.imread('img/gui/scissors.png', cv2.IMREAD_COLOR)
        computerImages[utils.SCISSORS] = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)

        # Load empty image
        emptyImage = cv2.imread('img/gui/empty.png', cv2.IMREAD_COLOR)
        emptyImage = cv2.cvtColor(emptyImage, cv2.COLOR_BGR2RGB)

        while True:
            # Draw GUI
            gui.draw()

            # Flip pygame display
            pg.display.flip()

            # Get image from camera
            inputImagePath = promptFile()
            inputImage = cv2.imread(inputImagePath)

            inputImageRGB = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
            inputImageGrayscale = imp.getGray(inputImageRGB, threshold = 17)

            amountNonZeroPixels = np.count_nonzero(inputImageGrayscale)

            # Check if player hand is present
            if amountNonZeroPixels > 9000:
                # Player predicted gesture
                playerGesture = classifier.predict([inputImageGrayscale])[0]
                computerGesture = random.randint(0,2)

                gestureDifference = computerGesture - playerGesture

                # Set computer image to computer gesture
                gui.setPlayerMove(inputImageRGB, playerGesture)
                gui.setComputerMove(computerImages[computerGesture], computerGesture)

                if gestureDifference in [-2, 1]:
                    gui.setWinner('computer')
                elif gestureDifference in [-1, 2]:
                    gui.setWinner('player')
                else:
                    gui.setWinner(None)
            else:
                # Set computer image to green
                gui.setComputerMove(emptyImage)
                gui.setWinner(None)

            # Draw GUI
            gui.draw()

            # Flip pygame display
            pg.display.flip()

            # Wait
            pg.time.wait(timeBetweenRounds)

            # Check pygame events
            for event in pg.event.get():
                if event.type == pg.locals.QUIT:
                    gui.quit()

            # Check if scores reach endScore (end of game)
            if gui.playerScore == maxScore or gui.computerScore == maxScore:
                gui.gameOver()

    finally:
        f.close()