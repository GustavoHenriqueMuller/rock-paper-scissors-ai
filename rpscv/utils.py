import time
import numpy as np

# Define possible gestures as constants
ROCK = 0
PAPER = 1
SCISSORS = 2

# Define text labels corresponding to gestures
gestureTxt = {ROCK: 'rock', PAPER: 'paper', SCISSORS: 'scissors'}

# Define paths to raw image folders
imagePathsRaw = {ROCK: './img/rock/', PAPER: './img/paper/', SCISSORS: './img/scissors/'}

def getGestureNameByIndex(gestureIndex):
    if gestureIndex == ROCK:
        return "rock"
    elif gestureIndex == PAPER:
        return "paper"
    elif gestureIndex == SCISSORS:
        return "scissors"

class Filter1D:
    """A one dimensional filter class. Useful for real-time filtering of noisy
    time series data such as sensor signal, etc."""

    def __init__(self, maxSize=3):
        """Class constructor. maxSize argument defines the size (number of data
        points) of the signal to be kept. maxSize must be an odd integer >= 3.
        """
        if maxSize % 2 == 1 and maxSize >= 3:
            self._maxSize = maxSize
        else:
            raise ValueError("maxSize must be an odd integer >= 3")
        self._data = np.ndarray(0)

    def addDataPoint(self, dataPoint):
        """Adds new data point(s) to the data array. If the data array size
        exceeds the maxSize attribute, the older data points will be trimmed
        from the array (left trim). dataPoint can be a single point, a list or
        a numpy one dimensional array."""

        # Append new data point(s) to end of array
        self._data = np.insert(self._data, self._data.size, dataPoint)

        # Trim begining of array if longer than maxSize
        if self._data.size > self._maxSize:
            self._data = self._data[self._data.size - self._maxSize:]

    def getData(self):
        """Returns the complete data array."""
        return self._data

    def getLast(self):
        """Returns the last (most recent) data point from the data array."""
        return self._data[-1]

    def getMean(self, windowSize=0):
        """Returns the mean of the last n points from the data array where n
        equals windowSize. If windowSize is not specified, is set to 0 or is
        greater than maxSize, windowSize will be automatically set to maxSize
        and the mean of the entire data array will be returned."""
        if self._data.size == 0:
            raise RuntimeError("Filter1D data is empty. Call Filter1D.addDataPoint() to add data prior calling Filter1D.getMean().")
        if type(windowSize) is int:
            if windowSize <= 0 or windowSize > self._maxSize:
                windowSize = self._maxSize
            return np.mean(self._data[-windowSize:])
        else:
            raise TypeError("windowSize must be an integer")

    def getMedian(self, windowSize=0):
        """Returns the median of the last n points from the data array where n
        equals windowSize. windowSize must be an odd integer. If windowSize
        is not specified or is set to 0, windowSize will be automatically set
        to maxSize and the median of the entire data array will be returned."""
        if self._data.size == 0:
            raise RuntimeError("Filter1D data is empty. Call Filter1D.addDataPoint() to add data prior calling Filter1D.getMedian().")
        if type(windowSize) is not int:
            raise TypeError("windowSize must be an integer")
        if windowSize <= 0 or windowSize > self._maxSize:
            windowSize = self._maxSize
        if windowSize % 2 == 1 and windowSize <= self._maxSize:
            return np.median(self._data[-windowSize:])
        else:
            raise ValueError("windowSize must be an odd integer <= maxSize")