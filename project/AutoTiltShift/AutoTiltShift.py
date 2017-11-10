import cv2         
import numpy as np
from matplotlib import pyplot as plt 
import matplotlib   

# imports
import os
import random
import sys

#return
def Difference(a,b):
    return np.abs(a-b)

if __name__ == "__main__":

    videoName = sys.argv[1]

    vidSource = cv2.VideoCapture(videoName)

    firstFrame = True
    
    while True:
        ret, frameSource = vidSource.read()
        if not ret:
            break

        if firstFrame:

            prevFrame = frameSource
            accumulation = np.zeros_like(frameSource)

            firstFrame= False

        else:

            #get motion of two frames
            accumulation += Difference(prevFrame,frameSource)

