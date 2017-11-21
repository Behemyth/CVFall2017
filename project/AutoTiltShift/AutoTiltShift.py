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
    return np.abs(a - b)

def tiltShift(frame, center):

    blurIterations = 5
    focus = frame.shape[0]/10

    yTopOriginal = center-focus
    yBotOriginal = center+focus
    #TODO if the top or bottom goes past the image edge, clamp it

    yTop = yTopOriginal
    yBot = yBotOriginal  

    distTop = yTop
    distBot = frame.shape[0]-yBot

    blurred = frame

    for i in range(blurIterations):
        ksize = (i*2)+1
        blurred = cv2.GaussianBlur(frame,(ksize,ksize),0)

        shapeImage = (frame.shape[0],frame.shape[1],1)
        shape = (frame.shape[0],frame.shape[1])

        mask = np.zeros(shape)

        row,col = np.indices(shape)

        mask[(row<yTop) | (row>yBot)] = 1

        frame[mask == 1] = blurred[mask == 1]

        yTop = yTopOriginal - distTop*(i/float(blurIterations))
        yBot = yBotOriginal + distBot*(i/float(blurIterations))


    return frame

def findBestVerticle(videoName):
    vidSource = cv2.VideoCapture(videoName)

    ok, prevFrame = vidSource.read()

    return prevFrame.shape[0]/2

    #TODO programatically set center
    # 
    
    #hog = cv2.HOGDescriptor()
    #hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    #avgX = 0
    #avgY = 0

    #boxCount = 0

    #ok, prevFrame = vidSource.read()

    ##second loop for write processing
    #while True:
    #    ret, frame = vidSource.read()
    #    if not ret:
    #        break

    #    # detect people in the image
    #    rects, weights = hog.detectMultiScale(frame, winStride=(4, 4),
    #    padding=(8, 8), scale=1.05)

    #    #average the boxes
    #    for (x, y, w, h) in rects:
    #        avgX += x + w / 2.0
    #        avgY += y + h / 2.0
    #        boxCount =  boxCount + 1

    #return avgX/boxCount, avgY/boxCount

if __name__ == "__main__":

    videoName = sys.argv[1]
 
    avgY = findBestVerticle(videoName)

    #writer info
    cap = cv2.VideoCapture(0)
    vidSource = cv2.VideoCapture(videoName)

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter("output.avi", 
							cv2.VideoWriter_fourcc('m','p','4','v'), 
							vidSource.get(5),  # setting the frame rate of composite to be same as vidSource
							(int(vidSource.get(3)), int(vidSource.get(4))), True)  # setting size of composite to be same as vidSource

    ok, prevFrame = vidSource.read()

    #second loop for write processing
    while True:
        ret, frame = vidSource.read()
        if not ret:
            break

        #color changes
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

        saturationAdd = 35
        valueAdd = 20

        maskS = 255 - hsv[:,:,1]
        maskV = 255 - hsv[:,:,2]

        hsv[:,:,1] = np.where(maskS < saturationAdd,255,hsv[:,:,1] + saturationAdd)
        hsv[:,:,2] = np.where(maskV < valueAdd,255,hsv[:,:,2] + valueAdd)

        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) 

        frame = tiltShift(frame,avgY)

        out.write(frame)

        prevFrame = frame

    # Release everything if job is finished
    cap.release()
    out.release()