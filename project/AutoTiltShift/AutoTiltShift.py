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

    blurIterations = 7
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

def findBestVerticle(frame,fgbg,kernel):

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    y,x = np.indices(fgmask.shape)

    cv2.imwrite("BackgroundMask.png", fgmask)

    mask = y[fgmask>=1]

    height = np.average(mask)
    
    if not height >= 0:
        height = frame.shape[0]/2

    return height


if __name__ == "__main__":

    videoName = sys.argv[1]
 
    #writer info
    cap = cv2.VideoCapture(0)
    vidSource = cv2.VideoCapture(videoName)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

    pre = videoName.split('.')[0]
    outName = pre + "_" + "Post.avi"
    outName2 = pre + "_" + "PostColor.avi"

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(outName, 
							cv2.VideoWriter_fourcc('m','p','4','v'), 
							vidSource.get(5),  # setting the frame rate of composite to be same as vidSource
							(int(vidSource.get(3)), int(vidSource.get(4))), True)  # setting size of composite to be same as vidSource

    # Define the codec and create VideoWriter object
    outBare = cv2.VideoWriter(outName2, 
							cv2.VideoWriter_fourcc('m','p','4','v'), 
							vidSource.get(5),  # setting the frame rate of composite to be same as vidSource
							(int(vidSource.get(3)), int(vidSource.get(4))), True)  # setting size of composite to be same as vidSource

    ok, prevFrame = vidSource.read()

    frameNumber = 1
    #second loop for write processing
    while True:
        ret, frame = vidSource.read()
        if not ret:
            break


        avgY = findBestVerticle(frame,fgbg,kernel)

        frame = tiltShift(frame,avgY)



         #color changes
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

        saturationAdd = 35
        valueAdd = 20

        maskS = 255 - hsv[:,:,1]
        maskV = 255 - hsv[:,:,2]

        hsv[:,:,1] = np.where(maskS < saturationAdd,255,hsv[:,:,1] + saturationAdd)
        hsv[:,:,2] = np.where(maskV < valueAdd,255,hsv[:,:,2] + valueAdd)

        moddedFrame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) 


        #writes
        outBare.write(moddedFrame)
        out.write(frame)


        #frame logic
        prevFrame = frame

        print(frameNumber)

        frameNumber = frameNumber+1

    # Release everything if job is finished
    cap.release()
    out.release()