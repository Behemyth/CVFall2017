import cv2         
import numpy as np
from matplotlib import pyplot as plt 
import matplotlib   

# imports
import os
import random
import sys

HeadBias = 50
changeFactor = 4

#return
def Difference(a,b):
    return np.abs(a - b)

def tiltShift(frame, center,heights):

    blurIterations = 7
    focus = frame.shape[0] / 8

    yTopOriginal = int(center - focus)
    yBotOriginal = int(center + focus)
    #TODO if the top or bottom goes past the image edge, clamp it

    yTop = yTopOriginal
    yBot = yBotOriginal  

    distTop = yTop
    distBot = frame.shape[0] - yBot

    blurred = frame

    for i in range(blurIterations):
        ksize = (i * 2) + 3
        blurred = cv2.GaussianBlur(frame,(ksize,ksize),0)

        shapeImage = (frame.shape[0],frame.shape[1],1)
        shape = (frame.shape[0],frame.shape[1])

        mask = np.zeros(shape)

        row,col = np.indices(shape)

        mask[(row < yTop) | (row > yBot)] = 1

        frame[mask == 1] = blurred[mask == 1]

        val = int((i / float(blurIterations))*255)
        heights[mask == 1] = [val,val,val]

        yTop = yTopOriginal - distTop * (i / float(blurIterations))
        yBot = yBotOriginal + distBot * (i / float(blurIterations))

    heights[yTopOriginal,:] = [0,255,0]
    heights[yBotOriginal,:] = [0,255,0]
    heights[center,:] = [0,0,255]

    return frame,heights

def findBestVerticle(frame,fgbg,kernel, prevHeight):

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    y,x = np.indices(fgmask.shape)

    mask = y[fgmask >= 1]

    height = np.average(mask)
    
    if not height >= 0:
        height = frame.shape[0] / 2

    height = height - HeadBias

    change = height - prevHeight 
    if(np.abs(change) > changeFactor):
        height = prevHeight + np.sign(change)*changeFactor
   
    return int(height), fgmask


if __name__ == "__main__":

    videoName = sys.argv[1]
 
    #writer info
    cap = cv2.VideoCapture(0)
    vidSource = cv2.VideoCapture(videoName)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

    fgbg = cv2.createBackgroundSubtractorMOG2()

    pre = videoName.split('.')[0]
    outName = pre + "_" + "Post.avi"
    outName2 = pre + "_" + "PostColor.avi"
    outName3 = pre + "_" + "Movement.avi"
    outName4 = pre + "_" + "Center.avi"

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(outName, 
							cv2.VideoWriter_fourcc('m','p','4','v'), 
							vidSource.get(5),  # setting the frame rate of composite to be same as vidSource
							(int(vidSource.get(3)), int(vidSource.get(4))), True)  # setting size of composite to be same as vidSource

    # Define the codec and create VideoWriter object
    outPost = cv2.VideoWriter(outName2, 
							cv2.VideoWriter_fourcc('m','p','4','v'), 
							vidSource.get(5),  # setting the frame rate of composite to be same as vidSource
							(int(vidSource.get(3)), int(vidSource.get(4))), True)  # setting size of composite to be same as vidSource

    # Define the codec and create VideoWriter object
    outMovement = cv2.VideoWriter(outName3, 
							cv2.VideoWriter_fourcc('m','p','4','v'), 
							vidSource.get(5),  # setting the frame rate of composite to be same as vidSource
							(int(vidSource.get(3)), int(vidSource.get(4))), True)  # setting size of composite to be same as vidSource

    # Define the codec and create VideoWriter object
    outHeight = cv2.VideoWriter(outName4, 
							cv2.VideoWriter_fourcc('m','p','4','v'), 
							vidSource.get(5),  # setting the frame rate of composite to be same as vidSource
							(int(vidSource.get(3)), int(vidSource.get(4))), True)  # setting size of composite to be same as vidSource

    ok, prevFrame = vidSource.read()
    prevHeight = (prevFrame.shape[0] / 2) - HeadBias

    frameNumber = 1
    #second loop for write processing
    while True:
        ret, frame = vidSource.read()
        if not ret:
            break


        avgY, mask = findBestVerticle(frame,fgbg,kernel, prevHeight)
        prevHeight = avgY

        #reshape the mask so it can be printed out
        mask = mask[:,:,np.newaxis]
        mask = np.repeat(mask,3,axis=2)

        heights = np.zeros_like(mask)

        frame, heights = tiltShift(frame,avgY,heights)

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
        outPost.write(moddedFrame)
        out.write(frame)
        outMovement.write(mask)

        outHeight.write(heights)

        #frame logic
        prevFrame = frame

        print(frameNumber)

        frameNumber = frameNumber + 1

    # Release everything if job is finished
    cap.release()
    out.release()
    outPost.release()
    outMovement.release()
    outHeight.release()