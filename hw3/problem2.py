import cv2         
import numpy as np

import matplotlib   
matplotlib.use('Agg')
from matplotlib import pyplot as plt 


# imports
import os
import random
import sys
import math

def GetCorners(image):
    #find the features for the frame image
    #read the frame as grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,30,255,0)

    #grab the harris corner points
    corners = cv2.cornerHarris(thresh,4,1,0.04)
 
    return corners>0.001

def DiffFrames(frame1, frame2):
    return np.square((frame2 - frame1)**2).sum()


# Cuts each frame of the video along the seam in the mask
# Composites the source on the left and the target on the right and creates the video file
def CreateComposite(source, outputName):
    compositeVideo = cv2.VideoWriter(outputName, cv2.VideoWriter_fourcc('D','I','V','X'), source.get(5), (int(source.get(3)), int(source.get(4))), True)  
	
    prevFrame = None

	# Loop through frames of source and target and
	# composite them together until no more frames in one
    while(True):
        retSource, frameSource = source.read()
 
        #grab the corners of this image
        pixelTest = GetCorners(frameSource)

        #set if none
        if prevFrame is None:
            prevFrame = pixelTest

        #find the related points between each frame
        closestMatches = DiffFrames(pixelTest, prevFrame)

        #calculate the homography

        cv2.imshow('Harris Corner Detector',pixelTest.astype(np.uint8)*255)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

        if (not retSource):
            break

		# Add this frame to the composite video
        compositeVideo.write(frameSource.astype(np.uint8))

	# Close the video writer
    compositeVideo.release()

if __name__ == "__main__":

    if (len(sys.argv) != 4) :
        print('Invalid number of arguments')
        sys.exit()

    sourceName = sys.argv[1]
    outputName = sys.argv[2]

    # Open the videos
    source = cv2.VideoCapture(sourceName)

	# Checks that the videos opened properly
    if (not source.isOpened()) :
        print('Videos did not open correctly')
        sys.exit()

    CreateComposite(source, outputName)


