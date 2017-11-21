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
        ret, frame = vidSource.read()
        if not ret:
            break

        if firstFrame:

            prevFrame = frame
            accumulation = np.zeros_like(frame)

            firstFrame= False

            #writer info
            cap = cv2.VideoCapture(0)

            # Define the codec and create VideoWriter object
            out = cv2.VideoWriter("output.avi", 
									cv2.VideoWriter_fourcc('m','p','4','v'), 
									vidSource.get(5),  # setting the frame rate of composite to be same as vidSource
									(int(vidSource.get(3)), int(vidSource.get(4))), True)  # setting size of composite to be same as vidSource

        else:

            #get motion of two frames
            accumulation += Difference(prevFrame,frame)

            # write the flipped frame
            out.write(frame.astype('uint8'))

    # Release everything if job is finished
    cap.release()
    out.release()