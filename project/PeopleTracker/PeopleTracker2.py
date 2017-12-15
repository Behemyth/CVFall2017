# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="input video file")
args = vars(ap.parse_args())
 
# initialize the HOG descriptor/person detector
print(cv2.HOGDescriptor_getDefaultPeopleDetector())
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

iv = cv2.VideoCapture(args["input"])

while (iv.isOpened()):
    ret, frame = iv.read()
    if (ret == True):
        frame = imutils.resize(frame, width=800)

        (rects, weights) = hog.detectMultiScale(frame, winStride=(4,4), padding=(16,16), scale=1.2)
        
        rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        for (xA, yA, xB, yB) in rects:
            cv2.rectangle(frame, (xA,yA), (xB,yB), (0,0,255), 2)

        cv2.imshow("frame", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    else:
        break
    # cv2.waitKey(0)

iv.release()
cv2.destroyAllWindows()