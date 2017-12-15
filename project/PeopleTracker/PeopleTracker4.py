# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import AEKeyframeGenerator as ae

trackPt = []
ptSelected = False

def trackVideo(iv, old_gray, trackPt, lk_params):
    oldGray = old_gray.copy()
    mask = np.zeros_like(oldFrame)
    cv2.namedWindow('Tracked Frame')
    while (1):
        ret, frame = iv.read()

        if (not ret):
            break

        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # print(np.array([trackPt[-1]], dtype=np.float32))
        # calculate optical flow
        newPos, st, err = cv2.calcOpticalFlowPyrLK(oldGray, frameGray, np.array([trackPt[-1]], dtype=np.float32), None, **lk_params)

        # if we couldn't find the tracking point,
        # just use the previous one
        if st[0] == 0:
            newPos[0] = trackPt[-1]

        x0 = trackPt[-1][0]
        y0 = trackPt[-1][1]
        x1 = newPos[0,0]
        y1 = newPos[0,1]
        mask = cv2.line(mask, (x0,y0), (x1,y1), (0,0,255), thickness=2, lineType=cv2.LINE_AA)

        trackPt.append(newPos[0])
        oldGray = frameGray

        img = cv2.add(frame, mask)
        img = imutils.resize(img,width=1920//2)

        cv2.imshow('Tracked Frame', img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    return trackPt

def getPointOnImage(event, x, y, flags, param):
    global trackPt, ptSelected

    if event == cv2.EVENT_LBUTTONDOWN:
        trackPt = [(x,y)]

    elif event == cv2.EVENT_LBUTTONUP:
        trackPt = [(x,y)]
        ptSelected = True

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input video file")
    ap.add_argument("-o", "--output", required=True, help="output keyframe text file")
    args = vars(ap.parse_args())

    iv = cv2.VideoCapture(args["input"])
    frameRate = "%.2f" % iv.get(cv2.CAP_PROP_FPS)
    frameHeight = int(iv.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameWidth = int(iv.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(frameRate)
     
    # lucas kanade OF parameters
    lk_params = dict(
        winSize=(15,15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, oldFrame = iv.read()
    oldGray = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('First Frame')
    cv2.setMouseCallback('First Frame', getPointOnImage)

    while (not ptSelected):
        cv2.imshow('First Frame', oldFrame)
        k = cv2.waitKey(4) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

    data = trackVideo(iv, oldGray, trackPt, lk_params)

    ae.exportkeyframes(args["output"], frameRate, frameWidth, frameHeight, "1", "1", 1, 1, data)