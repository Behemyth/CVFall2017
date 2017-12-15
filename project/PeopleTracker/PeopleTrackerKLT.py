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

newPt = None
newPtSelected = False

def getNewPoint(event, x, y, flags, param):
    global newPtSelected, newPt

    if event == cv2.EVENT_LBUTTONDOWN:
        newPt = (x,y)

    elif event == cv2.EVENT_LBUTTONUP:
        print("Selecting new point...")
        newPt = (x,y)
        newPtSelected = True

def trackVideo(iv, old_gray, trackPt, lk_params):
    global newPtSelected, newPt

    offsetPt = trackPt.copy()
    offsetX = 0.0
    offsetY = 0.0

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

        # if we couldn't find the tracked point in the next frame
        # or if the error (absolute distance) is significant
        # use manual intervention to select a new point
        if st[0] == 0 or err[0] > 15.0:
            print("WARNING LARGE TRACKING ERROR: please select new tracking point")
            cv2.namedWindow('Select New Tracking Point')
            newPtSelected = False
            cv2.setMouseCallback('Select New Tracking Point', getNewPoint)

            while (not newPtSelected):
                cv2.imshow('Select New Tracking Point', frame)
                k = cv2.waitKey(4) & 0xFF
                if k == 27:
                    break

            offsetX = offsetPt[-1][0] - newPt[0]
            offsetY = offsetPt[-1][1] - newPt[1]
            newPos[0] = newPt
            cv2.destroyWindow('Select New Tracking Point')

        x0 = int(offsetPt[-1][0])
        y0 = int(offsetPt[-1][1])
        x1 = int(newPos[0,0] + offsetX)
        y1 = int(newPos[0,1] + offsetY)
        mask = cv2.line(mask, (x0,y0), (x1,y1), (0,0,255), thickness=2, lineType=cv2.LINE_AA)

        trackPt.append(newPos[0])
        offsetPt.append(newPos[0] + (offsetX,offsetY))
        oldGray = frameGray

        img = cv2.add(frame, mask)
        img = imutils.resize(img,width=1920//2)

        cv2.imshow('Tracked Frame', img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    return offsetPt

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
    ap.add_argument("--start", default=0, type=int, help="The starting frame for the track")
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

    for i in range(args["start"] + 1):
        ret, oldFrame = iv.read()
    oldGray = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('First Frame')
    cv2.setMouseCallback('First Frame', getPointOnImage)

    while (not ptSelected):
        cv2.imshow('First Frame', imutils.resize(oldFrame,width=1920//2))
        k = cv2.waitKey(4) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

    print(trackPt)
    trackPt[0] = (trackPt[0][0] * 2, trackPt[0][1] * 2)
    print(trackPt)
    data = trackVideo(iv, oldGray, trackPt, lk_params)

    ae.exportkeyframes(args["output"], frameRate, frameWidth, frameHeight, "1", "1", 1, 1, data)