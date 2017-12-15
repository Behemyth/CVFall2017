import cv2
import numpy as np
import os
import sys
import argparse as ap
from matplotlib import pyplot as plt

# 1. detect features in query and frame
# 2. match features
# 3. take top matching features and find homography
# 4. complete perspective transformation on frame to get new query image
def processFrame(query, pos, frameLarge):
    h,w,d = query.shape
    hl,wl,dl = frameLarge.shape
    sy = pos[0] - 100
    sx = pos[1] - 100

    if (sy < 0):
        sy = 0
    if (sx < 0):
        sx = 0

    by = pos[0] + h + 100
    bx = pos[1] + h + 100

    if (by >= hl):
        by = hl-1
    if (bx >= wl):
        bx = wl-1

    frame = frameLarge[sy:by, sx:bx]
    # initiate ORB detector
    # # Initiate SIFT detector
    # sift = cv2.xfeatures2d.SIFT_create()

    # # find the keypoints and descriptors with SIFT
    # kp1, des1 = sift.detectAndCompute(query,None)
    # kp2, des2 = sift.detectAndCompute(frame,None)

    # # BFMatcher with default params
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1,des2, k=2)

    # # Apply ratio test
    # good = []
    # for m,n in matches:
    #     if m.distance < 0.75*n.distance:
    #         good.append([m])

    # # cv2.drawMatchesKnn expects list of lists as matches.
    # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)

    # plt.imshow(img3),plt.show()
    orb = cv2.ORB_create(nfeatures=10000)

    (kpq, desq) = orb.detectAndCompute(query, None)
    (kpf, desf) = orb.detectAndCompute(frame, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(desq, desf)
    matches = sorted(matches, key = lambda x:x.distance)[:25]

    # img3 = cv2.drawMatches(query,kpq,frame,kpf,matches[:40], None, flags=2)

    # plt.imshow(img3),plt.show()

    print(len(matches))

    query_pts = np.float32([kpq[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    frame_pts = np.float32([kpf[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    (M, mask) = cv2.findHomography(query_pts, frame_pts, cv2.RANSAC, 1.0)

    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = np.int32(cv2.perspectiveTransform(pts,M))
    
    minx = np.min(dst[:,:,0])
    miny = np.min(dst[:,:,1])
    maxx = np.max(dst[:,:,0])
    maxy = np.max(dst[:,:,1])

    if minx < 0:
        minx = 0

    if miny < 0:
        miny = 0

    newQuery = frame[miny:miny+h,minx:minx+w]
    cv2.imshow("Frame", frame)
    cv2.imshow("New Query Image", newQuery)
    cv2.waitKey(0)

    return (newQuery, [miny, minx])

# gets the bounding box of the person from the first frame
def getBoundingBox(frame):
    r = cv2.selectROI(frame)
    imCrop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    return (imCrop, [r[1], r[0]])

# main function
if __name__ == "__main__":
    inputVid = sys.argv[1]
    outputVid = sys.argv[2]

    iv = cv2.VideoCapture(inputVid)

    if (not iv.isOpened()):
        sys.exit(1)

    (ret, frame) = iv.read()

    (query, pos) = getBoundingBox(frame)
    cv2.imshow("Query Image", query)

    newQuery = np.copy(query)
    
    while (iv.isOpened()):
        (ret, frame) = iv.read()

        (newQuery, pos) = processFrame(newQuery, pos, frame)
