import cv2         
import numpy as np
from matplotlib import pyplot as plt 
import matplotlib   

# imports
import os
import random
import sys

def GetMatches(descriptorI,descriptorJ):

    bf = cv2.BFMatcher()
    matches = np.array(bf.knnMatch(descriptorI,descriptorJ, k=2))
    return [[match[0]] for match in matches if (match[0].distance < 0.75 * match[1].distance) ]

def distanceOf(a):
	return a.distance

#this function is taken from the opencv documentation
def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
    return img1,img2

if __name__ == "__main__":

    matchCount = 15

    image1Name = sys.argv[1]
    image2Name = sys.argv[2]

    #setup
    sift = cv2.xfeatures2d.SIFT_create()

    imgI = cv2.imread(image1Name)
    imgJ = cv2.imread(image2Name)

    grayI = cv2.cvtColor(imgI, cv2.COLOR_BGR2GRAY)
    grayJ = cv2.cvtColor(imgJ, cv2.COLOR_BGR2GRAY)

    bestName = 'image'

    #get features
    keypointsI, descriptorI = sift.detectAndCompute(grayI,None)
    keypointsJ, descriptorJ = sift.detectAndCompute(grayJ,None)
                
    matches = GetMatches(descriptorI,descriptorJ)

    #extract data from the knn data, list comprehension taken from opencv
    #documentation
    keypointsI = np.array([keypointsI[match[0].queryIdx] for match in matches])
    keypointsJ = np.array([keypointsJ[match[0].trainIdx] for match in matches])
    descriptorI = np.array([descriptorI[match[0].queryIdx] for match in matches])
    descriptorJ = np.array([descriptorJ[match[0].trainIdx] for match in matches])
    pointsI = np.array([keypoint.pt for keypoint in keypointsI])
    pointsJ = np.array([keypoint.pt for keypoint in keypointsJ])

    F, mask = cv2.findFundamentalMat(pointsI, pointsJ, cv2.FM_RANSAC)

    #mask it all
    mask = mask.ravel()

    keypointsI = keypointsI[mask == 1]
    keypointsJ = keypointsJ[mask == 1]
    descriptorI = descriptorI[mask == 1]
    descriptorJ = descriptorJ[mask == 1]
    pointsI = pointsI[mask == 1]
    pointsJ = pointsJ[mask == 1]

    lines1 = cv2.computeCorrespondEpilines(pointsJ.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    linesI,dummy1 = drawlines(grayI,grayJ,lines1,pointsI,pointsJ)

    lines2 = cv2.computeCorrespondEpilines(pointsI.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    linesJ,dummy2 = drawlines(grayJ,grayI,lines2,pointsJ,pointsI)

    #write the image
    cv2.imwrite(bestName + "_EpipolarLines1.jpg",linesI)

    #write the image
    cv2.imwrite(bestName + "_EpipolarLines2.jpg",linesJ)

    matches = GetMatches(descriptorI,descriptorJ)


    size = (imgI.shape[0]*2,imgI.shape[1]*2)
    retBool, matrixI, matrixJ = cv2.stereoRectifyUncalibrated(pointsI,pointsJ,F,size)

    #sizes
    heightI, widthI = imgI.shape[0], imgI.shape[1]
    heightJ, widthJ = imgJ.shape[0], imgJ.shape[1]

    #taken from opencv documentation
    cornersI = np.array([[0,0], [0,heightI], [widthI,heightI], [widthI,0]], np.float32).reshape(-1,1,2)
    cornersJ = np.array([[0,0], [0,heightJ], [widthJ,heightJ], [widthJ,0]], np.float32).reshape(-1,1,2)

    #merge into one coordinate system
    boundsI = np.concatenate((cornersI, cv2.perspectiveTransform(cornersI, matrixI)), axis=0)
    boundsJ = np.concatenate((cornersI, cv2.perspectiveTransform(cornersJ, matrixJ)), axis=0)

    bounds = np.concatenate((boundsI,boundsJ), axis=0)

    #grab the min/max of the bounding box
    [xMin, yMin] = bounds.min(axis=0).ravel()
    [xMax, yMax] = bounds.max(axis=0).ravel()

    #round to nearest pixel value
    xMin = np.round(xMin).astype(np.int32)
    yMin = np.round(yMin).astype(np.int32)
    xMax = np.round(xMax).astype(np.int32)
    yMax = np.round(yMax).astype(np.int32)

    #pure translation homorgraphy
    translatedI = np.array([[1, 0, -xMin],
                            [0, 1, -yMin], 
                            [0, 0, 1]]).dot(matrixI)

    #pure translation homorgraphy
    translatedJ = np.array([[1, 0, -xMin],
                            [0, 1, -yMin], 
                            [0, 0, 1]]).dot(matrixJ)

    #for affine
    translate = np.float32([[1, 0, -xMin],
                            [0, 1, -yMin]])

    #size of the image to warp to
    size = (xMax - xMin, yMax - yMin)

    #best interp with opencv
    warpedI = cv2.warpPerspective(imgI, translatedI, size)
    #best interp with opencv
    warpedJ = cv2.warpPerspective(imgJ, translatedJ, size)

    #write the image
    cv2.imwrite(bestName + "_Warped1.jpg",warpedI)

    #write the image
    cv2.imwrite(bestName + "_Warped2.jpg",warpedJ)

    stereo = cv2.StereoBM_create(numDisparities=16*12, blockSize=9)

    grayWarpedI = cv2.cvtColor(warpedI, cv2.COLOR_BGR2GRAY)
    grayWarpedJ = cv2.cvtColor(warpedJ, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(grayWarpedI,grayWarpedJ)

    #write the image
    cv2.imwrite(bestName + "_Disparity.jpg",disparity)
