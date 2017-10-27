# 1. Select 4 bounding boxes around each dot
# 2. For each frame:
#     a. search some radius around the bounding box
#     b. select a new bounding box from your search such that the SDD is minimized
#     c. compute projective transformation based on the centers of each bounding box
#     d. warp an image using that projection transformation
#     e. output to png file

import cv2
import numpy as np
import sys
import os
import math

WHITE = [255,255,255]

def GetPointsFromBoxes(boxes):
  p = np.zeros((4,2), dtype = "float32")

  for i in range(len(boxes)):
    p[i] = [boxes[i][0] + boxes[i][2]/2.0, boxes[i][1] + boxes[i][3]/2.0]

  return p

# find the sum of squares difference between two images
def SumOfSquaresDifference(im1, im2):
  diffIm = (im2 - im1)**2
  
  return np.sum(diffIm)

# find the best bounding box in the next frame within a given radius
# that most closely matches the current bounding box
def BestBoundingBoxInRegion(prevFrame, curFrame, box, radius):
  oldRegion = prevFrame[int(box[1]):int(box[1]+box[3]), int(box[0]):int(box[0]+box[2])]
  testRegion = curFrame[(int(box[1]) - radius):(int(box[1]+box[3]) - radius), (int(box[0]) - radius):(int(box[0]+box[2]) - radius)]
  bestSSD = math.inf
  newBox = ((int(box[0]) - radius), (int(box[1]) - radius), box[2], box[3])
  
  h,w = curFrame.shape

  for i in range(-radius,radius):
    for j in range(-radius,radius):
      if ((int(box[1]) - i) < 0 or (int(box[0]) - j) < 0 or (int(box[1]+box[3]) - i) >= h or (int(box[0]+box[2]) - j) >= w):
        continue

      testRegion = curFrame[(int(box[1]) - i):(int(box[1]+box[3]) - i), (int(box[0]) - j):(int(box[0]+box[2]) - j)]

      #the harris corners for both regions
      oldCorners = cv2.cornerHarris(oldRegion,4,1,0.04)
      testCorners = cv2.cornerHarris(testRegion,4,1,0.04)

      testSSD = SumOfSquaresDifference(oldCorners, testCorners)
      if (testSSD < bestSSD):
        bestSSD = testSSD
        newBox = ((int(box[0]) - j), (int(box[1]) - i), box[2], box[3])
        
  return newBox

# selects 4 bounding boxes around each dot
# NOTE: when selecting bounding box, select from the center of the dot!
def SelectBoundingBoxes(stillFrame):
  cv2.namedWindow('ROIs')
  r1 = cv2.selectROI('ROIs', stillFrame)
  r2 = cv2.selectROI('ROIs', stillFrame)
  r3 = cv2.selectROI('ROIs', stillFrame)
  r4 = cv2.selectROI('ROIs', stillFrame)
  
  return [r1, r2, r3, r4]

def CreateComposite(source, inputImage, outputName, startFrame, numFrames, radius):
    
  filename, fileExtension = os.path.splitext(outputName)
  
  #spin to the start frame, no checks here
  for sp in range(startFrame):
    retSource, frameSource = source.read()
    frameSource = cv2.resize(frameSource, (1280, 720))
    # frameSource = cv2.copyMakeBorder(frameSource, radius+1, radius+1, radius+1, radius+1, cv2.BORDER_CONSTANT, value = WHITE)
        
  #image dimensions
  h,w,d = inputImage.shape
  
  #the image coordinates that will be transformed
  imagePoints = [ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]
  
  #the point information for this frame and last frame
  framePoints = []
  framePointsPrev = [] 
  
  #storage for the boxes
  boxes = SelectBoundingBoxes(frameSource)
  
  #signifies the first frame
  firstFrame = True
  
  for frameIndex in range(numFrames):
    oldSource = frameSource

    #Read the source video by a frame
    retSource, frameSource = source.read()

    if (not retSource):
      break

    frameSource = cv2.resize(frameSource, (1280, 720))
    # frameSource = cv2.copyMakeBorder(frameSource, radius+1, radius+1, radius+1, radius+1, cv2.BORDER_CONSTANT, value = WHITE)

    oldGray = cv2.cvtColor(oldSource, cv2.COLOR_BGR2GRAY)
    newGray = cv2.cvtColor(frameSource, cv2.COLOR_BGR2GRAY)

    outIm = frameSource

    for bindex in range(len(boxes)):
      print('hello ' + str(frameIndex) + ' ' + str(bindex) + '\n')
      boxes[bindex] = BestBoundingBoxInRegion(oldGray, newGray, boxes[bindex], radius)
      cv2.rectangle(outIm, (boxes[bindex][0], boxes[bindex][1]), (boxes[bindex][2] + boxes[bindex][0], boxes[bindex][3] + boxes[bindex][1]), (255,0,255))

    pTarget = GetPointsFromBoxes(boxes)
    pInput = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype = "float32")
    M = cv2.getPerspectiveTransform(pInput, pTarget)
    warped = cv2.warpPerspective(inputImage, M, (1280, 720))
    warpmatte = warped == 0
    outIm = warped + outIm * warpmatte
    cv2.imwrite(filename + '_' + str(frameIndex) + fileExtension, outIm)
  
if __name__ == "__main__":
    vidFile = sys.argv[1]
    startFrame = int(sys.argv[2])
    numFrames = int(sys.argv[3])
    searchRadius = int(sys.argv[4])
    outFileName = sys.argv[5]
    inputImageName = sys.argv[6]                            
    
    source = cv2.VideoCapture(vidFile)
    inputImage = cv2.imread(inputImageName)
    CreateComposite(source,inputImage, outFileName, startFrame, numFrames, searchRadius)