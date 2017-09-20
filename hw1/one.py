import cv2         
import numpy as np
from matplotlib import pyplot as plt 
import matplotlib   

# imports
import os
import random
import sys
import math

if __name__ == "__main__":
	
	img1Name = sys.argv[1]
	img2Name = sys.argv[2]
	bitName = int(sys.argv[3])
	imgNameOut = sys.argv[4]

	img1 = cv2.imread(img1Name)
	img2 = cv2.imread(img2Name)
	bitmask = cv2.imread(bitName)

	result = np.zeros(img2.shape)

	cv2.imwrite(imgNameOut, result)



