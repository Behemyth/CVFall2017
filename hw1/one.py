import cv2         
import numpy as np
import scipy.sparse

# imports
import os
import random
import sys
import math

def poisson(source,target, bitmask):
	height = source.shape[0]
	width = source.shape[1]

	result = np.zeros((source.shape))
	mask = np.broadcast_to(bitmask == 0, result.shape)
	result = target * mask

	mask = np.invert(mask)

	product = target.shape[0] * target.shape[1]
	coeff = scipy.sparse.identity(product, format='dia')

	#process coefficients
	'''
	for y in range(height):
		for x in range(width):
			if not mask[y,x,1]:
				print(y,x)
				index = x+y*width
				coeff[index, index] = 4
				if index+1 < product:
					coeff[index, index+1] = -1
				if index-1 >= 0:
					coeff[index, index-1] = -1
				if index+width < product:
					coeff[index, index+width] = -1
				if index-width >= 0:
					coeff[index, index-width] = -1
	'''


	#calculate gradients
	rGradientTarget = None
	gGradientTarget = None
	bGradientTarget = None

	rGradientSource = None
	gGradientSource = None
	bGradientSource = None

	#mix the gradients
	alpha = 0.5
	rGradientSource = (alpha) * rGradientSource + (1 - alpha) * rGradientTarget
	gGradientSource = (alpha) * gGradientSource + (1 - alpha) * gGradientTarget
	bGradientSource = (alpha) * bGradientSource + (1 - alpha) * bGradientTarget

	#get the colors
	rCol = coeff / rGradient
	gCol = coeff / gGradient
	bCol = coeff / bGradient

	#wrap into one image
	colors = npm.concatonate(rCol,gCol,bCol, axis = 2)

	result += colors * mask
	return result


if __name__ == "__main__":
	
	img1Name = sys.argv[1]
	img2Name = sys.argv[2]
	bitName = sys.argv[3]
	imgNameOut = sys.argv[4]

	source = cv2.imread(img1Name)
	target = cv2.imread(img2Name)
	bitmask = cv2.imread(bitName)[:,:,:1] #read only one channel (they should all the same)

	mask = bitmask > 0
	bitmask[mask] = 1
	#bitmask is now either 0 or 1
	
	result = poisson(source, target,bitmask)

	'''
	#OpenCV implementation
	center = (int(target.shape[1]/2),int(target.shape[0]/2))
	
	result = cv2.seamlessClone(source, target, bitmask, center, cv2.NORMAL_CLONE)
	'''

	cv2.imwrite(imgNameOut, result)



