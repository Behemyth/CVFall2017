import cv2
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

# imports
import os
import random
import sys
import math

def poisson(source, bitmask):
	height = source.shape[0]
	width = source.shape[1]

	mask3 = np.repeat(bitmask, 3, axis = 2)

	product = height * width
	coeff = scipy.sparse.identity(product, format='lil')
	gradients = np.zeros((product, 3), dtype=np.float)

	#process coefficients and gradients

	for y in range(height):
		for x in range(width):

			index = x + y * width

			if bitmask[y,x] == 1:
				coeff[index, index] = 4
				grad = [0.0,0.0,0.0] + source[y, x] * 4.0

				if y - 1 >= 0:
					grad -= source[y - 1, x]
					coeff[index, index - 1] = -1

				if y + 1 < height:
					grad -= source[y + 1, x]
					coeff[index, index + 1] = -1

				if x - 1 >= 0:
					grad-= source[y, x - 1]
					coeff[index, index - width] = -1

				if x + 1 < width:
					grad -= source[y, x + 1]
					coeff[index, index + width] = -1

				gradients[index] = 0.0

			else:
				gradients[index] = source[y, x]


	coeff = coeff.tocsr()

	colors = np.zeros(source.shape,source.dtype)

	for i in range(3):
		x = scipy.sparse.linalg.spsolve(coeff, gradients[:, i])

		#can be 318 or <0, so clamp
		x[x > 255] = 255
		x[x < 0] = 0

		colors[:,:,i] = x.reshape(height,width).astype(np.uint8)

	'''
	#mix the gradients
	alpha = 0.5
	rGradientSource = (alpha) * rGradientSource + (1 - alpha) * rGradientTarget
	gGradientSource = (alpha) * gGradientSource + (1 - alpha) * gGradientTarget
	bGradientSource = (alpha) * bGradientSource + (1 - alpha) * bGradientTarget
	'''

	return colors


if __name__ == "__main__":

	img1Name = sys.argv[1]
	bitName = sys.argv[2]
	imgNameOut = sys.argv[3]

	source = cv2.imread(img1Name)
	bitmask = cv2.imread(bitName,0).reshape(source.shape[0],source.shape[1],1)#read only one channel (they should all the same)

	bitmask[bitmask < 255] = 0
	bitmask[bitmask >= 255] = 1

	#bitmask is now either 0 or 1

	result = poisson(source,bitmask)

	'''
	#OpenCV implementation
	center = (int(target.shape[1]/2),int(target.shape[0]/2))

	result = cv2.seamlessClone(source, target, bitmask, center, cv2.NORMAL_CLONE)
	'''

	cv2.imwrite(imgNameOut, result)
