import cv2
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

# imports
import os
import random
import sys
import math

def poisson(source,target, bitmask):
	height = source.shape[0]
	width = source.shape[1]

	mask3 = np.repeat(bitmask, 3, axis = 2)

	product = height * width
	coeff = scipy.sparse.identity(product, format='lil')
	gradients = np.zeros((product, 3), dtype=np.float)

	#process coefficients and gradients

	#iterate over all pixels
	for y in range(height):
		for x in range(width):

			#grab the coefficient index
			index = x + y * width

			#only edit where the mask is true
			if bitmask[y,x] == 1:
				tempGradient = np.array([0.0, 0.0, 0.0])
				coeff[index, index] = 4
				grad = [0.0,0.0,0.0] + source[y, x] * 4.0

				#mixed gradient
				gradT = [0.0,0.0,0.0] + target[y, x] * 4.0
				alpha = 0.5

				if y - 1 >= 0:
					grad -= source[y - 1, x]
					gradT -= target[y - 1, x]

					if bitmask[y - 1, x] == 1:
						coeff[index, index - 1] = -1
					else:
						tempGradient += target[y - 1, x]

				if y + 1 < height:
					grad -= source[y + 1, x]
					gradT -= target[y + 1, x]

					if bitmask[y + 1, x] == 1:
						coeff[index, index + 1] = -1

					else:
						tempGradient += target[y + 1, x]

				if x - 1 >= 0:
					grad-= source[y, x - 1]
					gradT-= target[y, x - 1]

					if bitmask[y, x - 1] == 1:
						coeff[index, index - width] = -1
					else:
						tempGradient += target[y, x - 1]

				if x + 1 < width:
					grad -= source[y, x + 1]
					gradT -= target[y, x + 1]

					if bitmask[y, x + 1] == 1:
						coeff[index, index + width] = -1
					else:
						tempGradient += target[y, x + 1]

				#gradients[index] = grad + tempGradient
				gradients[index] = (alpha) * (grad + tempGradient) + (1 - alpha) * (gradT + tempGradient)

			else:
				gradients[index] = target[y, x]


	coeff = coeff.tocsr()

	colors = np.zeros(source.shape,source.dtype)

	#iterate over each channel and solve the matrices
	for i in range(3):
		x = scipy.sparse.linalg.spsolve(coeff, gradients[:, i])

		#can be 318 or <0, so clamp
		x[x > 255] = 255
		x[x < 0] = 0

		colors[:,:,i] = x.reshape(height,width).astype(np.uint8)

	'''
	#mix the gradients (initial implementation)
	alpha = 0.5
	rGradientSource = (alpha) * rGradientSource + (1 - alpha) * rGradientTarget
	gGradientSource = (alpha) * gGradientSource + (1 - alpha) * gGradientTarget
	bGradientSource = (alpha) * bGradientSource + (1 - alpha) * bGradientTarget
	'''

	return colors


if __name__ == "__main__":

	img1Name = sys.argv[1]
	img2Name = sys.argv[2]
	bitName = sys.argv[3]
	imgNameOut = sys.argv[4]

	#read in all images
	source = cv2.imread(img1Name)
	target = cv2.imread(img2Name)
	bitmask = cv2.imread(bitName,0).reshape(target.shape[0],target.shape[1],1)#read only one channel (they should all the same)

	bitmask[bitmask < 255] = 0
	bitmask[bitmask >= 255] = 1

	#bitmask is now either 0 or 1

	result = poisson(source, target,bitmask)

	'''
	#OpenCV implementation
	center = (int(target.shape[1]/2),int(target.shape[0]/2))

	result = cv2.seamlessClone(source, target, bitmask, center, cv2.NORMAL_CLONE)
	'''

	cv2.imwrite(imgNameOut, result)
