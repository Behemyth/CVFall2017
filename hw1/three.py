import cv2         
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

# imports
import os
import random
import sys
import math

# Runs through all of the frames in the video
# Returns image that is an average of all the frames
def FlattenImage(vid) :
	# Holds the sum of all the frames' RGB values at each pixel
	sumImg = np.zeros( (int(vid.get(4)), int(vid.get(3)), 3) )

	# Loop through all of the frames or either source or target and add to sum
	while(True):
		ret, frame = vid.read()
		if (not ret):
			# Reached end of file
			break
		sumImg += frame

	return sumImg / vid.get(3)

# Determines the best seam to composite two images together
# Source will be image on the left of the seam, target will be image on the right
# Seam is described as pixels in each row in which pixels to the left and itself are from source 
# and pixels to the right are from the target
# Returns the matrix of cost energies at each pixel
def FindSeam(source, target):
	# Calculate the forward energy of seam, storing it in a row X col matrix
	energy = np.zeros((source.shape[0], source.shape[1]))
	for y in range(source.shape[0]):
		for x in range(source.shape[1]):
			# Forward energy is the absolute difference between the current pixel and the one to the right

			# If looking at the top row of pixels, don't need to look at row before it
			if (y == 0):
				if (x == source.shape[1]-1):
					# This is the last pixel in the first row
					# Energy would just be the magnitude of the pixel
					energy[y][x] = np.linalg.norm(source[y][x])
				else:
					# Energy is the difference between the right pixel from the target
					# and the current pixel from the source
					energy[y][x] = np.linalg.norm(np.absolute(target[y][x+1] - source[y][x]))

			else:
				# Energy is the energy from the pixel before plus the energy of the current pixel
				if (x == source.shape[1]-1):
					# Only need to look at 2 pixels above
					currentEnergy = np.linalg.norm(source[y][x])
					energy[y][x] = min( (energy[y-1][x] + currentEnergy), (energy[y-1][x-1] + currentEnergy) )
				else:
					# Need to look at the 3 pixels above because that is how got to current pixel
					currentEnergy = np.linalg.norm(np.absolute(target[y][x+1] - source[y][x]))
					energy[y][x] = min( min( (energy[y-1][x] + currentEnergy), (energy[y-1][x-1] + currentEnergy) ), 
									(energy[y-1][x+1] + currentEnergy))

	return energy

# Given a matrix of cost energies, backtrace along the seam to create the mask
def CreateMask(energy):
	# Look at the bottom row and backtrace to find the seam and create mask from it
	minCost = energy[-1][0]
	index = 0
	for x in range(energy.shape[1]):
		if (energy[-1][x] < minCost):
			minCost = energy[-1][x]
			index = x

	# Backtracing along the seam and creating a mask from it
	mask = np.zeros( (energy.shape) )
	mask[-1][:index+1] = 255  # making the whole row up to and including the index a 1
	print(energy.shape[0])
	for row in range(energy.shape[0]-1,0,-1):
		# At each row, look at the row above and of the 3 possible pixels that the min cost path could've 
		# traversed, take the index of the pixel with the smallest energy cost
		# Make the whole row up to and including the index 1
		minCost = min( min( energy[row-1][index-1], energy[row-1][index+1] ), energy[row-1][index] )

		if (energy[row-1][index-1] == minCost):
			index = index - 1
		elif (energy[row-1][index+1] == minCost):
			index = index + 1

		mask[row-1][:index+1] = 255

	# Blurring the mask to feather the edges
	mask = cv2.GaussianBlur(mask, (11,11), 0)

	return mask

# Cuts each frame of the video along the seam in the mask
# Composites the source on the left and the target on the right and creates the video file
def CreateComposite(source, target, mask, outputName):
	compositeVideo = cv2.VideoWriter(outputName, 
									cv2.VideoWriter_fourcc('h','2','6','4'),  # using H.264 codec
									source.get(5),  # setting the frame rate of composite to be same as source
									(int(source.get(3)), int(source.get(4))))  # setting size of composite to be same as source
	
	# Need to convert the values in the mask from grayscale to 0-1
	print(mask)
	mask = (255.0 - mask) / 255.0
	print(mask)

	# Loop through frames of source and target and
	# composite them together until no more frames in one
	while(True):
		retSource, frameSource = source.read()
		retTarg, frameTarget = target.read()

		if (not retSource or not retTarg):
			# Reached end of one the files
			break

		# Composite source and target using linear interpolation
		compositeImage = mask * frameSource + (1-mask) * frameTarget
		print(compositeImage.shape)
		print(frameSource.shape)
		# print(compositeImage)

		# Add this frame to the composite video
		compositeVideo.write(compositeImage)

	# Close the video writer
	compositeVideo.release()


if __name__ == "__main__":
	# Checking for 3 arguments and terminates if 3 arguments are not given
	if (len(sys.argv) != 4) :
		print('Invalid number of arguments')
		sys.exit()

	sourceName = sys.argv[1]
	targetName = sys.argv[2]
	outputName = sys.argv[3]

	# Open the videos
	source = cv2.VideoCapture(sourceName)
	target = cv2.VideoCapture(targetName)

	# Checks that the videos opened properly
	if (not source.isOpened() or not target.isOpened()) :
		print('Videos did not open correctly')
		sys.exit()

	# Makes sure that the videos are the same size
	if (source.get(3) != target.get(3) and \
		source.get(4) != target.get(4)):
		print('Videos are not the same size')
		sys.exit()

	# Get video size and create an empty image of the same size
	width = int(source.get(3))  # 3rd property of videos is the width
	height = int(source.get(4)) # 4th property of videos is the height
	combinedAvg = np.zeros((height, width, 3))  # holds the average of all frames

	# Get the average frame of each video 
	# avgSource = FlattenImage(source)
	# avgTarget = FlattenImage(target)
	avgSource = cv2.imread('avgSource.jpg')
	avgTarget = cv2.imread('avgTarget.jpg')

	# cv2.imwrite('avgSource.jpg', avgSource)
	# cv2.imwrite('avgTarget.jpg', avgTarget)

	# Reset the video to start at the beginning (?? NEED TO DO ??)
	source.open(sourceName)
	target.open(targetName)

	# mask = CreateMask(FindSeam(avgSource, avgTarget))
	mask = cv2.imread('twin_mask.jpg')

	# cv2.imwrite('twin_mask.jpg', mask)

	CreateComposite(source, target, mask, outputName)
