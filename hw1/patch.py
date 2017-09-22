import cv2
import numpy as np

def gradient(img,y,x):
	grad = np.array([0.0, 0.0, 0.0])
	grad = img[y, x] * 4 - img[y + 1, x] - img[y - 1, x] - img[y, x + 1] - img[y, x - 1]

	return grad

def isophote(patch, patchSize):
	p = patchSize//2+1
	pGrad = np.array([patch[p,p,:] - patch[p+1,p,:], patch[p,p,:] - patch[p,p+1,:]])

	return pGrad / np.linalg.norm(pGrad)

def orthoomega(bitPatch, patchSize):
	p = patchSize//2+1
	small = bitPatch[p-1:p+1,p-1:p+1,:]
	sobelX = -small[0,0,:] - 2*small[0,1,:] - small[0,2,:] + small[2,0,:] + 2*small[2,1,:] + small[2,2,:]
	sobelY = -small[0,0,:] - 2*small[1,0,:] - small[2,0,:] + small[0,2,:] + 2*small[1,2,:] + small[2,2,:]
	sobelVec = np.array([sobelX,sobelY])
	return sobelVec / np.linalg.norm(sobelVec)

def confidence(bitPatch, patchSize):
    total = 0

    for i in range(patchSize):
        for j in range(patchSize):
            if not bitPatch[i,j]:
                total += 1

    return total / (patchSize * patchSize)

def data(patch, bitPatch, patchSize):
	# p is the coordinate of the center pixel of the patch
	p = patchSize//2+1

	# find the scalar magnitude of the gradient at p
    gradMag = np.linalg.norm(np.linalg.norm(gradient(patch, p, p)))

	# find the isophote at p
    isophote = isophote(patch, patchSize)

	# find the
	normalAtP = orthoomega(bitPatch, patchSize)

	return gradMag * np.linalg.norm(np.dot(isophote, normalAtP))

def inpaint(img, bitMask, patchSize):
    height = img.shape[0]
    width = img.shape[1]

	importantX = 0
	importantY = 0
	importance = 0

	# go through every pixel checking its importance
    for i in range(patchSize // 2 + 1, height - patchSize // 2):
        for j in range(patchSize // 2 + 1, width - patchSize // 2):
			# check if pixel is on an edge
			# we're not too worried about selecting a pixel not on the seam since
			# confidence would be too low to select it
            if bitMask[i,j,:]:
				# select a neighborhood around the current pixel of size patchSize*patchSize
                bitPatch = bitMask[i-patchSize//2:i+patchSize//2,j-patchSize//2:j+patchSize//2,:]
                patch = img[i-patchSize//2:i+patchSize//2,j-patchSize//2:j+patchSize//2,:]

				# compute the confidence and data terms around this pixel
                cP = confidence(bitPatch, patchSize)
                dP = data(patch, bitPatch, patchSize)

				# if the combined score is higher than the highest "importance" score,
				# select this pixel instead
				if cP*dP > importance:
					importance = cP * dP
					importantY = i
					importantX = j

	# now search for the known patch that's most similar and replace our unknown patch

if __name__ == '__main__':
    inputImgSrc = sys.argv[1]
    bitMaskSrc = sys.argv[2]
    outputImgSrc = sys.argv[3]
    patchSize = sys.argv[4]

    inputImg = cv2.imread(inputImgSrc)
    bitMask = cv2.imread(bitMaskSrc)[:,:,:1]

    # set any non-zero values to 1
    bitMask[bitMask > 0] = 1

    outputImg = inpaint(inputImg, bitMask, patchSize)

    cv2.imwrite(outputImgSrc, outputImg)
