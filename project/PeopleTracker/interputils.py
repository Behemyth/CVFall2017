import numpy as np

# # Performs interpolation along a Bezier curve
# # p0, p1, p2, and p3 are 2D coordinates
# # t is a time value from 0 to 1
# def bezier_interp(p0, p1, p2, p3, t):
#     l01 = p1 - p0
#     l12 = p2 - p1
#     l23 = p3 - p2

#     p01 = l01 * t + p0
#     p12 = l12 * t + p1
#     p23 = l23 * t + p2

#     l02 = p12 - p01
#     l13 = p23 - p12

#     p02 = l02 * t + p01
#     p13 = l13 * t + p12

#     l03 = p13 - p02

#     p = l03 * t + p02

#     return p

def linear_interp(p0, p1, t):
    print(t)
    print(p0)
    print(p1)
    return (1-t)*p0 + t*p1

def quadratic_interp(p0, p1, p2, t):
    term1 = (1-t) ** 2 * p0
    term2 = 2 * (1-t) * p1
    term3 = t ** 2 * p2
    return term1 + term2 + term3

def cubic_interp(p0, p1, p2, p3, t):
    term1 = (1-t) ** 3 * p0
    term2 = 3 * (1-t) ** 2 * t * p1
    term3 = 3 * (1-t) * t ** 2 * p2
    term4 = t ** 3 * p3
    return term1 + term2 + term3 + term4

def interp_frames(p0, p1, numFrames):
    # frames = np.fromfunction(lambda i: linear_interp(p0,p1,i/numFrames), (numFrames,), dtype=int)
    frames = np.zeros((numFrames,4))
    for i in range(numFrames):
        frames[i] = linear_interp(p0,p1,i/numFrames)
    print(frames.shape)
    return frames

def interp_entire_video(knownPts, numFrames):
    iPts = np.zeros((numFrames,4))
    
    for i in range(len(knownPts)-1):
        curFrame = knownPts[i][1]
        curRect = knownPts[i][0]

        nextFrame = knownPts[i+1][1]
        nextRect = knownPts[i+1][0]

        print('cuframe %d next frame %d' % (curFrame, nextFrame))

        if i == 0 and curFrame > 0:
            iPts[0:curFrame] = np.repeat(curRect,curFrame)

        print(iPts[curFrame:nextFrame].shape)
        if curFrame == nextFrame - 1:
            iPts[curFrame] = curRect
        else:
            iPts[curFrame:nextFrame] = interp_frames(curRect, nextRect, nextFrame - curFrame)

    return iPts