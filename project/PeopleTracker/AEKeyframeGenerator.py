import sys

def writeAEVersion(f, ver):
    f.write('Adobe After Effects ' + ver + ' Keyframe Data\n')

def writeTrackingMetaData(f, fps, width, height, sourceAR, compAR):
    f.write(
        '\n\tUnits Per Second\t%s\n'
        '\tSource Width\t%d\n'
        '\tSource Height\t%d\n'
        '\tSource Pixel Aspect Ratio\t%s\n'
        '\tComp Pixel Aspect Ratio\t%s\n' % (fps, width, height, sourceAR, compAR));

def writeTrackerInfo(f, trackerNum, trackPtNum):
    f.write('\nMotion Trackers\tTracker #%d\tTrack Point #%d\tFeature Center\n' % (trackerNum, trackPtNum))

def writeDataHeader(f):
    f.write('\tFrame\tX pixels\tY pixels\t\n')

def writeData(f, data):
    for i in range(len(data)):
        f.write('\t%d\t%.3f\t%.3f\t\n' % (i, data[0][0], data[0][1]))

def writeEndOfKeyframeData(f):
    f.write('\n\nEnd of Keyframe Data\n\n')

'''
Export keyframe data to a text file that can be imported to After Effects
fileName: name of output file
fps: framerate of tracking data. This must correspond to your videos framerate.
width: video width
height: video height
sourceAR: source pixel aspect ratio
compAR: comp pixel aspect ratio
trackerNum: the tracker number for your motion tracker
trackPtNum: the track point number for your tracker
data: keyframe data consisting of 2D coordinates
'''
def exportkeyframes(
    fileName,
    fps,
    width,
    height,
    sourceAR,
    compAR,
    trackerNum,
    trackPtNum,
    data):
    f = open(fileName, 'w')
    
    writeAEVersion(f, '8.0')
    writeTrackingMetaData(f, fps, width, height, sourceAR, compAR)
    writeTrackerInfo(f, trackerNum, trackPtNum)
    writeDataHeader(f)
    writeData(f,data)
    writeEndOfKeyframeData(f)

    f.close()