import cv2         
import numpy as np
from matplotlib import pyplot as plt 
import matplotlib   

# imports
import os
import random
import sys

if __name__ == "__main__":

   videoName = sys.argv[1]

   vidSource = cv2.VideoCapture(videoName)

