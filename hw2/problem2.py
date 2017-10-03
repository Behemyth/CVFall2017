
import cv2         
import numpy as np

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.graphics.texture import Texture

# imports
import os
import random
import sys
import math

if __name__ == "__main__":
    
    imgName = sys.argv[1]
    img = cv2.imread(imgName)
    
    texture = Texture.create(size=(16, 16), colorfmt="rgb")
    
    arr = np.ndarray(shape=[16, 16, 3], dtype=np.uint8)


    data = arr.tostring()
    texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="rgb")