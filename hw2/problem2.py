
import cv2         
import numpy as np

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle
from kivy.base import runTouchApp

# imports
import os
import random
import sys
import math

class SegmentationWidget(Widget):
    pass

class SegmentationApp(App):

    def build(self):

        self.root = root = SegmentationWidget()
        root.bind(size=self._update_rect, pos=self._update_rect)

        imgName = sys.argv[1]
        img = cv2.imread(imgName)
    
        img = np.flip(img,axis=0)

        width = img.shape[1]
        height = img.shape[0]

        texture = Texture.create(size=(width, height), colorfmt="bgr")
        texture.blit_buffer(img.tostring(), bufferfmt="ubyte", colorfmt="bgr")

        with root.canvas.before:
            self.rect = Rectangle(texture=texture, size=root.size, pos=root.pos)

        return root

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

if __name__ == "__main__":
    SegmentationApp().run()