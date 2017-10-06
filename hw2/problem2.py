
import cv2         
import numpy as np
import maxflow

# app imports
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle
from kivy.base import runTouchApp
from kivy.graphics import Color, Line
from kivy.properties import ObjectProperty

# imports
import os
import random
import sys
import math

class SegmentationWidget(Widget):
    
    def __init__(self, *args, **kwargs):
        #init the firsttouch and call the super 
        super(SegmentationWidget, self).__init__(*args, **kwargs)
        self.firstTouch = True

    def on_touch_down(self, touch):

        with self.canvas:

            if self.firstTouch:
                #create the first set of points for the rectangle
                self.Ax, self.Ay = touch.x, touch.y

            #start the new Line gand give the first point
            touch.ud['line'] = Line(points=(touch.x, touch.y))

    def on_touch_move(self, touch):

        if self.firstTouch:
            # Assign the position of the touch at the point C
            self.Cx, self.Cy = touch.x, touch.y

            # There are two known points A (starting point) and C (endpoint)
            # Assign the positions x and y known of the points
            self.Bx, self.By = self.Cx, self.Ay
            self.Dx, self.Dy = self.Ax, self.Cy

            # Assign points positions to the last line created
            
            touch.ud['line'].points.clear()
            touch.ud['line'].points += [self.Ax, self.Ay]
            touch.ud['line'].points += [self.Bx, self.By]
            touch.ud['line'].points += [self.Cx, self.Cy]
            touch.ud['line'].points += [self.Dx, self.Dy]
            touch.ud['line'].points += [self.Ax, self.Ay]

        else:
            #continue the list of points
            touch.ud['line'].points += [touch.x, touch.y]

    def on_touch_up(self, touch):

        #TODO update graph based on firstTouch and the lines


        #change the draw type
        self.firstTouch = False

        #remove the lines and update the graph based on this line       
        self.canvas.clear()
        

class SegmentationApp(App):

    def build(self):
        #load the image and pass it to the widget to render
        self.root = root = SegmentationWidget()
        root.bind(size=self._update_rect, pos=self._update_rect)

        imgName = sys.argv[1]
        img = cv2.imread(imgName)
    
        #align image
        img = np.flip(img,axis=0)

        width = img.shape[1]
        height = img.shape[0]

        #attach the image to a texture
        texture = Texture.create(size=(width, height), colorfmt="bgr")
        texture.blit_buffer(img.tostring(), bufferfmt="ubyte", colorfmt="bgr")

        with root.canvas.before:
            #attach the texture to the app
            self.rect = Rectangle(texture=texture, size=root.size, pos=root.pos)

        return root

    def _update_rect(self, instance, value):
        #resizing app
        self.rect.pos = instance.pos
        self.rect.size = instance.size

if __name__ == "__main__":
    SegmentationApp().run()