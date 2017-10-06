
import cv2         
import numpy as np
import maxflow

# app imports
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,disable_multitouch')

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

sigma = 2
lamb = 1

'''
taken from https://stackoverflow.com/questions/32328179/opencv-3-0-python-lineiterator
'''

def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 2], row = [x,y])     
    """
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    
    ma = np.maximum(dYa,dXa)
    itbuffer = np.empty(shape=(ma,2),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y + 1,P1Y + dYa + 1)              
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X - 1,P1X - dXa - 1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X + 1,P1X + dXa + 1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32) / dY.astype(np.float32)
            if negY:
                itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y + 1,P1Y + dYa + 1)
            itbuffer[:,0] = (slope * (itbuffer[:,1] - P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32) / dX.astype(np.float32)
            if negX:
                itbuffer[:,0] = np.arange(P1X - 1,P1X - dXa - 1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X + 1,P1X + dXa + 1)
            itbuffer[:,1] = (slope * (itbuffer[:,0] - P1X)).astype(np.int) + P1Y
 
    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]
 
    return itbuffer.astype(np.int)

class SegmentationWidget(Widget):
    
    def __init__(self, *args, **kwargs):
        #init the firsttouch and call the super
        super(SegmentationWidget, self).__init__(*args, **kwargs)

        self.firstTouch = True
        self.isKeep = True

        self.bind(size=self._update_rect, pos=self._update_rect)

        #keyboard stuff
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

        imgName = sys.argv[1]
        img = cv2.imread(imgName).astype(np.uint8)
        img = np.flip(img,axis=0)
        #align image

        self.imgWidth = img.shape[1]
        self.imgHeight = img.shape[0]
        self.flatShape = (self.imgHeight,self.imgWidth,1)
        self.flatestShape = (self.imgHeight,self.imgWidth)

        #attach the image to a texture
        self.texture = Texture.create(size=(self.imgWidth , self.imgHeight), colorfmt="bgr")
        self.texture.blit_buffer(img.tostring(), bufferfmt="ubyte", colorfmt="bgr")

        #create the copy
        
        self.imgOut = img
        self.img = img
        self.intensity = np.average(img,axis=2).astype(np.uint8)

        img = np.flip(img,axis=0)
        cv2.imwrite('out2.jpg', img)

        #create the graph and do initialization
        self.g = maxflow.Graph[int]()
        self.nodes = self.g.add_grid_nodes(self.flatShape)
        self.structure = np.array([[0, 1, 0],[1, 0, 1],[0, 1, 0]])


        self.f = np.zeros(self.flatShape)
        self.b = np.full(self.flatShape,np.inf,dtype=np.float)

        self.wf = np.zeros(self.flatShape)
        self.wb = np.zeros(self.flatShape)

        with self.canvas.before:
            #attach the texture to the app
            self.rect = Rectangle(texture=self.texture, size=self.size, pos=self.pos)

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
        if self.firstTouch:
            
            ys = np.sort([self.yResize(self.Ay),self.yResize(self.Cy)]).astype(np.int)
            xs = np.sort([self.xResize(self.Ax),self.xResize(self.Cx)]).astype(np.int)

            self.f[ys[0]:ys[1],xs[0]:xs[1]] = 1.
            self.b[ys[0]:ys[1],xs[0]:xs[1]] = 0.

            #change the draw type
            self.firstTouch = False
        else:
            points = touch.ud['line'].points
            for i in range(1,len(points) // 2):
                x1 = self.xResize(points[2 * (i - 1)])
                y1 = self.yResize(points[2 * (i - 1) + 1])
                x2 = self.xResize(points[2 * i + 0])
                y2 = self.yResize(points[2 * i + 1])

                buffer = createLineIterator(np.array([x1,y1]).astype(np.int),np.array([x2,y2]).astype(np.int),self.imgOut)
                for data in buffer:

                    '''
                    The default draw is Keep. Press s to change it
                    '''
                    if self.isKeep:
                        self.f[data[1]][data[0]] = np.inf
                        self.b[data[1]][data[0]] = 0.

                    else:
                        self.f[data[1]][data[0]] = 0.
                        self.b[data[1]][data[0]] = np.inf

        
        #do all the graph stuff
        segments = self.cut()

        #remove the lines and update the graph based on this line
        self.canvas.clear()

        #create the new image and display it
        self.imgOut = (self.img * segments.astype(int)).astype(np.uint8)

        self.texture.blit_buffer(self.imgOut.tostring(), bufferfmt="ubyte", colorfmt="bgr")

        self.imgOut = np.flip(self.imgOut,axis=0)
        cv2.imwrite('out2.jpg', self.imgOut)
        self.imgOut = np.flip(self.imgOut,axis=0)

        self.f = np.flip(self.f,axis=0)
        cv2.imwrite('fWeights2.jpg', self.f)
        self.f = np.flip(self.f,axis=0)

        self.b = np.flip(self.b, axis=0)
        cv2.imwrite('bWeights2.jpg', self.b)
        self.b = np.flip(self.b,axis=0)

        self.w = np.flip(self.w, axis=0)
        cv2.imwrite('nWeights2.jpg', self.w)
        self.w = np.flip(self.w,axis=0)

    '''
    All the graph work goes here
    '''
    def cut(self):
        fHist = cv2.calcHist([self.intensity],[0],self.f.astype(np.uint8),[256],[0,256])
        bHist = cv2.calcHist([self.intensity],[0],self.b.astype(np.uint8),[256],[0,256])

        #removed logs for sanity
        # #-lamb *log()
        self.wf = bHist[self.intensity]
        self.wb =  fHist[self.intensity]

        diff = np.abs(np.gradient(self.intensity))
        diff = np.average(diff, axis = 0)
        self.w = np.exp(-np.square(diff) / (2 * np.square(sigma)))

        #add the weights to the grid
        self.g.add_grid_edges(self.nodes, weights=1, structure=self.structure,symmetric=False)

        # Add the terminal edges. The [2] are the capacities
        # of the edges from the source node. The [3]
        # are the capacities of the edges to the sink node.
        self.g.add_grid_tedges(self.nodes, self.b, self.f)

        self.g.maxflow()

        return self.g.get_grid_segments(self.nodes)


    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 's':
            self.isKeep = not self.isKeep

        return True

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def xResize(self,x):
        return x / self.size[0] * self.imgWidth

    def yResize(self,y):
        return y / self.size[1] * self.imgHeight

    def _update_rect(self, instance, value):
            #resizing app
            self.rect.pos = instance.pos
            self.rect.size = instance.size


class SegmentationApp(App):

    def build(self):

        #load the image and pass it to the widget to render
        self.root = root = SegmentationWidget()
    
        return root

if __name__ == "__main__":
    SegmentationApp().run()