'''
https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
'''

import cv2
import numpy as np
import glob
import os

path = os.path.join('.', 'faces', 'output', '*.png')
img_array = []
for filename in glob.glob(path):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()