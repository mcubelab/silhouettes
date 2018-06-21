import numpy as np
import sys, os
import cv2, math, scipy.io
from PIL import Image
from scipy.misc import toimage
from depth_helper import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import yaml

size = (600, 600)
center_px = (300, 200)
angle = np.radians(20)
slope = np.tan(np.radians(15))
C_px = 20*12.5


C_mm = C_px/12.5
c_mm = 15

H_mm = C_mm*np.tan(slope)/2
h_mm = c_mm*np.tan(slope)/2

dz_dx_mat = np.zeros(size)
dz_dy_mat = np.zeros(size)
xvalues = np.array(range(size[0]))
yvalues = np.array(range(size[1]))
x_pixel, y_pixel = np.meshgrid(xvalues, yvalues)

x = (x_pixel - center_px[0]).astype(np.float32)
y = (y_pixel - center_px[1]).astype(np.float32)

x = x*np.cos(angle) - y*np.sin(angle)
y = y*np.cos(angle) + x*np.sin(angle)


z = slope*np.maximum(0, np.minimum(np.minimum(C_px-x-y, C_px-x+y), np.minimum(C_px+x-y, C_px+x+y)))
z = (z/np.amax(z))*H_mm
z = z*(z < (H_mm - h_mm))+(H_mm - h_mm)*(z >= (H_mm - h_mm))

gx, gy = np.gradient(z)

zz = poisson_reconstruct(gx, gy)



# print y
# print "#####"
# print xx
# print yy

# print "#####"
#
# print rot(x_dif, y_dif)[0]
# print rot(x_dif, y_dif)[1]
