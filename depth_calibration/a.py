import numpy as np
import cv2
import sys
sys.path.append('/home/mcube/silhouettes/')
import depth_calibration.depth_helper 

paths = ['ball_D6.35', 'ball_D28.5', 'hollowcone', 'semicone', 'semipyramid']
img_num = 125
for path in paths:
	dir = '/media/mcube/data/shapes_data/processed/' + path + '/gradient/'
	gx = np.load(dir + 'gx_{}.npy'.format(img_num))
	gy = np.load(dir + 'gy_{}.npy'.format(img_num))
	height = depth_calibration.depth_helper.poisson_reconstruct(gy, gx)
	print path
	print 'Min: ', np.amin(height)
	print 'Max: ', np.amax(height)
