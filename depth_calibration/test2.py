from labeller import *
from depth_helper import *
import cv2

center_px = (200, 200)
radius_px = 100

l = Labeller()
gx, gy = l.get_testtool1_gradient(center_px, radius_px)

# cv2.imshow('x', gx)
# cv2.imshow('y', gy)
# cv2.waitKey(0)

hm = poisson_reconstruct(gy, gx)

hm = cv2.resize(hm, dsize=(99, 96), interpolation=cv2.INTER_LINEAR)

point_list = [
    (38, 40),
    (38, 45),
    (38, 50),
    (38, 54)
]

for point in point_list:
    print str(point) + ": " + str(hm[point[0]][point[1]])


plot_depth_map(hm)
