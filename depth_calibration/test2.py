from labeller import *
from depth_helper import *
import cv2

center_px = (200, 200)
radius_px = 100

l = Labeller()
hm = l.get_testtool1_gradient(center_px, radius_px)

hm = cv2.resize(hm, dsize=(99*2, 96*2), interpolation=cv2.INTER_LINEAR)

point_list = [
    (38, 40),
    (38, 45),
    (38, 50),
    (38, 54)
]

for point in point_list:
    print str(point) + ": " + str(hm[point[0]][point[1]])


plot_depth_map(hm)
