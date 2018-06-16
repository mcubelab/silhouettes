import numpy as np
import sys, os
import cv2, math, scipy.io
from PIL import Image
from scipy.misc import toimage
from grad_to_depth import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
SHAPES_ROOT = os.getcwd().split("/silhouettes/")[0] + "/silhouettes/"

class Labeller():
    def __init__(self):
        params_dict = yaml.load(open(SHAPES_ROOT + 'resources/params.yaml'))
        self.mm2px_param_list = params_dict['params_gs2']

        input_shape = params_dict['input_shape_gs2'][0:2]
        print input_shape
        print type(input_shape)
        self.__compute_xy_px(input_shape)

    def __compute_xy_px(self, size):
        # We precompute x_pixel, y_pixel matrices (faster later)
        dz_dx_mat = np.zeros(size)
        dz_dy_mat = np.zeros(size)
        xvalues = np.array(range(size[0]))
        yvalues = np.array(range(size[1]))
        self.x_pixel, self.y_pixel = np.meshgrid(xvalues, yvalues)

    def __distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return np.linalg.norm((x1-x2, y1-y2))

    def __pos_px_to_mm(self, point):
        (x, y) = point
        k1, k2, k3,   l1, l2, l3,   dx, dy, dz = self.mm2px_param_list
        p1 = (x, y - 640.0/2)
        p2 = (p1[0]*k1 + p1[1]*k2 + k3*p1[0]*p1[1],   p1[1]*l1 + p1[0]*l2 + l3*p1[1]*p1[0])
        #p3 = (normal*(Dx + dx), p2[1] + dy, Dz + dz + p2[0])
        return p2

    def __radius_px_to_mm(self, center_px, radius_mm):
        p1_px = (center_px[0] + radius_mm, center_px[1])
        p2_px = (center_px[0] - radius_mm, center_px[1])
        p3_px = (center_px[0], center_px[1] + radius_mm)
        p4_px = (center_px[0], center_px[1] - radius_mm)

        center_mm = self.__pos_px_to_mm(center_px)
        p1_mm = self.__pos_px_to_mm(p1_px)
        p2_mm = self.__pos_px_to_mm(p2_px)
        p3_mm = self.__pos_px_to_mm(p3_px)
        p4_mm = self.__pos_px_to_mm(p4_px)

        d1 = np.linalg.norm(np.subtract(center_mm, p1_mm))
        d2 = np.linalg.norm(np.subtract(center_mm, p2_mm))
        d3 = np.linalg.norm(np.subtract(center_mm, p3_mm))
        d4 = np.linalg.norm(np.subtract(center_mm, p4_mm))

        # print d1, d2, d3, d4
        return float(d1+d2+d3+d4)/4.

    def __get_sphere_gradient(self, center_px, radius_px, R_mm):
        # # First we ajust the origin of the image to the top center point
        # center_mm = (self.__x_px_to_mm(center_px[0], center_px[1]), self.__y_px_to_mm(center_px[0], center_px[1]))
        # radius_mm =  radius_px/self.mm2px_average
        #
        # # We convert the values of each cell to mm
        # x_mm = (self.__x_px_to_mm(self.x_pixel, self.y_pixel)).astype(np.float32)
        # y_mm = (self.__y_px_to_mm(self.x_pixel, self.y_pixel)).astype(np.float32)
        #
        # x_dif = (x_mm - center_mm[0]).astype(np.float32)
        # y_dif = (y_mm - center_mm[1]).astype(np.float32)
        #
        # dz_dx_mat = -x_dif/(np.sqrt(np.abs(self.sphere_r_mm**2 - x_dif**2 - y_dif**2)))
        # dz_dy_mat = -y_dif/(np.sqrt(np.abs(self.sphere_r_mm**2 - x_dif**2 - y_dif**2)))
        #
        # mask = ((x_dif**2 + y_dif**2) < min(self.sphere_r_mm, radius_mm)**2).astype(np.float32)
        # dz_dx_mat = (dz_dx_mat * mask).astype(np.float32)
        # dz_dy_mat = (dz_dy_mat * mask).astype(np.float32)
        r_mm = self.__radius_px_to_mm(center_px, radius_px)
        R_mm = float(R_mm)
        if R_mm < r_mm:
            return None, None
        r_px = radius_px
        R_px = R_mm*r_px/r_mm  # TODO: Think if we can do this better

        h_mm = R_mm - np.sqrt(R_mm**2 - r_mm**2)
        print 'mm: ', R_mm, r_mm
        print 'px: ', R_px, r_px
        print 'h_mm: ', h_mm
        x_dif = (self.x_pixel - center_px[0]).astype(np.float32)
        y_dif = (self.y_pixel - center_px[1]).astype(np.float32)

        dz_dx_mat = -x_dif/(np.sqrt(np.abs(R_px**2 - x_dif**2 - y_dif**2)))
        dz_dy_mat = -y_dif/(np.sqrt(np.abs(R_px**2 - x_dif**2 - y_dif**2)))

        mask = ((x_dif**2 + y_dif**2) < min(R_px, r_px)**2).astype(np.float32)
        dz_dx_mat = (dz_dx_mat * mask).astype(np.float32)
        dz_dy_mat = (dz_dy_mat * mask).astype(np.float32)

        h_px = np.amax(poisson_reconstruct(dz_dy_mat, dz_dx_mat))
        h_px2mm = h_mm/h_px

        if h_px == 0 or h_mm < 0.1:
            return None, None
        return dz_dx_mat*h_px2mm, dz_dy_mat*h_px2mm

    def __get_semicone_1_gradient(self, center_px, radius_px):
        # Shape params
        r_mm = 10./2 # Smaller radius object
        cone_slope=np.tan(np.radians(20))
        R_mm = self.__radius_px_to_mm(center_px, radius_px) # Detected radius # NOTE: center_px and radius_px need to come from warped image

        if R_mm <= r_mm:
            return None, None

        # In pixels
        R_px = radius_px
        r_px = r_mm*R_px/R_mm  # TODO: Think a way of improving this


        h_mm = (R_mm - r_mm)*cone_slope
        '''
        print 'mm: ', R_mm, r_mm
        print 'px: ', R_px, r_px
        print 'h_mm: ', h_mm
        '''

        x_dif = (self.x_pixel - center_px[0]).astype(np.float32)
        y_dif = (self.y_pixel - center_px[1]).astype(np.float32)

        _EPS = 1e-8

        dz_dx_mat = -cone_slope*x_dif/(_EPS + np.sqrt(np.abs(x_dif**2 + y_dif**2)))
        dz_dy_mat = -cone_slope*y_dif/(_EPS + np.sqrt(np.abs(x_dif**2 + y_dif**2)))


        mask1 = ((x_dif**2 + y_dif**2) > r_px**2).astype(np.float32)
        mask2 = ((x_dif**2 + y_dif**2) < R_px**2).astype(np.float32)
        mask = (mask1 * mask2)

        # cv2.imshow('grad_x',dz_dx_mat)
        # cv2.imshow('grad_y',dz_dy_mat)
        # cv2.waitKey(0)

        # cv2.imshow('m1', mask1)
        # cv2.imshow('m2', mask2)
        # cv2.imshow('m', mask)
        # cv2.waitKey(0)

        dz_dx_mat = (dz_dx_mat * mask).astype(np.float32)
        dz_dy_mat = (dz_dy_mat * mask).astype(np.float32)

        dz_dx_mat = np.nan_to_num(dz_dx_mat)
        dz_dy_mat = np.nan_to_num(dz_dy_mat)


        h_px = np.amax(poisson_reconstruct(dz_dy_mat, dz_dx_mat)) #TODO: keep in mind y, x   vs. x, y
        h_px2mm = h_mm/h_px

        if h_px == 0 or h_mm < 0.1:
            return None, None

        return dz_dx_mat*h_px2mm, dz_dy_mat*h_px2mm

    def get_semicone_2_gradient(self, center_px, radius_px, cone_slope=np.tan(np.radians(10))):
        RR_mm = self.__radius_px_to_mm(center_px, radius_px)
        R_mm = 18.43/2
        r_mm = 10./2
        print RR_mm
        print R_mm
        print r_mm
        if r_mm > R_mm or R_mm > RR_mm:
            return None, None

        RR_px = radius_px
        R_px = R_mm*RR_px/RR_mm  # TODO: Think a way of improving this
        r_px = r_mm*RR_px/RR_mm  # TODO: Think a way of improving this

        EPS = 1e-5
        h_mm = (RR_mm - R_mm)*cone_slope

        x_dif = (self.x_pixel - center_px[0]).astype(np.float32)
        y_dif = (self.y_pixel - center_px[1]).astype(np.float32)

        dz_dx_mat = -cone_slope*x_dif/(_EPS + np.sqrt(np.abs(x_dif**2 + y_dif**2)))
        dz_dy_mat = -cone_slope*y_dif/(_EPS + np.sqrt(np.abs(x_dif**2 + y_dif**2)))

        mask1 = ((x_dif**2 + y_dif**2) > R_px**2).astype(np.float32)
        mask2 = ((x_dif**2 + y_dif**2) < RR_px**2).astype(np.float32)
        mask = (mask1 * mask2)

        dz_dx_mat = (dz_dx_mat * mask).astype(np.float32)
        dz_dy_mat = (dz_dy_mat * mask).astype(np.float32)

        # We add the whole gradient
        cone_slope = np.tan(np.radians(85))
        dx_p = cone_slope*x_dif/(_EPS + np.sqrt(np.abs(x_dif**2 + y_dif**2)))
        dy_p = cone_slope*y_dif/(_EPS + np.sqrt(np.abs(x_dif**2 + y_dif**2)))
        mask1_p = ((x_dif**2 + y_dif**2) > (r_px-1)**2).astype(np.float32)
        mask2_p = ((x_dif**2 + y_dif**2) < (r_px+1)**2).astype(np.float32)
        mask_p = (mask1_p * mask2_p)

        dx_p = (dx_p * mask_p).astype(np.float32)
        dy_p = (dy_p * mask_p).astype(np.float32)

        dz_dx_mat = np.nan_to_num(dz_dx_mat) + np.nan_to_num(dx_p)
        dz_dy_mat = np.nan_to_num(dz_dy_mat) + np.nan_to_num(dy_p)

        mask_pp = mask2 * mask1_p
        dz_dx_mat = mask_pp * dz_dx_mat
        dz_dy_mat = mask_pp * dz_dy_mat

        h_px = np.amax(poisson_reconstruct(dz_dy_mat, dz_dx_mat))
        h_px2mm = h_mm/h_px
        if h_px == 0 or h_mm < 0.1:
            return None, None
        return dz_dx_mat*h_px2mm, dz_dy_mat*h_px2mm

    def get_gradient_matrices(self, center_px, radius_px, shape='sphere', sphere_R_mm=28.5/2):
        # Everyhting given in pixel space
        if shape == 'sphere':
            gx, gy = self.__get_sphere_gradient(center_px, radius_px, R_mm=sphere_R_mm)
            return gx, gy
        if shape == 'semicone_1':
            gx, gy = self.__get_semicone_1_gradient(center_px, radius_px)
            return gx, gy
        if shape == 'semicone_2':
            gx, gy = self.get_semicone_2_gradient(center_px, radius_px)
            return gx, gy
        return None


if __name__ == "__main__":
    labeller = Labeller2()
    x, y = labeller.get_gradient_matrices(center_px=(200, 300), radius_px=90, shape='semicone_1')
    # print x
    # print y

    cv2.imshow('gx', x)
    cv2.imshow('gy', y)
    cv2.waitKey(0)

    depth_map = poisson_reconstruct(y, x)
    print depth_map.shape
    print "Max: " + str(np.amax(depth_map))

    def plot(depth_map):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.arange(depth_map.shape[0], step=1)
        Y = np.arange(depth_map.shape[1], step=1)
        X, Y = np.meshgrid(X, Y)
        surf = ax.plot_surface(X, Y, np.transpose(depth_map), rstride=1, cstride=1, cmap=cm.BuPu, linewidth=0, antialiased=False)
        ax.set_zlim(0, 5)
        ax.view_init(elev=90., azim=0)
        #ax.axes().set_aspect('equal')
        # plt.savefig(path + "img_" + str(img_number) + "_semicone_obj_weights.png")
        plt.show()
    depth_map = cv2.resize(depth_map, dsize=(50, 83), interpolation=cv2.INTER_LINEAR)
    plot(depth_map)