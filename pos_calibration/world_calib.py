import os, pickle, sys
import cv2, math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
SHAPES_ROOT = os.getcwd().split("/silhouettes/")[0] + "/silhouettes/"
from location.world_positioning import pxb2grb, wb2grb, grb2wb


class PosCalib():
    def __init__(self):
        self.ini_x = [0, 0, 0,  0, 0, 0,  0, 0, 0]
        self.ini_x = [0.08306509459312332, -0.0006996618309678892, 1.5431217937797345e-06, 0.08153356084518493, 0.005461568194849241, -1.5973694835128038e-06, 2.0757556565435884, 7.694306015560224, -13.986576864912136]

    def __pos_after_warp(self, point, size, warp_matrix_path):
        a = np.zeros(size)
        a[int(point[0])][int(point[1])] = 1
        M = np.load(warp_matrix_path)
        rows,cols = a.shape
        a_p = cv2.warpPerspective(a, M, (cols, rows))
        new_pos = np.where(a_p > 0)
        #print new_pos
        x = new_pos[0][0]
        y = new_pos[1][0]
        return (x, y)

    def get_px2mm_params(self, path_list, warp_matrix_path):
        self.mc = []
        n_points = []
        for path in path_list:
            info = np.load(path).tolist()
            self.mc = self.mc + info
            n_points.append(float(len(info)))

        # self.mc = self.mc[0:200]
        warped_points_dict = {}

        # Parameter evaluation before tunning
        error = 0
        for (gs_point, gs_id, gripper_state, real_point) in self.mc:
            gs_point_poswarp = self.__pos_after_warp( point=gs_point, size=(640, 640), warp_matrix_path=warp_matrix_path)
            warped_points_dict[gs_point] = gs_point_poswarp
            real_point = wb2grb(point=real_point, gripper_pos=gripper_state['pos'], quaternion=gripper_state['quaternion'])
            fx, fy, fz = real_point
            (fp_x, fp_y, fp_z) = pxb2grb(gs_point, gs_id, gripper_state, self.ini_x)
            # print (fp_x-fx, fp_y-fy, fp_z-fz)
            error += np.linalg.norm((fp_x-fx, fp_y-fy, fp_z-fz))
        print "Average distance before tunning: " + str(error/len(self.mc))

        # Tunning
        def eq_sys(params):
            dif = 0
            n = len(self.mc)
            for (gs_point, gs_id, gripper_state, real_point) in self.mc:
                # TODO: Compute weights
                weight = 1
                real_point = wb2grb(point=real_point, gripper_pos=gripper_state['pos'], quaternion=gripper_state['quaternion'])
                fx, fy, fz = real_point
                gs_point_poswarp = warped_points_dict[gs_point] # We take into account the future warping of the image
                (fp_x, fp_y, fp_z) = pxb2grb(gs_point_poswarp, gs_id, gripper_state, params)
                #(fp_x, fp_y, fp_z) = pxb2grb(gs_point, gs_id, gripper_state, params)

                dif += weight*np.linalg.norm((fp_x-fx, fp_y-fy, fp_z-fz))
            return dif/n  # This number is the average distance between estimations and real points

        # Solve optimization problem
        x0 = self.ini_x
        # res = minimize(eq_sys, x0, bounds=bounds, options={'xtol': 1e-8, 'disp': False})
        res = minimize(eq_sys, x0, options={'xtol': 1e-8, 'disp': True})
        print "Success: " + str(res.success)
        print "Average distance: " + str(res.fun)

        return res

    def test_all(self, params, warp_matrix_path):
        dif = 0
        n = float(len(self.mc))
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)

        for (point, gs_id, gripper_state, real_point) in self.mc:
            fx, fy, fz = real_point
            # print "Real in world: " + str((fx, fy, fz))
            real_point = wb2grb(point=real_point, gripper_pos=gripper_state['pos'], quaternion=gripper_state['quaternion'])
            fx, fy, fz = real_point
            point_poswarp = self.__pos_after_warp(point=point, size=(480, 640), warp_matrix_path=warp_matrix_path)
            #(fp_x, fp_y, fp_z) = pxb2grb(point_poswarp, gs_id, gripper_state, params)
            (fp_x, fp_y, fp_z) = pxb2grb(point, gs_id, gripper_state, params)
            print "Gripper state: " + str(gripper_state['pos'])
            print "Gripper state Dz: " + str(gripper_state['Dz'])
            print "Gripper state Dx: " + str(gripper_state['Dx'])
            print "Gripper state Quat: " + str(gripper_state['quaternion'])
            print "Point: " + str((point[1]-640.0/2, point[0]))
            print "Real: " + str((fx, fy, fz))
            print "Guessed: " + str((fp_x, fp_y, fp_z))
            print "Diference: " + str((fp_x-fx, fp_y-fy, fp_z-fz))
            print "Distance: " + str(np.linalg.norm((fp_x-fx, fp_y-fy, fp_z-fz)))
            print "*****"
            dif += np.linalg.norm((fp_x-fx, fp_y-fy, fp_z-fz))
            ax1.plot([fp_y, fy], [fp_z, fz], '-o')
            ax1.plot([fy, fy], [fz, fz] , 'ro')
            #ax1.set_xlim(ax1.get_xlim()[::-1])
            ax1.set_aspect('equal')

            real_point = grb2wb(point=real_point, gripper_pos=gripper_state['pos'], quaternion=gripper_state['quaternion'])
            # print real_point
            fx, fy, fz = real_point
            fp_x, fp_y, fp_z = grb2wb(point=(fp_x, fp_y, fp_z), gripper_pos=gripper_state['pos'], quaternion=gripper_state['quaternion'])
            ax2.plot([fp_x, fx], [fp_z, fz], '-o')
            ax2.plot([fx, fx], [fz, fz] , 'ro')
            ax2.set_aspect('equal')
            import matplotlib.image as mpimg
            img=mpimg.imread('pos_calibration/pos_calibration_squares_color/p_0/GS2_0.png')

            # Image warping
            M = np.load(warp_matrix_path)
            rows,cols,cha = img.shape
            img_warped = cv2.warpPerspective(img, M, (cols, rows))
            img_warped = img_warped[:,63:-80,:]
            ax3.imshow(img)
            #ax3.plot(point[1]- 640.0/2, point[0], 'ro')
            #ax3.plot(640-point[1], point[0], 'o')
            ax3.plot(point[1], point[0], 'o')
            ax3.set_xlim(ax3.get_xlim()[::-1])
            ax3.plot(point_poswarp[1], point_poswarp[0], 'bo')
            #ax3.plot(640-point_poswarp[1], point_poswarp[0], 'bo')
            #ax3.gca().invert_xaxis()
        plt.show()


if __name__ == "__main__":
    # file_list = ['pos_calibration/no_mirror/border.npy']
    # file_list = ['pos_calibration/no_mirror/squares.npy']
    file_list = ['pos_calibration/no_mirror/squares.npy', 'pos_calibration/no_mirror/border.npy', 'pos_calibration/no_mirror/line.npy']

    pc = PosCalib()
    cts = pc.get_px2mm_params(path_list=file_list, warp_matrix_path=SHAPES_ROOT + '/resources/GS2_M_color.npy')
    print cts.x.tolist()
    # pc.test_all(params=cts.x, warp_matrix_path=SHAPES_ROOT + '/resources/GS2_M_color.npy')
    print "Average distance: " + str(cts.x.fun)
    for elem in cts.x:
        print("%.4f" % float(elem))
