import math, cv2, os, pickle, scipy.io, pypcd, subprocess, sys
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import math, cv2, os, pickle
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
SHAPES_ROOT = os.getcwd().split("/silhouettes/")[0] + "/silhouettes/"
from world_positioning import pxb_2_wb_3d
from depth_calibration.depth_helper import *


class Location():
    def __init__(self):
        self.compress_factor = .2
        self.params_dict = yaml.load(open(SHAPES_ROOT + 'resources/params.yaml'))
        self.params_gs2 = self.params_dict['params_gs2']

    def visualize_pointcloud(self, np_pointcloud):
        pointcloud = {'x': [], 'y': [], 'z': []}
        for i in range(len(np_pointcloud)):
            pointcloud['x'].append(np_pointcloud[i][0])
            pointcloud['y'].append(np_pointcloud[i][1])
            pointcloud['z'].append(np_pointcloud[i][2])

        ax = plt.axes(projection='3d')
        ax.scatter3D(pointcloud['x'], pointcloud['y'], pointcloud['z'],
            c=pointcloud['y'], cmap='Blues')

        # Set viewpoint.
        ax.azim = -90
        ax.elev = 0

        # Label axes.
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')

        def axisEqual3D(ax):
            extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
            sz = extents[:, 1] - extents[:, 0]
            centers = np.mean(extents, axis=1)
            maxsize = max(abs(sz))
            r = maxsize/2
            for ctr, dim in zip(centers, 'xyz'):
                getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

        axisEqual3D(ax)
        plt.show()

    def get_contact_info(self, directory, num):
        def get_cart(path):
            cart = np.load(path)
            a = cart[3]
            cart[3] = cart[6]
            b = cart[4]
            cart[4] = a
            c = cart[5]
            cart[5] = b
            cart[6] = c
            return cart

        def load_obj(path):
            with open(path, 'rb') as f:
                return pickle.load(f)

        gs1_back = None
        gs2_back = None

        try:
            gs1_back = cv2.imread(directory + '/air/GS1.png')
        except:
            print 'GS1 backgorund not found'
        try:
            gs2_back = cv2.imread(directory + '/air/GS2.png')
        except:
            print 'GS2 backgorund not found'

        directory += '/p_' + str(num)
        file_list = os.listdir(directory)

        cart = get_cart(directory + '/cart.npy')
        gs1_list = []
        gs2_list = []
        wsg_list = []

        for elem in file_list:
            path = directory + '/' + elem
            if 'GS1' in elem:
                gs1_list.append(cv2.imread(path))
            elif 'GS2' in elem:
                gs2_list.append(cv2.imread(path))
            elif 'wsg' in elem:
                wsg_list.append(load_obj(path))

        force_list = []
        for elem in wsg_list:
            force_list.append(elem['force'])

        return cart, gs1_list, gs2_list, wsg_list, force_list, gs1_back, gs2_back

    def get_local_pointcloud(self, gs_id, directory='', num=-1):
        # 1. We convert the raw image to height_map data
        cart, gs1_list, gs2_list, wsg_list, force_list, gs1_back, gs2_back = self.get_contact_info(directory, num) #TODOM: why so many info?
        if gs_id == 1:
            pass # TODO: GET height_map
        elif gs_id == 2:
            params_gs = self.params_gs2
            size = self.params_dict['input_shape_gs2'][0:2]


            test_image = gs2_list[0]
            test_image2 = cv2.imread("GS2_" + str(1) + '.png')

            for it in range(3):
                print np.mean(test_image[:,:,it])
                print np.mean(test_image2[:,:,it])
                test_image[:,:,it] = test_image[:,:,it]/np.mean(test_image[:,:,it])*np.mean(test_image2[:,:,it])
                print np.mean(test_image[:,:,it])

            height_map = raw_gs_to_depth_map(
                test_image=test_image,
                ref=None,
                model_path=SHAPES_ROOT + 'depth_calibration/weights/weights.color_semicone1_and_sphere.xy.hdf5',
                plot=False,
                save=False,
                path='')
            mean = np.mean(height_map)
            dev = np.std(height_map)
            a = mean - dev
            height_map = (height_map > 0.1)*height_map*10  # NOTE: We are multiplying the height by 10 for visualization porpuses!!!!!
            height_map = (height_map > a)*height_map
            height_map = cv2.resize(height_map, dsize=(size[1], size[0]), interpolation=cv2.INTER_LINEAR) #TODOM: why?
            # cv2.imshow('a', height_map)
            # cv2.waitKey()

        # 2. We convert height_map data into world position
        gripper_state = {}
        gripper_state['pos'] = cart[0:3]
        gripper_state['quaternion'] = cart[-4:]
        gripper_state['Dx'] = wsg_list[0]['width']/2.0
        gripper_state['Dz'] = 139.8 + 72.5 + 160  # Base + wsg + finger

        pointcloud = []
        for i in range(height_map.shape[0]):
            for j in range(height_map.shape[1]):
                if(height_map[i][j] != 0):
                    world_point = pxb_2_wb_3d(
                        point_3d=(i, j, height_map[i][j]),
                        gs_id=gs_id,
                        gripper_state = gripper_state,
                        fitting_params = params_gs
                    )
                    a = np.asarray(world_point)
                    pointcloud.append(a)

        return pointcloud

    def simple_pointcloud_merge(self, pointcloud1, pointcloud2):
        print '******'
        print len(pointcloud1)
        print len(pointcloud2)
        merged = pointcloud1 + pointcloud2
        print len(merged)
        return merged

    def __get_string_pc(self, pointcloud):
        string = "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH " + str(pointcloud.shape[0]) + "\nHEIGHT 1\nVIEWPOINT 0.0 0.0 0.0 1.0 0.0 0.0 0.0\nPOINTS " + str(pointcloud.shape[0]) + "\nDATA ascii"
        for elem in pointcloud:
            string += '\n' + str(elem[0]) + ' ' + str(elem[1]) + ' ' + str(elem[2])
        return string

    def stitch_pointclouds(self, fixed, moved):
        fixed = np.asarray(fixed)
        moved = np.asarray(moved)

        # 1. We save the pointclouds in .pcd format
        name = 'c++/cloud' + str(0) + '.pcd'
        with open(name, 'w') as f:
            f.write(self.__get_string_pc(fixed))

        name = 'c++/cloud' + str(1) + '.pcd'
        with open(name, 'w') as f:
            f.write(self.__get_string_pc(moved))

        # 2. We run the c++ program to stitch them
        command = 'cd c++/; ./pairwise_incremental_registration cloud[0-1].pcd'
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()

        # 3. We load and return the merged pointcloud
        path = 'c++/1.pcd'
        pc = pypcd.PointCloud.from_path(path)
        # pointcloud = npPC2dictPC(pc.pc_data)
        print 'Stitched'
        return pc.pc_data

'''
TODOM: purpose of translate_pointcloud function? visualization? 
IDEA: it might be useful some tool that justs helps visualize all the localpointclouds without merging
'''

    def translate_pointcloud(self, pointcloud, v): 
        new_pc = []
        for elem in pointcloud:
            new_elem = (elem[0]+v[0], elem[1]+v[1], elem[2]+v[2])
            new_pc.append(new_elem)
        return new_pc

    def get_global_pointcloud(self, gs_id, directory, touches, global_pointcloud):
        for i in touches:
            exp = str(i)
            print "Processing img " + exp + "..."
            local_pointcloud = loc.get_local_pointcloud(
                gs_id=gs_id,
                directory=directory,
                num=i)

            if global_pointcloud is None:
                global_pointcloud = local_pointcloud
            else:
                # local_pointcloud = self.translate_pointcloud(local_pointcloud, v=(0, 5*i, 0))
                global_pointcloud = self.simple_pointcloud_merge(global_pointcloud, local_pointcloud)
        return global_pointcloud

if __name__ == "__main__":
    loc = Location()

    touch_list = [0, 5, 23, 25]
    # touch_list = [0, 5, 23]
    global_pointcloud = loc.get_global_pointcloud(
        gs_id=2,
        directory='sample/',
        touches=touch_list,
        global_pointcloud = None
    )
    print 'len: ' + str(len(global_pointcloud))
    # local_pointcloud_0 = loc.get_local_pointcloud(
    #     gs_id=2,
    #     directory='sample/',
    #     num=0)

    # loc.visualize_pointcloud(local_pointcloud_0)

    # local_pointcloud_1 = loc.get_local_pointcloud(
    #     gs_id=2,
    #     directory='sample/',
    #     num=1)
    # # loc.visualize_pointcloud(local_pointcloud_1)

    # merged = loc.simple_pointcloud_merge(local_pointcloud_0, local_pointcloud_1)
    # merged = loc.stitch_pointclouds(local_pointcloud_0, local_pointcloud_1)
    loc.visualize_pointcloud(global_pointcloud)
