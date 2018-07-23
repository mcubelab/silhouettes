import math, cv2, os, pickle, scipy.io, pypcd, subprocess, sys
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import math, cv2, os, pickle
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
SHAPES_ROOT = os.getcwd().split("/silhouettes/")[0] + "/silhouettes/"
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from world_positioning import pxb_2_wb_3d
from depth_calibration.depth_helper import *
import scipy
import open3d

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.draw_geometries([source_temp, target_temp])


class Location():
    def __init__(self):
        self.compress_factor = .2
        self.params_dict = yaml.load(open(SHAPES_ROOT + 'resources/params.yaml'))
        self.params_gs2 = self.params_dict['params_gs2']

    def old_visualize_pointcloud(self, pointcloud):
        
        pointcloud = {'x': [], 'y': [], 'z': []}
        a = np.random.permutation(len(np_pointcloud))
        max_points = 5000
        a = np.sort(a[0:min(max_points, len(np_pointcloud))])
        for i in a:
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

    def old_visualize2_pointclouds(self, pcs):
        pc1 = pcs[0]
        pc2 = pcs[1]
        pointcloud = {'x': [], 'y': [], 'z': []}
        a = np.random.permutation(len(pc1))
        max_points = 5000
        a =np.sort(a[0:min(max_points, len(pc1))])
        for i in a:
            pointcloud['x'].append(pc1[i][0])
            pointcloud['y'].append(pc1[i][1])
            pointcloud['z'].append(pc1[i][2])
        mean_y = np.mean(pointcloud['y'])
        std_y = np.std(pointcloud['y'])
        vmin = mean_y - 4*std_y
        vmax = mean_y + 4*std_y
        ax.scatter3D(pointcloud['x'], pointcloud['y'], pointcloud['z'],
            c=pointcloud['y'], cmap='Blues', s=3, vmin = vmin, vmax=vmax)
        #surf = ax.plot_trisurf(pointcloud['x'], pointcloud['y'], pointcloud['z'], cmap='Blues', linewidth=0)
        pointcloud = {'x': [], 'y': [], 'z': []}
        a = np.random.permutation(len(pc2))
        max_points = 5000
        a = np.sort(a[0:min(max_points, len(pc2))])
        for i in a:
            pointcloud['x'].append(pc2[i][0])
            pointcloud['y'].append(pc2[i][1])
            pointcloud['z'].append(pc2[i][2])
        mean_y = np.mean(pointcloud['y'])
        std_y = np.std(pointcloud['y'])
        vmin = mean_y - 4*std_y
        vmax = mean_y + 2*std_y
        ax.scatter3D(pointcloud['x'], pointcloud['y'], pointcloud['z'],
            c=pointcloud['y'], cmap='Reds', s=6, vmin = vmin, vmax=vmax)

        #surf = ax.plot_trisurf(pointcloud['x'], pointcloud['y'], pointcloud['z'], cmap='Reds', linewidth=0)
        '''
        from scipy.interpolate import griddata
        X, Y = np.meshgrid(pointcloud['x'], pointcloud['z'])
        Z = griddata((pointcloud['x'], pointcloud['z']), pointcloud['y'], (X, Y), method='linear')
        ax.plot_surface(X,Z,Y, cmap=cm.jet)
        '''
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
        
    def check_pointcloud_type(self, pointcloud):
        if isinstance(pointcloud, np.ndarray):
            pcd = open3d.PointCloud()
            pcd.points = open3d.Vector3dVector(pointcloud)
            pointcloud = pcd
        return pointcloud
        
    def visualize_pointcloud(self, pointcloud, color = None):
        pcd = self.check_pointcloud_type(pointcloud)
        if color is not None:
            pcd.paint_uniform_color(color)
        open3d.draw_geometries([pcd])

    def visualize2_pointclouds(self, pc_list, color_list):
        for i, pc in enumerate(pc_list):
            pc = self.check_pointcloud_type(pc)
            if color_list is not None:
                pc.paint_uniform_color(color_list[i])
        open3d.draw_geometries(pc_list)

    def get_contact_info(self, directory, num, only_cart=False):
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

        if only_cart:
            directory += '/p_' + str(num)
            cart = get_cart(directory + '/cart.npy')
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

    def get_local_pointcloud(self, gs_id, directory='', num=-1, new_cart=None):
        # 1. We convert the raw image to height_map data
        cart, gs1_list, gs2_list, wsg_list, force_list, gs1_back, gs2_back = self.get_contact_info(directory, num) #TODOM: why so many info?
        if new_cart is not None:
            cart = new_cart

        if gs_id == 1:
            pass # TODO: GET height_map
        elif gs_id == 2:
            params_gs = self.params_gs2
            size = self.params_dict['input_shape_gs2'][0:2]


            test_image = gs2_list[0]
            # test_image2 = cv2.imread("GS2_" + str(1) + '.png')

            # cv2.imshow('test_image', test_image)
            # cv2.imshow('test_image2', test_image2)


            # for it in range(3):
            #     print np.mean(test_image[:,:,it])
            #     print np.mean(test_image2[:,:,it])
            #     test_image[:,:,it] = test_image[:,:,it]/np.mean(test_image[:,:,it])*np.mean(test_image2[:,:,it])
            #     print np.mean(test_image[:,:,it])

            height_map = raw_gs_to_depth_map(
                test_image=test_image,
                ref=None,
                model_path=SHAPES_ROOT + 'depth_calibration/weights/weights.aug.v1.hdf5',
                plot=False,
                save=False,
                path='')

            # cv2.imshow('hm', height_map)
            # cv2.waitKey(0)

            mean = np.mean(height_map)
            dev = np.std(height_map)
            a = mean - dev
            height_map = (height_map > 0.1)*height_map*10  # NOTE: We are multiplying the height by 10 for visualization porpuses!!!!!
            height_map = (height_map > a)*height_map
            #height_map = height_map+0.05
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
                if(height_map[i][j] >= 0.10):
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

    def stitch_pointclouds_open3D(self, source, target, threshold = 1, trans_init = np.eye(4), max_iteration = 2000, with_plot = True):
        source = self.check_pointcloud_type(source)
        target = self.check_pointcloud_type(target)
        target.paint_uniform_color([0.1, 0.1, 0.7])
        print dir(open3d)
        if with_plot:
            draw_registration_result(source, target, trans_init)
            
        print("Initial alignment")
        evaluation = open3d.evaluate_registration(source, target, threshold, trans_init)
        print(evaluation)

        print("Apply point-to-point ICP")
        reg_p2p = open3d.registration_icp(source, target, threshold, trans_init,
                open3d.TransformationEstimationPointToPoint(),
                open3d.ICPConvergenceCriteria(max_iteration = max_iteration))
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        print("")
        if with_plot:
            draw_registration_result(source, target, reg_p2p.transformation)

    def stitch_pointclouds(self, fixed, moved):
        print 'stitching'
        fixed = np.asarray(fixed)
        moved = np.asarray(moved)

        # 1. We save the pointclouds in .pcd format
        print 'saving PointClouds'
        name = 'c++/cloud' + str(0) + '.pcd'
        with open(name, 'w') as f:
            f.write(self.__get_string_pc(fixed))

        name = 'c++/cloud' + str(1) + '.pcd'
        with open(name, 'w') as f:
            f.write(self.__get_string_pc(moved))

        # 2. We run the c++ program to stitch them
        print 'command sent'
        command = 'cd c++/; ./pairwise_incremental_registration cloud[0-1].pcd'
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()

        # 3. We load and return the merged pointcloud
        print 'c++ part done'
        path = 'c++/1.pcd'
        pc = pypcd.PointCloud.from_path(path)
        # pointcloud = npPC2dictPC(pc.pc_data)
        print type(pc)
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
            try:
                local_pointcloud = loc.get_local_pointcloud(
                    gs_id=gs_id,
                    directory=directory,
                    num=i)

                if global_pointcloud is None:
                    global_pointcloud = local_pointcloud
                else:
                    # local_pointcloud = self.translate_pointcloud(local_pointcloud, v=(0, 5*i, 0))
                    global_pointcloud = self.simple_pointcloud_merge(global_pointcloud, local_pointcloud)
                    # merged = self.stitch_pointclouds(local_pointcloud_0, local_pointcloud_1)
            except Exception as e:
                print "Error computing local PointCloud"
                print e
        return global_pointcloud

    def get_distance_images(self, img1, img2):

        model = ResNet50(weights='imagenet', include_top=False)

        x = image.img_to_array(img1)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        x2 = image.img_to_array(img2)
        x2 = np.expand_dims(x2, axis=0)
        x2 = preprocess_input(x2)

        features = model.predict(x).flatten()
        features2 = model.predict(x2).flatten()
        return scipy.spatial.distance.cosine(features, features2)

if __name__ == "__main__":
    name = 'only_front.npy'
    name = 'pointcloud_bar.npy'
    loc = Location()

    touch_list = range(3, 6)
    touch_list += range(7, 50)
    
    touch_list_aux = copy.deepcopy(touch_list)
    #touch_list = range(3, 7)
    #touch_list += range(7, 13)
    # touch_list = [0, 5, 23]
    '''
    global_pointcloud = loc.get_global_pointcloud(
         gs_id=2,
         directory='/media/mcube/data/shapes_data/pos_calib/bar_front/',
         touches=touch_list,
         global_pointcloud = None
     )
    global_pointcloud = np.array(global_pointcloud)
    np.save('/media/mcube/data/shapes_data/pointclouds/' + name, global_pointcloud)
    '''
    global_pointcloud = np.load('/media/mcube/data/shapes_data/pointclouds/' + name)
    
    # loc.visualize_pointcloud(global_pointcloud)

    # missing = loc.get_local_pointcloud(
    #     gs_id=2,
    #     directory='/media/mcube/data/shapes_data/pos_calib/bar_front/',
    #     num=16
    # )
    # loc.visualize_pointcloud(missing)
    from skimage.measure import compare_ssim as ssim
    imageA = cv2.imread('/media/mcube/data/shapes_data/pos_calib/full_bar_v2_front/p_20/GS2_0.png')
    #touch_list = touch_list_aux
    touch_list = range(16, 21)
    touch_list += range(21, 50)
    for num in touch_list[1:10]:
        directory = '/media/mcube/data/shapes_data/pos_calib/bar_front/'
        cart = loc.get_contact_info(directory, num, only_cart=True)
        print cart
        missing = loc.get_local_pointcloud(
            gs_id=2,
            directory=directory,
            num=6,
            new_cart=cart
        )
        #np.save('/media/mcube/data/shapes_data/pointclouds/' + 'pcd_6_front_bar.npy', global_pointcloud)
        missing = np.array(missing)
        loc.visualize_pointcloud(global_pointcloud)
        
        #new_pointcloud = loc.stitch_pointclouds(global_pointcloud, missing)
        trans_init = np.eye(4)
        trans_init[0,-1] = 105
        trans_init[1,-1] = 0
        new_pointcloud = loc.stitch_pointclouds_open3D(global_pointcloud, missing, trans_init = trans_init, threshold = 0.5)
        print 'num: ', num
        imageB = cv2.imread('/media/mcube/data/shapes_data/pos_calib/full_bar_front/p_{}/GS2_0.png'.format(num))
        print 'ssim: ', ssim(imageA, imageB, multichannel=True)
        print 'cosine distance', loc.get_distance_images(imageA, imageB)
        imageB = cv2.imread('/media/mcube/data/shapes_data/pos_calib/full_bar_v2_front/p_{}/GS2_0.png'.format(num))
        #print 'ssim: ', ssim(imageA, imageB, multichannel=True)
        print 'cosine distance v2 ', loc.get_distance_images(imageA, imageB)
        imageB = cv2.imread('/media/mcube/data/shapes_data/pos_calib/full_bar_f=20_v1_front/p_{}/GS2_0.png'.format(num))
        #print 'ssim: ', ssim(imageA, imageB, multichannel=True)
        print 'cosine distance f=20 ', loc.get_distance_images(imageA, imageB)

        loc.visualize2_pointclouds([new_pointcloud, missing])

    # touch_list = range(1, 17)
    # global_pointcloud = loc.get_global_pointcloud(
    #     gs_id=2,
    #     directory='/media/mcube/data/shapes_data/pos_calib/bar_side/',
    #     touches=touch_list,
    #     global_pointcloud = global_pointcloud
    # )
    #
    # touch_list = range(1, 50)
    # global_pointcloud = loc.get_global_pointcloud(
    #     gs_id=2,
    #     directory='/media/mcube/data/shapes_data/pos_calib/bar_back/',
    #     touches=touch_list,
    #     global_pointcloud = global_pointcloud
    # )

    #np.save('/media/mcube/data/shapes_data/pointclouds/' + name, global_pointcloud)
    #loc.visualize2_pointclouds([global_pointcloud, missing])
