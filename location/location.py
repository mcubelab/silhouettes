import math, cv2, os, pickle, scipy.io, pypcd, subprocess, sys
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import math, cv2, os, pickle
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
SHAPES_ROOT = os.getcwd().split("/silhouettes/")[0] + "/silhouettes/"
import time
import tf
import tf.transformations as tfm

try:
    from keras.models import Sequential
    from keras.layers import *
    from keras.models import model_from_json
    from keras import optimizers
    from keras.callbacks import ModelCheckpoint
    import keras.losses
    from keras.models import load_model
    from keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    from keras import backend as K
    from keras.applications.resnet50 import ResNet50
    from keras.preprocessing import image
    from keras.applications.resnet50 import preprocess_input, decode_predictions
except Exception as e:
    print "Not importing keras"

from world_positioning import pxb_2_wb_3d, quaternion_matrix
from depth_calibration.depth_helper import *
import scipy
import open3d
import time


global x_off, y_off, z_off

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

    def old_visualize_pointcloud(self, np_pointcloud):

        pointcloud = {'x': [], 'y': [], 'z': []}
        a = np.random.permutation(len(np_pointcloud))
        max_points = 1000
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

    def old_visualize3_pointclouds(self, pcs):
        pc1 = pcs[0]
        pc2 = pcs[1]
        pc3 = pcs[2]
        
        pointcloud = {'x': [], 'y': [], 'z': []}
        a = np.random.permutation(len(pc1))
        max_points = 400
        a = np.sort(a[0:min(max_points, len(pc1))])
        for i in a:
            pointcloud['x'].append(pc1[i][0])
            pointcloud['y'].append(pc1[i][1])
            pointcloud['z'].append(pc1[i][2])
        mean_y = np.mean(pointcloud['y'])
        std_y = np.std(pointcloud['y'])
        vmin = mean_y - 4*std_y
        vmax = mean_y + 4*std_y
        
        ax = plt.axes(projection='3d')
        ax.scatter3D(pointcloud['x'], pointcloud['y'], pointcloud['z'],
            c=pointcloud['y'], cmap='Blues', s=3, vmin = vmin, vmax=vmax)
        
        #surf = ax.plot_trisurf(pointcloud['x'], pointcloud['y'], pointcloud['z'], cmap='Blues', linewidth=0)
        pointcloud = {'x': [], 'y': [], 'z': []}
        a = np.random.permutation(len(pc2))
        max_points = 400
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
        
        
        pointcloud = {'x': [], 'y': [], 'z': []}
        a = np.random.permutation(len(pc3))
        max_points = 400
        a = np.sort(a[0:min(max_points, len(pc3))])
        for i in a:
            pointcloud['x'].append(pc3[i][0])
            pointcloud['y'].append(pc3[i][1])
            pointcloud['z'].append(pc3[i][2])
        mean_y = np.mean(pointcloud['y'])
        std_y = np.std(pointcloud['y'])
        vmin = mean_y - 4*std_y
        vmax = mean_y + 4*std_y
        
        ax.scatter3D(pointcloud['x'], pointcloud['y'], pointcloud['z'],
            c=pointcloud['y'], cmap='Greens', s=3, vmin = vmin, vmax=vmax)

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

    def old_visualize2_pointclouds(self, pcs):
        pc1 = pcs[0]
        pc2 = pcs[1]
        pointcloud = {'x': [], 'y': [], 'z': []}
        a = np.random.permutation(len(pc1))
        max_points = 500
        a =np.sort(a[0:min(max_points, len(pc1))])
        for i in a:
            pointcloud['x'].append(pc1[i][0])
            pointcloud['y'].append(pc1[i][1])
            pointcloud['z'].append(pc1[i][2])
        mean_y = np.mean(pointcloud['y'])
        std_y = np.std(pointcloud['y'])
        vmin = mean_y - 4*std_y
        vmax = mean_y + 4*std_y
        
        ax = plt.axes(projection='3d')
        ax.scatter3D(pointcloud['x'], pointcloud['y'], pointcloud['z'],
            c=pointcloud['y'], cmap='Blues', s=3, vmin = vmin, vmax=vmax)
        #surf = ax.plot_trisurf(pointcloud['x'], pointcloud['y'], pointcloud['z'], cmap='Blues', linewidth=0)
        pointcloud = {'x': [], 'y': [], 'z': []}
        a = np.random.permutation(len(pc2))
        max_points = 500
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

    def visualize_pointcloud(self, pointcloud, color = None, shape = None):
        pcd = self.check_pointcloud_type(pointcloud)
        if color is not None:
            pcd.paint_uniform_color(color)

        mesh = None
        if shape is not None:
            if 'semicone' in shape:
                mesh = open3d.read_triangle_mesh("stitching_big_semicone.ply")
            elif 'line' in shape:
                mesh = open3d.read_triangle_mesh("big_line.ply")
            elif 'cilinder' in shape:
                mesh = open3d.read_triangle_mesh("cil.ply")
            if mesh is not None:
                mesh.compute_vertex_normals()
                #import pdb; pdb.set_trace()
                open3d.draw_geometries([pcd, mesh])
                return
        open3d.draw_geometries([pcd])

    def visualize2_pointclouds(self, pc_list, color_list):
        for i, pc in enumerate(pc_list):
            pc = self.check_pointcloud_type(pc)
            if color_list is not None:
                pc.paint_uniform_color(color_list[i])
        open3d.draw_geometries(pc_list)

    def get_contact_info(self, directory, num, only_cart=False):
        #import pdb; pdb.set_trace()
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
                gs2_list.append(cv2.imread(path)) #.replace(str(num),'5')))
            elif 'wsg' in elem:
                wsg_list.append(load_obj(path))

        force_list = []
        for elem in wsg_list:
            force_list.append(elem['force'])

        return cart, gs1_list, gs2_list, wsg_list, force_list, gs1_back, gs2_back

    def get_local_pointcloud(self, gs_id, directory='', num=-1, new_cart=None, model_path=None, model=None, threshold=0.1, swap = None):
        # 1. We convert the raw image to height_map data
        #import pdb; pdb.set_trace()
        print swap
        #import pdb; pdb.set_trace()
        if swap is not None:
            new_cart, gs1_list, gs2_list, wsg_list, force_list, gs1_back, gs2_back = self.get_contact_info(directory, swap) #TODOM: why so many info?
        cart, gs1_list, gs2_list, wsg_list, force_list, gs1_back, gs2_back = self.get_contact_info(directory, num) #TODOM: why so many info?
        if new_cart is not None:
            cart = new_cart
        print cart
        if gs_id == 1:
            params_gs = self.params_gs1
            size = self.params_dict['input_shape_gs1'][0:2]
        elif gs_id == 2:
            params_gs = self.params_gs2
            size = self.params_dict['input_shape_gs2'][0:2]


            test_image = gs2_list[0]

            weights_file = model_path
            if 'grad' in weights_file: output_type = 'grad'
            elif 'angle' in weights_file: output_type = 'angle'
            else: output_type = 'height'
            if 'gray' in weights_file: input_type = 'gray'
            else: input_type = 'rgb'
            height_map, processed_image = raw_gs_to_depth_map(test_image=test_image, ref=None, model_path=weights_file, 
                plot=False, save=False, path='', output_type=output_type, model=model, input_type=input_type, get_NN_input = True)

            # cv2.imshow('hm', height_map)
            # cv2.waitKey(0)

            mean = np.mean(height_map)
            dev = np.std(height_map)
            a = mean - dev
            height_map = (height_map > 0.01)*height_map  # NOTE: We are multiplying the height by 10 for visualization porpuses!!!!!
            height_map = (height_map > a)*height_map
            #height_map = height_map+0.05
            height_map = cv2.resize(height_map, dsize=(size[1], size[0]), interpolation=cv2.INTER_LINEAR) #TODO: why? TODO WE NEED THIS?
            
            #outImage = cv2.add(cv2.cvtColor((height_map*5).astype(np.uint8),cv2.COLOR_GRAY2RGB).astype(np.float32),processed_image[0,:,:,0:3])
            #import pdb; pdb.set_trace()
            #cv2.imshow('a', outImage); 
            #cv2.waitKey()
            
        forloops_time_start = time.time()

        # 2. We convert height_map data into world position
        gripper_state = {}
        gripper_state['pos'] =  cart[0:3]
        print cart[0:3]
        gripper_state['quaternion'] = cart[-4:]
        gripper_state['Dx'] = wsg_list[0]['width']/2.0
        gripper_state['Dz'] = 372.3#365#139.8 + 72.5 + 160  # Base + wsg + finger
        print gripper_state
        pointcloud = []
        for i in range(height_map.shape[0]):
            for j in range(height_map.shape[1]):
                #print 'shape: ', height_map.shape[1]
                if(height_map[i][j] >= threshold) : # TODO: HACK MARIA
                    world_point = pxb_2_wb_3d(
                        point_3d=(i, j, height_map[i][j]),
                        gs_id=gs_id,
                        gripper_state = gripper_state,
                        fitting_params = params_gs
                    )
                    a = np.asarray(world_point)
                    pointcloud.append(a)
        print 'Mean pointcloud: ', np.mean(np.array(pointcloud), axis = 0)
        print 'Gripper opening: ', gripper_state['Dx']
        print pointcloud[0]

        print "Time used by forloops: " + str(time.time() - forloops_time_start)
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

    def stitch_pointclouds_open3D(self, source, target, threshold = 1, trans_init = np.eye(4), max_iteration = 2000, with_plot = True, with_transformation = False):
        source = self.check_pointcloud_type(source)
        target = self.check_pointcloud_type(target)
        target.paint_uniform_color([0.1, 0.1, 0.7])
        print dir(open3d)
        if with_plot:
            draw_registration_result(source, target, trans_init)

        print("Initial alignment")
        evaluation = open3d.evaluate_registration(source, target, threshold, trans_init)
        print(evaluation)
        #import pdb; pdb.set_trace()
        print("Apply point-to-point ICP")
        reg_p2p = open3d.registration_icp(source, target, threshold, trans_init,
                open3d.TransformationEstimationPointToPoint(),
                open3d.ICPConvergenceCriteria(max_iteration = max_iteration))
        print(reg_p2p)
        #import pdb; pdb.set_trace()
        print("Transformation is:")
        print(reg_p2p.transformation)
        print("")
        if with_plot:
            draw_registration_result(source, target, reg_p2p.transformation)
        source.transform(reg_p2p.transformation)
        if with_transformation:
            return np.concatenate([np.array(source.points), np.array(target.points)], axis=0), reg_p2p.transformation
        return np.concatenate([np.array(source.points), np.array(target.points)], axis=0)
        
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

    def get_global_pointcloud(self, gs_id, directory, touches, global_pointcloud, model_path=None, model=None, threshold = 0.1, swap =[]):
        for it,i in enumerate(touches):
            exp = str(i)
            print "Processing img " + exp + "..."
            try:
                swap_it = None
                if it < len(swap): swap_it = swap[it]
                local_pointcloud = self.get_local_pointcloud(gs_id=gs_id, directory=directory, num=i, model_path=model_path, 
                                model=model, threshold = threshold, swap =swap_it)

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


def create_point_cloud(name_id = '', gs_id = 2, touch_list = range(300), model_path = None, threshold = 0.1, 
            is_save = False, is_visualize = True, is_half = False, global_pointcloud = None, swap =[]):
    
    loc = Location()
    directory = '/media/mcube/data/shapes_data/object_exploration/' + name_id + '/'
    gs_id = 2

    # Build model
    keras.losses.custom_loss = custom_loss
    model = load_model(model_path)

    global_pointcloud = loc.get_global_pointcloud(gs_id=gs_id, directory=directory, touches=touch_list, 
                global_pointcloud = global_pointcloud, model_path=model_path, model=model, threshold = threshold, swap = swap)
    global_pointcloud = np.array(global_pointcloud)
    # Rotate
    if is_half: rotation = int(directory[directory.find('rot=')+4])*np.pi/4
    else: rotation = int(directory[directory.find('rot=')+4])*np.pi/8 
    aux_global_pointcloud = copy.deepcopy(global_pointcloud)
    aux_global_pointcloud[:,1] = np.cos(rotation)*(global_pointcloud[:,1]-y_off) + np.sin(rotation)*(global_pointcloud[:,2]-z_off) + y_off
    aux_global_pointcloud[:,2] = np.cos(rotation)*(global_pointcloud[:,2]-z_off) - np.sin(rotation)*(global_pointcloud[:,1]-y_off) + z_off
    global_pointcloud = aux_global_pointcloud
    
    # Save and visualize
    if is_save:
        np.save('/media/mcube/data/shapes_data/pointclouds/' + name_id + '.npy', global_pointcloud)
    if is_visualize:
        loc.visualize_pointcloud(global_pointcloud)
    
    return global_pointcloud
    

def add_point_clouds(pc_ids, rotations = [], is_save = False, is_visualize = True, final_global_pointcloud = None, 
                        gs_id = 2, touch_list = range(300), model_path = '', is_half = False, threshold = 0.1):
        
    directory = '/media/mcube/data/shapes_data/pointclouds/'
    if len(pc_ids) == 1:
        aux_pc_ids = []
        for rotation in rotations:
            aux_pc_ids.append(pc_ids[0].format(rotation))
        pc_ids = aux_pc_ids
        
    for pc_id in pc_ids:
        print pc_id
        if not os.path.isfile(directory + pc_id+'.npy'):
            create_point_cloud(name_id = pc_id, gs_id = gs_id, touch_list = touch_list, model_path = model_path, 
            is_save = True, is_visualize = False, is_half = is_half, threshold = threshold)
        global_pointcloud = np.load(directory + pc_id+'.npy')
        print global_pointcloud.shape
        
        if final_global_pointcloud is None: final_global_pointcloud = copy.deepcopy(global_pointcloud)
        else: final_global_pointcloud = np.concatenate([final_global_pointcloud, global_pointcloud], axis=0)

    if is_save:
        np.save(pc_id[0]+'_plus_{}_pointclouds.npy'.format(len(pc_ids)-1), final_global_pointcloud)
    if is_visualize:
        loc = Location()
        loc.visualize_pointcloud(final_global_pointcloud)
    
    return final_global_pointcloud
    
def rotate_pointcloud(pc_id, rotations = [], is_save = False, is_visualize = True, final_global_pointcloud = None, 
                        gs_id = 2, touch_list = range(300), model_path = '', is_half = False, threshold = 0.1):
    directory = '/media/mcube/data/shapes_data/pointclouds/'
    if not os.path.isfile(directory + pc_id+'.npy'):
        create_point_cloud(name_id = pc_id, gs_id = gs_id, touch_list = touch_list, model_path = model_path, 
        is_save = True, is_visualize = False, is_half = is_half, threshold = threshold)
    global_pointcloud = np.load(directory + pc_id+'.npy')
    
    
    for i in rotations:
        print i
        aux_global_pointcloud = copy.deepcopy(global_pointcloud)
        aux_global_pointcloud[:,1] = np.cos(i)*(global_pointcloud[:,1]-y_off) + np.sin(i)*(global_pointcloud[:,2]-z_off) + y_off
        aux_global_pointcloud[:,2] = np.cos(i)*(global_pointcloud[:,2]-z_off) - np.sin(i)*(global_pointcloud[:,1]-y_off) + z_off
        if final_global_pointcloud is None: final_global_pointcloud = aux_global_pointcloud
        else: final_global_pointcloud = np.concatenate([final_global_pointcloud, aux_global_pointcloud], axis=0)
        
    
    
    if is_save:
        np.save(pc_id[0]+'_rotated_{}_times.npy'.format(len(rotations)), final_global_pointcloud)
    if is_visualize:
        loc = Location()
        loc.visualize_pointcloud(final_global_pointcloud)
    return final_global_pointcloud
    
    
def compute_stitch_error(full_pointcloud, initial_pointcloud, local_pointcloud, threshold = 10, max_iteration = 2000):
    
    loc=Location()
    full_pointcloud = loc.check_pointcloud_type(full_pointcloud)
    bad_fix = np.concatenate([initial_pointcloud, local_pointcloud], axis=0)
    bad_fix = loc.check_pointcloud_type(bad_fix)
    initial_evaluation = open3d.evaluate_registration(bad_fix, full_pointcloud, threshold, np.eye(4))
    print initial_evaluation
    result_pointcloud = loc.stitch_pointclouds_open3D(initial_pointcloud, local_pointcloud, 
            threshold = threshold, max_iteration = max_iteration, with_plot = True) 
    result_pointcloud = loc.check_pointcloud_type(result_pointcloud)
    final_evaluation = open3d.evaluate_registration(result_pointcloud, full_pointcloud, threshold, np.eye(4))
    print final_evaluation
    return initial_evaluation, final_evaluation
    
def compre_pointcloud_ply(global_pointcloud, shape, is_visualize = True):
    if 'semicone' in shape: 
        x_off = 830
        
    afinal_global_pointcloud = copy.deepcopy(global_pointcloud)
    if 'line' in shape:
        afinal_global_pointcloud[:,0] = global_pointcloud[:,1] - y_off
        afinal_global_pointcloud[:,1] = global_pointcloud[:,0] - x_off
        afinal_global_pointcloud[:,2] = global_pointcloud[:,2] - z_off
    else:
        afinal_global_pointcloud[:,0] = global_pointcloud[:,2] - z_off
        afinal_global_pointcloud[:,1] = global_pointcloud[:,1] - y_off
        afinal_global_pointcloud[:,2] = -global_pointcloud[:,0] + x_off 
    if is_visualize:
        loc = Location()
        loc.visualize_pointcloud(afinal_global_pointcloud, shape = shape)

def stitch_orientations(name_id, gs_id = 2, model_path = None, threshold = 0.1, 
            is_save = False, is_visualize = True, is_half = False, global_pointcloud = None):
    directory = '/media/mcube/data/shapes_data/object_exploration/' + name_id + '/'
    loc = Location()
    num_images = len(os.listdir(directory))-1 #Because of air
    global_pointcloud_0 = create_point_cloud(name_id = name_id, gs_id = gs_id, touch_list = range(0, num_images/3), model_path = model_path, 
        is_save = False, is_visualize = False, is_half = is_half, threshold = threshold)
    global_pointcloud_1 = create_point_cloud(name_id = name_id, gs_id = gs_id, touch_list = range(num_images/3, num_images/3*2), model_path = model_path, 
        is_save = False, is_visualize = False, is_half = is_half, threshold = threshold)
    global_pointcloud_2 = create_point_cloud(name_id = name_id, gs_id = gs_id, touch_list = range(num_images/3*2, num_images), model_path = model_path, 
        is_save = False, is_visualize = False, is_half = is_half, threshold = threshold)
    global_pointcloud = loc.stitch_pointclouds_open3D(global_pointcloud_0, global_pointcloud_1, threshold = 10, max_iteration = 2000, with_plot = True) 
    global_pointcloud = loc.stitch_pointclouds_open3D(global_pointcloud, global_pointcloud_2, threshold = 10, max_iteration = 2000, with_plot = True) 
    if is_save:
        np.save(pc_id[0]+'_orientation_stiched.npy'.format(len(rotations)), global_pointcloud)
    if is_visualize:
        loc = Location()
        loc.visualize_pointcloud(global_pointcloud)
    return global_pointcloud

def rotate_pointcloud_gripper(global_pointcloud, quaternion, gs_id = 2, is_visualize = True):
    euler = tfm.euler_from_quaternion(quaternion)
    if gs_id == 2: transform = tfm.euler_matrix(0, np.pi-euler[2], 0)
    else: transform = tfm.euler_matrix(euler[0], euler[2], 0)
    
    global_pointcloud[:,0] = global_pointcloud[:,0] - x_off
    global_pointcloud[:,1] = global_pointcloud[:,1] - y_off
    global_pointcloud[:,2] = global_pointcloud[:,2] - z_off
    initial_global_pointcloud = copy.deepcopy(global_pointcloud)
    global_pointcloud = loc.check_pointcloud_type(global_pointcloud)
    global_pointcloud.transform(transform)

    if is_visualize:
        loc.old_visualize_pointcloud(initial_global_pointcloud)
        loc.old_visualize_pointcloud(global_pointcloud.points)

    return global_pointcloud.points


if __name__ == "__main__":

    loc = Location()
    name_id = 'cilinder_l=50_h=20_dx=10_dy=10_rot=0_debug'#'flashlight_l=70_h=20_dx=10_dy=10_rot=0_debug'
    name_id = 'flashlight_l=70_h=20_dx=10_dy=10_rot=0_debug'#'flashlight_l=70_h=20_dx=10_dy=10_rot=0_debug'
    gs_id = 2
    model_path = '/home/mcube/weights_server_last/weights_type=all_08-23-2018_num=2000_gs_id=2_in=rgb_out=height_epoch=100_NN=basic_aug=5.hdf5'
    directory = '/media/mcube/data/shapes_data/pointclouds/'
    name_id = 'cilinder_l=50_h=20_dx=10_dy=10_rot=0_debug'
    x_off = 837.2
    y_off = 376#376.2#376.8 #371.13 
    z_off = 295#286.5 #291#293.65 #284.22
    
    touch_list = range(10,12) + range(26,28) #+ range(43,45)
    touch_list = range(200) #+ range(26,27) #+ range(43,45)
    
    ## Stitch orientations
    #name_id = 'flashlight_l=70_h=20_dx=10_dy=10_rot=0_debug' #'cilinder_l=50_h=20_dx=10_dy=10_rot=0_debug'
    #global_pointcloud = stitch_orientations(name_id, gs_id = 2, model_path = model_path, threshold = 0.1, 
            #is_save = False, is_visualize = True, is_half = False, global_pointcloud = None)
    
    ## Do stitching 
    name_id = 'flashlight_l=70_h=20_dx=10_dy=10_rot=0_debug' #'cilinder_l=50_h=20_dx=10_dy=10_rot=0_debug'
    global_pointcloud = np.load(directory + name_id + '.npy')
    it = 5
    it_2 = 5
    global_pointcloud_1 = create_point_cloud (name_id = name_id, gs_id = gs_id, touch_list = range(it) + range(it+1,200), model_path = model_path,
                is_save = False, is_visualize = True, is_half = False, threshold = 0.1)
    global_pointcloud_2 = create_point_cloud (name_id = name_id, gs_id = gs_id, touch_list = range(it,it+1), model_path = model_path,
                is_save = False, is_visualize = True, is_half = False, threshold = 0.1, swap =[it_2])
    compute_stitch_error(global_pointcloud, global_pointcloud_1, global_pointcloud_2, threshold = 1, max_iteration = 2000)
    import pdb;pdb.set_trace()
    
    
    ## Rotate same pointcloud 
    rotations = np.linspace(-180,180,11)*np.pi/180
    global_pointcloud = rotate_pointcloud(name_id, rotations = rotations, is_save = True, is_visualize = True,
        model_path = model_path, gs_id=gs_id, touch_list=touch_list, is_half = True, threshold = 0.15)
    global_pointcloud = compre_pointcloud_ply(global_pointcloud, shape=name_id)
    
    
    ## Get pointcloud from different rotations
    name_id = 'flashlight_l=70_h=20_dx=10_dy=10_rot={}_debug'
    
    global_pointcloud = add_point_clouds([name_id], rotations = range(8), is_save = True, is_visualize = True,
        model_path = model_path, gs_id=gs_id, touch_list=touch_list, is_half = True, threshold = 0.15)
    quaternion = np.array([0.122787803968973,  -0.696364240320019,   0.696364240320019, -0.122787803968973])
    rotate_pointcloud_gripper(global_pointcloud, quaternion)
        
    ## Rotate same pointcloud    
    rotations = np.array([0, 132])*np.pi/180
    name_id = 'flashlight_l=70_h=20_dx=10_dy=10_rot=0_debug'
    global_pointcloud = rotate_pointcloud(name_id, rotations = rotations, is_save = True, is_visualize = True,
        model_path = model_path, gs_id=gs_id, touch_list=touch_list, is_half = True, threshold = 0.15)
                    
                        
  
    
    
    #global_pointcloud_0 = np.load('/media/mcube/data/shapes_data/pointclouds/flashlight_l=70_h=20_dx=10_dy=10_rot=0_debug_angle_0_rotated.npy')
    #global_pointcloud_1 = np.load('/media/mcube/data/shapes_data/pointclouds/flashlight_l=70_h=20_dx=10_dy=10_rot=0_debug_angle_20_rotated.npy')
    #global_pointcloud_2 = np.load('/media/mcube/data/shapes_data/pointclouds/flashlight_l=70_h=20_dx=10_dy=10_rot=0_debug_angle_-20_rotated.npy')
    
    #global_pointcloud = loc.stitch_pointclouds_open3D(global_pointcloud_0, global_pointcloud_1, threshold = 10, max_iteration = 2000, with_plot = True) 
    #global_pointcloud = loc.stitch_pointclouds_open3D(global_pointcloud, global_pointcloud_2, threshold = 10, max_iteration = 2000, with_plot = True) 
    #loc.visualize_pointcloud(global_pointcloud, shape = name_id)



    # missing = loc.get_local_pointcloud(
    #     gs_id=2,
    #     directory='/media/mcube/data/shapes_data/pos_calib/bar_front/',
    #     num=16
    # )
    # loc.visualize_pointcloud(missing)
    # from skimage.measure import compare_ssim as ssim
    # imageA = cv2.imread('/media/mcube/data/shapes_data/pos_calib/full_bar_v2_front/p_20/GS2_0.png')
    # #touch_list = touch_list_aux
    # touch_list = range(16, 21)
    # touch_list += range(21, 50)
    # for num in touch_list[1:10]:
    #     directory = '/media/mcube/data/shapes_data/pos_calib/bar_front/'
    #     cart = loc.get_contact_info(directory, num, only_cart=True)
    #     print cart
    #     missing = loc.get_local_pointcloud(
    #         gs_id=2,
    #         directory=directory,
    #         num=6,
    #         new_cart=cart
    #     )
        #np.save('/media/mcube/data/shapes_data/pointclouds/' + 'pcd_6_front_bar.npy', global_pointcloud)
        # missing = np.array(missing)
        # loc.visualize_pointcloud(global_pointcloud)
        #
        # #new_pointcloud = loc.stitch_pointclouds(global_pointcloud, missing)
        # trans_init = np.eye(4)
        # trans_init[0,-1] = 105
        # trans_init[1,-1] = 0
        # new_pointcloud = loc.stitch_pointclouds_open3D(global_pointcloud, missing, trans_init = trans_init, threshold = 0.5)
        # print 'num: ', num
        # imageB = cv2.imread('/media/mcube/data/shapes_data/pos_calib/full_bar_front/p_{}/GS2_0.png'.format(num))
        # print 'ssim: ', ssim(imageA, imageB, multichannel=True)
        # print 'cosine distance', loc.get_distance_images(imageA, imageB)
        # imageB = cv2.imread('/media/mcube/data/shapes_data/pos_calib/full_bar_v2_front/p_{}/GS2_0.png'.format(num))
        # #print 'ssim: ', ssim(imageA, imageB, multichannel=True)
        # print 'cosine distance v2 ', loc.get_distance_images(imageA, imageB)
        # imageB = cv2.imread('/media/mcube/data/shapes_data/pos_calib/full_bar_f=20_v1_front/p_{}/GS2_0.png'.format(num))
        # #print 'ssim: ', ssim(imageA, imageB, multichannel=True)
        # print 'cosine distance f=20 ', loc.get_distance_images(imageA, imageB)
        #
        # loc.visualize2_pointclouds([new_pointcloud, missing])

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
    ##loc.visualize2_pointclouds([global_pointcloud, missing])
    #''
    #if 'semicone' in name_id: 
        #x_off = 833.5
        #y_off = 375.5#376.8 #371.13 
        #z_off = 299.5 #291#293.65 #284.22
    ##z_off = 284.22
    #if 'flash' in name_id: 
        #x_off = 833.5
        #y_off = 376#376.8 #371.13 
        #z_off = 295 #291#293.65 #284.22
    ##z_off = 284.22
    
#'''
