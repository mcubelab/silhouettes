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
from location import Location


if __name__ == "__main__":

    loc = Location()
    # Data info
    name_id = 'flashlight_l=70_h=20_dx=10_dy=10_rot=0_debug'
    directory = '/media/mcube/data/shapes_data/object_exploration/' + name_id + '/'
    gs_id = 2

    # Build model
    model_path = '/home/mcube/weights_server_last/weights_type=all_08-23-2018_num=2000_gs_id=2_in=rgb_out=height_epoch=100_NN=basic_aug=5.hdf5'
    keras.losses.custom_loss = custom_loss
    model = load_model(model_path)


    touch_list = range(10,12) + range(26,28) #+ range(43,45)
    touch_list = range(48) #+ range(26,27) #+ range(43,45)
    #touch_list = range(26,28) #+ range(43,45)
    #touch_list = [0,1,4,5,8,9]  #7,8, 12,13,17,18]0,1,2,3,4,
    global_pointcloud = None
    global_pointcloud = loc.get_global_pointcloud(gs_id=gs_id, directory=directory, touches=touch_list, 
                global_pointcloud = global_pointcloud, model_path=model_path, model=model, threshold = 0.15)
    
    global_pointcloud_1 = np.array(copy.deepcopy(global_pointcloud))
        
    np.save('/media/mcube/data/shapes_data/pointclouds/' + name_id + '.npy', global_pointcloud_1)
    loc.visualize_pointcloud(global_pointcloud_1)
        
        
        
        
        
    
    #loc.old_visualize_pointcloud(global_pointcloud)
    #loc.visualize_pointcloud(global_pointcloud)
    #
    
    rotation = int(directory[directory.find('rot=')+4])
    aux_global_pointcloud = copy.deepcopy(global_pointcloud)
    aux_global_pointcloud[:,1] = np.cos(rotation)*(global_pointcloud[:,1]-y_off) + np.sin(rotation)*(global_pointcloud[:,2]-z_off) + y_off
    aux_global_pointcloud[:,2] = np.cos(rotation)*(global_pointcloud[:,2]-z_off) - np.sin(rotation)*(global_pointcloud[:,1]-y_off) + z_off
    golab_pointcloud = aux_global_pointcloud
    np.save('/media/mcube/data/shapes_data/pointclouds/' + name, global_pointcloud)
    #'''

    
    name_id = 'flashlight_l=70_h=20_dx=10_dy=10_rot=0_debug' #'flashlight_l=100_h=20_dx=2_dy=10_rot=0_debug' #'big_semicone_l=40_h=20_d=5_rot=0_more_empty' #'cilinder_thick_l=50_h=20_dx=10_dy=10_rot=0_test' #  #'cilinder_l=50_h=20_dx=5_dy=5_rot=0_test' # 'big_semicone_l=40_h=20_d=5_rot=0_empty'#
    #name_id = 'cilinder_l=50_h=20_dx=10_dy=10_rot=0_debug'.format(0)
    name = name_id +'.npy'
    loc = Location()
    
    x_off = 837.5
    
    y_off = 376#376.2#376.8 #371.13 
    
    z_off = 299.5#286.5 #291#293.65 #284.22
    if 'semicone' in name: 
        x_off = 833.5
        y_off = 375.5#376.8 #371.13 
        z_off = 299.5 #291#293.65 #284.22
    #z_off = 284.22
    if 'flash' in name: 
        x_off = 833.5
        y_off = 376#376.8 #371.13 
        z_off = 295 #291#293.65 #284.22
    #z_off = 284.22
    
    rotations = range(8)
    for i in rotations:
        if i == 0:
            global_pointcloud = np.load('/media/mcube/data/shapes_data/pointclouds/' + name)
        else:
            global_pointcloud_2 = np.load('/media/mcube/data/shapes_data/pointclouds/' + name.replace('rot=0', 'rot={}'.format(i)))
            rotation = i*np.pi/4
            aux_global_pointcloud = copy.deepcopy(global_pointcloud_2)
            aux_global_pointcloud[:,1] = np.cos(rotation)*(global_pointcloud_2[:,1]-y_off) + np.sin(rotation)*(global_pointcloud_2[:,2]-z_off) + y_off
            aux_global_pointcloud[:,2] = np.cos(rotation)*(global_pointcloud_2[:,2]-z_off) - np.sin(rotation)*(global_pointcloud_2[:,1]-y_off) + z_off
            #global_pointcloud = loc.stitch_pointclouds_open3D(global_pointcloud, aux_global_pointcloud, threshold = 10, max_iteration = 2000, with_plot = True) 
            global_pointcloud = np.concatenate([global_pointcloud, aux_global_pointcloud], axis=0)
    #loc.visualize_pointcloud(global_pointcloud, shape = name_id)

    #import pdb; pdb.set_trace()
    
    #np.save('/media/mcube/data/shapes_data/pointclouds/' + name + '_rotated', global_pointcloud)
    
    final_global_pointcloud = copy.deepcopy(global_pointcloud)
    final_global_pointcloud[:,0] -= x_off
    final_global_pointcloud[:,1] -= y_off
    final_global_pointcloud[:,2] -= z_off
    
    loc.old_visualize_pointcloud(final_global_pointcloud)
    
    for i in np.linspace(-np.pi, np.pi, 0):
        aux_global_pointcloud = copy.deepcopy(global_pointcloud)
        aux_global_pointcloud[:,1] = np.cos(i)*(global_pointcloud[:,1]-y_off) + np.sin(i)*(global_pointcloud[:,2]-z_off) # - y_off
        aux_global_pointcloud[:,2] = np.cos(i)*(global_pointcloud[:,2]-z_off) - np.sin(i)*(global_pointcloud[:,1]-y_off) #- z_off
        aux_global_pointcloud[:,0] -= x_off
        final_global_pointcloud = np.concatenate([final_global_pointcloud, aux_global_pointcloud], axis=0)
    
    #import pdb; pdb.set_trace()
    afinal_global_pointcloud = copy.deepcopy(final_global_pointcloud)
    if 'line' in name_id:
        afinal_global_pointcloud[:,0] = final_global_pointcloud[:,1]
        afinal_global_pointcloud[:,1] = -final_global_pointcloud[:,0]
        afinal_global_pointcloud[:,2] = final_global_pointcloud[:,2]
    else:
        afinal_global_pointcloud[:,0] = final_global_pointcloud[:,2]
        afinal_global_pointcloud[:,1] = final_global_pointcloud[:,1]
        afinal_global_pointcloud[:,2] = -final_global_pointcloud[:,0] # +np.mean(final_global_pointcloud[:,0])-33.99#-final_global_pointcloud[:,0]
    print np.mean(afinal_global_pointcloud, axis=0)
    loc.visualize_pointcloud(afinal_global_pointcloud, shape = name_id)
    #loc.visualize_pointcloud(afinal_global_pointcloud)
    '''
    
    '''
    #global_pointcloud_0 = np.load('/media/mcube/data/shapes_data/pointclouds/flashlight_l=70_h=20_dx=10_dy=10_rot=0_debug_angle_0_rotated.npy')
    #global_pointcloud_1 = np.load('/media/mcube/data/shapes_data/pointclouds/flashlight_l=70_h=20_dx=10_dy=10_rot=0_debug_angle_20_rotated.npy')
    #global_pointcloud_2 = np.load('/media/mcube/data/shapes_data/pointclouds/flashlight_l=70_h=20_dx=10_dy=10_rot=0_debug_angle_-20_rotated.npy')
    
    #global_pointcloud = loc.stitch_pointclouds_open3D(global_pointcloud_0, global_pointcloud_1, threshold = 10, max_iteration = 2000, with_plot = True) 
    #global_pointcloud = loc.stitch_pointclouds_open3D(global_pointcloud, global_pointcloud_2, threshold = 10, max_iteration = 2000, with_plot = True) 
    #loc.visualize_pointcloud(global_pointcloud, shape = name_id)
   # '''



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
    #loc.visualize2_pointclouds([global_pointcloud, missing])
#'''
