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
from matplotlib.path import Path
from location import Location
import copy
from world_positioning import grb2wb, gr2w
from world_positioning import pxb_2_wb_3d, quaternion_matrix, fast_pxb2wb_3d
import pdb
global x_off, y_off, z_off
np.random.seed(seed=1)
import glob
import scipy.io as sio
import shutil

def pointcloud_rotation(global_pointcloud, rotation):
    aux_global_pointcloud = copy.deepcopy(global_pointcloud)
    aux_global_pointcloud[:,1] = np.cos(rotation)*(global_pointcloud[:,1]-y_off) + np.sin(rotation)*(global_pointcloud[:,2]-z_off) + y_off
    aux_global_pointcloud[:,2] = np.cos(rotation)*(global_pointcloud[:,2]-z_off) - np.sin(rotation)*(global_pointcloud[:,1]-y_off) + z_off
    return aux_global_pointcloud

def compare_poses(pos1, pos2):
    return np.linalg.norm(pos1-pos2)

def rmse_pcs(pc1, pc2):
    num_pts = pc1.shape[0]
    return np.sqrt(np.sum((pc1 - pc2) ** 2)/num_pts)



if __name__ == "__main__":

  x_off = 839#837.2
  y_off = 372.13#369.13# .13#376.2#376.8 #371.13 
  z_off = 286.5#284.45#286.5 #291#293.65 #284.22

  loc = Location()
  threshold = 0.1
  gripper_threshold = 0.5
  pc_threshold = 5
  
  name_ids = ['flashlight_l=80_h=20_dx=10_dy=10_rot={}_09-03-2018/']
  '''
  , 
   'tape_l=80_h=20_dx=10_dy=10_rot={}_09-03-2018/',
   'mentos_l=80_h=20_dx=10_dy=10_rot={}_09-03-2018/',
   'scissors_l=80_h=20_dx=10_dy=10_rot={}_09-03-2018/',
   'brush_l=80_h=20_dx=10_dy=10_rot={}_09-03-2018/']
  '''
  path_shape = '/media/mcube/data/shapes_data/processed_pointclouds/'
  all_NN_vectors = []
  all_gripper = []
  ### Get their vectors!
  for name_id in name_ids:
    if 'flashlight' in name_id: rotations = [0,4,8,12]#[0,2,4,6, 8,10, 12, 14] ##,8,12]#[0,2,4,6,8,14,12] #range(2)
    else: rotations = [0,4,8,12]

    data_path = path_shape + name_id.format(0) + '/dataset_threshold={}_rotations={}.npy'.format(threshold*100,rotations)
    if os.path.isfile(data_path):
      print data_path
      dataset = np.load(data_path)
      NN_vectors = dataset.item().get('NN_vectors')
      all_NN_vectors.append(NN_vectors)
      gripper_openings = dataset.item().get('gripper_openings')
      all_gripper.append(gripper_openings)
  
  test_path = path_shape + 'test_grasps_location/' #_l=1000_h=20_dx=5_dy=20_rot=0_09-09-2018/'
  test_path = path_shape + 'flashlight_track/' #_l=1000_h=20_dx=5_dy=20_rot=0_09-09-2018/'
  path_test_images = glob.glob(test_path + 'data_*.npy')
  path_test_images.sort()
  #path_test_images = [path_test_images[-1]]
  it_ini = 0 
  path_test_images = path_test_images[it_ini::] 
  print test_path
  best_options = None
  name_id = None
  for it_p, path_test in enumerate(path_test_images):
    best_score = -1
    real_it = int(path_test[:-4].replace('_', ' ').split()[-1])
    print it_p+it_ini, path_test, real_it
    if real_it > 170 or real_it < 63:
      continue
    '''
    try:
      shutil.copy( '/media/mcube/data/shapes_data/processed_pointclouds/flashlight_track/pc_{}.png'.format(it_p),
                  '/media/mcube/data/shapes_data/processed_pointclouds/flashlight_track/new_pc_{}.png'.format(real_it))
    except: 
      print "Failed: ", it_p
    continue
    '''
    dataset = np.load(path_test)
    NN_vector = dataset.item().get('vector')
    depth_vect = dataset.item().get('height')
    gripper_opening = dataset.item().get('gripper')
    img = dataset.item().get('image')
    new_cart = dataset.item().get('cart')
    good_its = np.where(depth_vect.flatten() >= threshold)
    p3 = dataset.item().get('p3')
    p3 = p3[good_its, :][0]
    initial_ICP_rmse = []
    final_ICP_rmse = []
    ##### Compare the vectors 
    for it_id in range(len(name_ids)):
      img_score = []
      for it2 in range(len(all_NN_vectors[it_id])):
        img_score.append(1-scipy.spatial.distance.cosine(all_NN_vectors[it_id][it2], NN_vector))
      gripper_score = np.abs(np.array(all_gripper[it_id])- gripper_opening)< gripper_threshold 
      final_score = np.array(img_score)*gripper_score
      print 'From {} you get: {}'.format(name_ids[it_id], np.amax(final_score))
      if np.amax(final_score) > best_score:
        best_score = np.amax(final_score)
        best_options = copy.deepcopy(final_score)
        best_id = copy.deepcopy(it_id)
        name_id = name_ids[it_id]
    print best_score

    #if 'flashlight' in name_id: rotations = [0,2,4,6, 8,10, 12, 14] ##,8,12]#[0,2,4,6,8,14,12] #range(2)
    #else: rotations = [0,4,8,12]

        
    max_iteration = 2000
    num_pointclouds = 0
    num_per_rotation = []
    num_selected = [5]
    vec_rotations = []
    
    data_path = path_shape + name_id.format(0) + '/dataset_threshold={}_rotations={}.npy'.format(threshold*100,rotations)
    print data_path
    dataset = np.load(data_path)
    NN_vectors = dataset.item().get('NN_vectors')
    aux_num_pointclouds = dataset.item().get('aux_num_pointclouds')
    carts = dataset.item().get('carts')
    depth_vect = dataset.item().get('depth_vect')
    global_pointcloud = dataset.item().get('global_pointcloud')
    gripper_openings = dataset.item().get('gripper_openings')
    img_vect = dataset.item().get('img_vect')
    name_id = dataset.item().get('name_id')
    p3_vect = dataset.item().get('p3_vect')
    p4_vect = dataset.item().get('p4_vect')
    new_vec_rotations = dataset.item().get('new_vec_rotations')    
    #loc.visualize_pointcloud(global_pointcloud)
    num_pointclouds = copy.deepcopy(aux_num_pointclouds)
    vec_rotations = copy.deepcopy(new_vec_rotations)
  
    N_points = range(num_pointclouds)
    
    #Take the best one
    it = np.argmax(best_options)
    best_rotation = vec_rotations[it]; cart = carts[it]
    print 'best cart:', cart
    
    new_p4 = gr2w(p3, cart[0:3], cart[-4:])  #There are only 3 possible cart[-4] --> simplify computations
          
    result_pointcloud, changed_pointcloud, transform_process, initial_evaluation = loc.stitch_pointclouds_open3D(p4_vect[it], new_p4, threshold = pc_threshold, max_iteration = max_iteration, with_plot = False,  with_transformation = True) 
    initial_ICP_rmse.append(initial_evaluation.inlier_rmse)
    final_ICP_rmse.append(transform_process.inlier_rmse)
    #loc.old_visualize_pointcloud( np.concatenate([p4_vect[it], new_p4], axis=0))
    #plt.imshow(np.concatenate([img,img_vect[it]], axis=1)); plt.show()
    #import pdb; pdb.set_trace()
    #continue
    
    
    #mean_new_p4 = np.mean(new_p4, axis = 0)
    #distance_points = np.linalg.norm(np.subtract(rotated_global_pointcloud, mean_new_p4), axis=1)
    '''
    num_points_pcs = [10000, 50000, 100000, 500000]
    for num_points_pc in num_points_pcs:
      closest_pointcloud = rotated_global_pointcloud[np.argsort(distance_points)[:num_points_pc],:]
      result_pointcloud, changed_pointcloud, transform_process, initial_evaluation = loc.stitch_pointclouds_open3D(closest_pointcloud, new_p4,   threshold = pc_threshold, max_iteration = max_iteration, with_plot = False,  with_transformation = True) 
      final_error_pcs_local.append(rmse_pcs(rotated_p4, changed_pointcloud))
      initial_ICP_rmse.append(initial_evaluation.inlier_rmse)
      final_ICP_rmse.append(transform_process.inlier_rmse)
      print 'third case, {}: '.format(num_points_pc), rmse_pcs(rotated_p4, changed_pointcloud)
    '''
    arg_best_options = np.argsort(best_options)[::-1]
    print best_options[arg_best_options[0:10]]
    #import pdb; pdb.set_trace()
    for num_sel in num_selected:
      local_global_pointcloud = None
      for it_opt in arg_best_options[0:int(num_sel)]:
        local_pointcloud = pointcloud_rotation(p4_vect[it_opt], (vec_rotations[it_opt] - best_rotation)*np.pi/8.0)
        if local_global_pointcloud is None: local_global_pointcloud = local_pointcloud
        else: local_global_pointcloud = np.concatenate([local_global_pointcloud, local_pointcloud], axis=0)

      result_pointcloud, changed_pointcloud, transform_process, initial_evaluation = loc.stitch_pointclouds_open3D(local_global_pointcloud, new_p4,   threshold = pc_threshold, max_iteration = max_iteration, with_plot = False,  with_transformation = True) 
      initial_ICP_rmse.append(initial_evaluation.inlier_rmse)
      final_ICP_rmse.append(transform_process.inlier_rmse)
         
    rotated_global_pointcloud = pointcloud_rotation(global_pointcloud, - best_rotation*np.pi/8.0)
    ### Apply invers matrix tranform
    trans_matrix = transform_process.transformation
    inv_trans_matrix = np.linalg.inv(trans_matrix)
    quaternion = cart[-4:]
    mat_quat = quaternion_matrix(quaternion)
    inv_mat_quat = np.linalg.inv(mat_quat)
    pos = cart[0:3]
    num_pts_g =  rotated_global_pointcloud.shape[0]
    num_pts_l =  changed_pointcloud.shape[0]
    g1 = inv_trans_matrix.dot(np.append(rotated_global_pointcloud, np.ones((num_pts_g,1)), axis=1).transpose())
    l1 = inv_trans_matrix.dot(np.append(changed_pointcloud, np.ones((num_pts_l,1)), axis=1).transpose())
    #print l1 == new_p4
    
    g2 = g1[:3,:].transpose() - 1000*pos
    l2 = l1[:3,:].transpose() - 1000*pos
    g3 = inv_mat_quat.dot(np.append(g2, np.ones((g2.shape[0],1)), axis=1).transpose()).transpose()
    l3 = inv_mat_quat.dot(np.append(l2, np.ones((l2.shape[0],1)), axis=1).transpose()).transpose()
    g4 = gr2w(g3, new_cart[0:3], new_cart[-4:]) 
    l4 = gr2w(l3, new_cart[0:3], new_cart[-4:]) 
    #loc.old_visualize_pointcloud( np.concatenate([g4, l4], axis=0))

    #new_cart = 838.6, 372.13 , 657., 0., np.sqrt(0.5), -np.sqrt(0.5), 0.
    #np.save(path_test + '{}_global.npy'.format(it_p), g4)
    #np.save(path_test + '{}_local.npy'.format(it_p), l4)
    sio.savemat('/media/mcube/data/shapes_data/processed_pointclouds/flashlight_track/' + '{}_local'.format(real_it), {'vect':l4})
    sio.savemat('/media/mcube/data/shapes_data/processed_pointclouds/flashlight_track/' + '{}_global'.format(real_it), {'vect':g4})
    #np.save(name_id.format(0)[:-1] + '_global.npy', g4)
    #np.save(name_id.format(0)[:-1] + '_local.npy', l4)
    #import pdb; pdb.set_trace()
