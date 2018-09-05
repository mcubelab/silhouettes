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
from world_positioning import grb2wb
import pdb
global x_off, y_off, z_off
np.random.seed(seed=0)
   
def pointcloud_rotation(global_pointcloud, rotation):
    aux_global_pointcloud = copy.deepcopy(global_pointcloud)
    aux_global_pointcloud[:,1] = np.cos(rotation)*(global_pointcloud[:,1]-y_off) + np.sin(rotation)*(global_pointcloud[:,2]-z_off) + y_off
    aux_global_pointcloud[:,2] = np.cos(rotation)*(global_pointcloud[:,2]-z_off) - np.sin(rotation)*(global_pointcloud[:,1]-y_off) + z_off
    return aux_global_pointcloud

def compare_poses(pos1, pos2):
    return np.linalg.norm(pos1-pos2)

def rmse_pointclouds(pc1, pc2):
    return np.sqrt(((pc1 - pc2) ** 2).mean())



if __name__ == "__main__":

  x_off = 839#837.2
  y_off = 372.13#369.13# .13#376.2#376.8 #371.13 
  z_off = 286.5#284.45#286.5 #291#293.65 #284.22

  loc = Location()
  ## Count how many point clouds there are
  rotations = range(5)
  name_id = 'flashlight_l=80_h=20_dx=10_dy=10_rot={}_09-03-2018/'
  path_shape = '/media/mcube/data/shapes_data/processed_pointclouds/'
  num_averages = 20
  threshold = 0.1
  gripper_threshold = 2
  pc_threshold = 1
  max_iteration = 1
  num_pointclouds = 0
  num_per_rotation = []
  num_selected = 20
  for rotation in rotations:
    files = os.listdir(path_shape + name_id.format(rotation))
    num_files = len(files)-1 #Do not count rotation.py
    num_pointclouds += num_files
    num_per_rotation.append(num_files)
  cum_sum_points = np.cumsum(num_per_rotation)
  print num_pointclouds
  
  num_removes = [1]
  for i in np.linspace(0,1,11):
    if i and i < 1:
      num_removes.append(int(num_pointclouds*i))

  for num_remove in num_removes:
    for it in range(num_averages):
      #Select the right amount of pointclouds (p4 + orientation change)
      N = num_pointclouds - num_remove
      perm = np.random.permutation(num_pointclouds)
      vec_rotations = np.searchsorted(cum_sum_points, perm+1)
      N_points = perm[:N]
      new_point = perm[-1]
      gripper_openings = []
      NN_vectors = []
      carts = []
      rotated_carts = []
      global_pointcloud = None
      raw_images = []
      #Select the stitched pointcloud
      for i in range(N):
        rotation = vec_rotations[i]
        if rotation: previous_points = cum_sum_points[rotation-1]
        else: previous_points = 0
        pc_path = path_shape + name_id.format(rotation) + 'data_{}.npy'.format(perm[i]-previous_points)
        data = np.load(pc_path)
        p4 = data.item().get('p4')
        gripper_openings.append(data.item().get('gripper'))
        NN_vectors.append(data.item().get('vector'))
        carts.append(data.item().get('cart'))
        rotated_carts.append(pointcloud_rotation(np.array([carts[-1][0:3]])*1000, rotation*np.pi/8.0))
        #raw_images.append(data.item().get('image'))
        depth_map = data.item().get('height')
        #### Use threshold ####
        ##
        ##### TODO #####
        local_pointcloud = pointcloud_rotation(p4, rotation*np.pi/8.0)
        if global_pointcloud is None: global_pointcloud = local_pointcloud
        else: global_pointcloud = np.concatenate([global_pointcloud, local_pointcloud], axis=0)
      
      #loc.visualize_pointcloud(global_pointcloud)
      #New point cloud
      i = -1
      rotation = vec_rotations[i]
      if rotation: previous_points = cum_sum_points[rotation-1]
      else: previous_points = 0
      pc_path = path_shape + name_id.format(rotation) + 'data_{}.npy'.format(perm[i]-previous_points)
      data = np.load(pc_path)
      p4 = data.item().get('p4')
      p3 = data.item().get('p3')
      #### Use threshold ####
      ##
      ##### TODO #####
      new_gripper_opening = data.item().get('gripper')
      new_vector = data.item().get('vector')
      new_image = data.item().get('image')
      new_cart = data.item().get('cart')
      new_rotated_cart = pointcloud_rotation(np.array([new_cart[0:3]])*1000, rotation*np.pi/8.0) #### TODO, very  likely to be bad!!!!
      gripper_score = np.abs(np.array(gripper_openings)-new_gripper_opening)
      img_score = []
      for i in range(N):
        img_score.append(1-scipy.spatial.distance.cosine(new_vector, NN_vectors[i]))
      img_score = np.array(img_score)
      combined_score = img_score*(gripper_score < gripper_threshold)
      good_local_pointcloud = pointcloud_rotation(p4, rotation*np.pi/8.0)
      
      
      stitch_score = []
      best_options = np.argpartition(combined_score, -num_selected)[-num_selected:]
      initial_scores = []
      final_scores = []
      final_errors = []
      initial_error_poses = []
      initial_error_pcs = []
      final_error_poses = []
      initial_pos = []
      final_pos = []
      for i in range(num_selected):
        it = best_options[i]
        cart = rotated_carts[it]
        initial_pos.append(cart)
        initial_error_poses.append(compare_poses(cart[0:3], new_rotated_cart[0:3]))
        ##
        
        ##
        ## I wanted to rotate the global pointcloud?????????????? THINK ABOUT IT!!!!
        '''
        #TODO: new_p4 = grb2wb(point=p3, gripper_pos=cart[0:3], quaternion=cart[-4:])
        new_p4 = copy.deepcopy(p3)
        initial_error_pcs.append(rmse_pc(p4, new_p4))
        result_pointcloud, changed_pointcloud, transform_proccess, initial_evaluation = loc.stitch_pointclouds_open3D(global_pointcloud, new_p4, 
            threshold = pc_threshold, max_iteration = max_iteration, with_plot = True,  with_transformation = True) 
        initial_scores.append(initial_evaluation)
        final_errors.append(np.sqrt(np.mean((p4-changed_pointcloud)**2)))
        import pdb; pdb.set_trace()
        '''
        # Compute final pose
        ## Use transformation matrix and carts to get results: multiply matrices
        
      plt.plot(initial_error_poses)
      plt.show()
      import pdb; pdb.set_trace()
      
      '''
      plt.plot(combined_score)
      plt.show()
      best_match = np.argmax(combined_score)
      best_rotation = vec_rotations[best_match]
      if best_rotation: previous_points = cum_sum_points[best_rotation-1]
      else: previous_points = 0
      best_pc_path = path_shape + name_id.format(best_rotation) + 'data_{}.npy'.format(perm[best_match]-previous_points)
      best_data = np.load(best_pc_path)
      best_image = best_data.item().get('image')
      plt.imshow(np.concatenate([new_image,best_image], axis=1)); plt.show()
      '''
      
