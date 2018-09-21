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
import pdb
global x_off, y_off, z_off
np.random.seed(seed=1)
import glob
  
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
  ## Count how many point clouds there are
  #rotations = [0,2,4,6, 8,10, 12, 14] ##,8,12]#[0,2,4,6,8,14,12] #range(2)
  rotations = [0,4,8,12]
  error_data = 1
  evaluate_random = True
  #name_id = 'tape_l=80_h=20_dx=10_dy=10_rot={}_09-03-2018/'; error_data = 2.0 #### alerta cal dividir per 2s
  #name_id = 'flashlight_l=80_h=20_dx=10_dy=10_rot={}_09-03-2018/'; error_data = 2.0#
  #name_id = 'mentos_l=80_h=20_dx=10_dy=10_rot={}_09-03-2018/'; error_data = 2.0#
  #name_id = 'scissors_l=80_h=20_dx=10_dy=10_rot={}_09-03-2018/'; error_data = 2.0 #### alerta cal dividir per 2s
  #name_id = 'brush_l=80_h=20_dx=10_dy=10_rot={}_09-03-2018/'; error_data = 2.0#
  path_shape = '/media/mcube/data/shapes_data/processed_pointclouds/'
  
  threshold = 0.1
  gripper_threshold = 2
  pc_threshold = 5
  max_iteration = 2000
  num_pointclouds = 0
  num_per_rotation = []
  num_selected = [5]#[5, 10, 20]
  vec_rotations = []
  
  for rotation in rotations:
    files = glob.glob(path_shape + name_id.format(rotation) + '/data_*.npy')
    num_files = len(files)
    print num_files
    num_pointclouds += int(num_files/error_data)
    vec_rotations += [rotation]*int(num_files/error_data)
  print num_pointclouds
  
  gripper_openings = []
  NN_vectors = []
  carts = []
  global_pointcloud = None
  p4_vect = []
  p3_vect = []
  img_vect = []
  depth_vect = []
  new_vec_rotations = []
  aux_num_pointclouds = copy.deepcopy(num_pointclouds)
  
  # Compute percentage of removals
  num_removes = []
  
  
  data_path = path_shape + name_id.format(0) + '/dataset_threshold={}_rotations={}.npy'.format(threshold*100,rotations)
  print data_path
  #import pdb; pdb.set_trace()
  if os.path.isfile(data_path):
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
  else:
    for i in range(num_pointclouds):    
      print i
      rotation = vec_rotations[i]
      
      pc_path = path_shape + name_id.format(rotation) + '/data_{}.npy'.format(np.mod(i,num_pointclouds/len(rotations)))
      print pc_path
      data = np.load(pc_path)
      depth_map = data.item().get('height')
      
      good_its = np.where(depth_map.flatten() >= threshold)
      p4 = data.item().get('p4')
      p4 = p4[good_its, :][0]
      if p4.shape[0] < 200: 
        print i
        aux_num_pointclouds -=1; 
        continue
      p4_vect.append(p4)
      p3 = data.item().get('p3')
      p3 = p3[good_its, :][0]
      p3_vect.append(p3)
      depth_vect.append(depth_map)
      new_vec_rotations.append(rotation)
      gripper_openings.append(data.item().get('gripper'))
      NN_vectors.append(data.item().get('vector'))
      img_vect.append(data.item().get('image'))
      carts.append(data.item().get('cart'))
      

      if len(p4) > 0:
        local_pointcloud = pointcloud_rotation(p4, rotation*np.pi/8.0)
        if global_pointcloud is None: global_pointcloud = local_pointcloud
        else: global_pointcloud = np.concatenate([global_pointcloud, local_pointcloud], axis=0)
    
    
    dataset = {}
    dataset['NN_vectors'] = NN_vectors
    dataset['aux_num_pointclouds'] = aux_num_pointclouds
    dataset['carts'] = carts
    dataset['depth_vect'] = depth_vect
    dataset['global_pointcloud'] = global_pointcloud
    dataset['gripper_openings'] = gripper_openings
    dataset['img_vect'] = img_vect
    dataset['name_id'] = name_id
    dataset['p3_vect'] = p3_vect
    dataset['p4_vect'] = p4_vect  
    dataset['new_vec_rotations'] = new_vec_rotations
    np.save( data_path, dataset)
  
  
  #loc.visualize_pointcloud(global_pointcloud)

  num_pointclouds = copy.deepcopy(aux_num_pointclouds)
  vec_rotations = copy.deepcopy(new_vec_rotations)
  
  num_averages = copy.deepcopy(num_pointclouds) #1000
  #num_removes.append(int(num_pointclouds*0.9)) #0.1 0.25 0.5 0.75 0.9
  num_removes = [1]
  for num_remove in num_removes:
    initial_errors_pcs = []
    final_errors_pcs = []
    final_errors_pcs_local = []
    initial_ICP_rmses = []
    final_ICP_rmses = []
    final_errors_pcs_w_local = []
    vect_best_points = []
    if num_remove == 1: num_averages = copy.deepcopy(num_pointclouds)
    for it_av in range(num_averages):
      print 'it_av: {}, from {}'.format(it_av, num_averages)
      print 'num_removes: 1'
      #Make sure it is not empty
      perm = np.random.permutation(num_pointclouds)
      while len(p3_vect[perm[-1]]) < 100: 
        perm = np.random.permutation(num_pointclouds)
      perm = np.delete(perm, np.where(perm == it_av))
      N = num_pointclouds - num_remove
      N_points = perm[:N]
      N_points.sort()
      if num_remove == 1: new_point = copy.deepcopy(it_av)
      else: new_point = perm[-1]
      #import pdb; pdb.set_trace()
      gripper_score = np.abs(np.array(gripper_openings)[N_points]-gripper_openings[new_point])
      img_score = []
      for it in N_points:
        img_score.append(1-scipy.spatial.distance.cosine(np.array(NN_vectors)[it], NN_vectors[new_point]))
      img_score = np.array(img_score)
      combined_score = img_score*(gripper_score < gripper_threshold)
      
      
      best_options = N_points[np.argsort(combined_score)[-num_selected[-1]:][::-1]]
      
      initial_scores = []
      final_scores = []
      final_errors = []
      initial_error_poses = []
      initial_ICP_rmse = []
      initial_local_ICP_rmse = []
      initial_error_pcs = []
      final_error_poses = []
      final_error_pcs = []
      final_error_pcs_local = []
      final_error_pcs_w_local = []
      final_ICP_rmse = []
      final_local_ICP_rmse = []
      initial_pos = []
      final_pos = []
      new_point_rotation = vec_rotations[new_point]
      
      # DO plot combined_score vs distance pcs
      #for it in N_points:
      print name_id
      print 'num_selected: ', num_selected      
      
      #Take the best one
      it = best_options[0]
      if evaluate_random: it = np.random.choice(N_points)
      
      best_rotation = vec_rotations[it]; cart = carts[it]
      relative_rotation = new_point_rotation - best_rotation
      rotated_p4  = pointcloud_rotation(p4_vect[new_point], relative_rotation*np.pi/8.0)
      print 'other cart:', cart
      print 'good cart:', carts[new_point]
      
      new_p4 = gr2w(p3_vect[new_point], cart[0:3], cart[-4:])  #There are only 3 possible cart[-4] --> simplify computations
      pc_error = rmse_pcs(rotated_p4, new_p4)
      print 'initial error: ', pc_error; initial_error_pcs.append(pc_error)
      if not evaluate_random:
        result_pointcloud, changed_pointcloud, transform_process, initial_evaluation = loc.stitch_pointclouds_open3D(p4_vect[it], new_p4, threshold = pc_threshold, max_iteration = max_iteration, with_plot = False,  with_transformation = True) 
        initial_ICP_rmse.append(initial_evaluation.inlier_rmse)
        final_ICP_rmse.append(transform_process.inlier_rmse)
        final_error_pcs_local.append(rmse_pcs(rotated_p4, changed_pointcloud))
        print 'second case: ', rmse_pcs(rotated_p4, changed_pointcloud)
      
      
        #loc.old_visualize_pointcloud( np.concatenate([rotated_p4, new_p4], axis=0))
        #plt.imshow(np.concatenate([img_vect[new_point],img_vect[it]], axis=1)); plt.show()
        #import pdb; pdb.set_trace()
        #plt.imshow(np.concatenate([img_vect[new_point],img_vect[it]], axis=1)); plt.show()
        # Take N good ones
        #import pdb; pdb.set_trace()
        rotated_global_pointcloud = pointcloud_rotation(global_pointcloud, - best_rotation*np.pi/8.0)
        mean_new_p4 = np.mean(new_p4, axis = 0)
        distance_points = np.linalg.norm(np.subtract(rotated_global_pointcloud, mean_new_p4), axis=1)
        num_points_pcs = []#[10000, 50000, 100000, 500000]
        for num_points_pc in num_points_pcs:
          closest_pointcloud = rotated_global_pointcloud[np.argsort(distance_points)[:num_points_pc],:]
          result_pointcloud, changed_pointcloud, transform_process, initial_evaluation = loc.stitch_pointclouds_open3D(closest_pointcloud, new_p4,   threshold = pc_threshold, max_iteration = max_iteration, with_plot = False,  with_transformation = True) 
          final_error_pcs_local.append(rmse_pcs(rotated_p4, changed_pointcloud))
          initial_ICP_rmse.append(initial_evaluation.inlier_rmse)
          final_ICP_rmse.append(transform_process.inlier_rmse)
          print 'third case, {}: '.format(num_points_pc), rmse_pcs(rotated_p4, changed_pointcloud)
        
        #import pdb; pdb.set_trace()
        for num_sel in num_selected:
          local_global_pointcloud = None
          for it_opt in best_options[0:num_sel]:
            local_pointcloud = pointcloud_rotation(p4_vect[it_opt], (vec_rotations[it_opt] - best_rotation)*np.pi/8.0)
            if local_global_pointcloud is None: local_global_pointcloud = local_pointcloud
            else: local_global_pointcloud = np.concatenate([local_global_pointcloud, local_pointcloud], axis=0)

          result_pointcloud, changed_pointcloud, transform_process, initial_evaluation = loc.stitch_pointclouds_open3D(local_global_pointcloud, new_p4,   threshold = pc_threshold, max_iteration = max_iteration, with_plot = False,  with_transformation = True) 
          final_error_pcs_local.append(rmse_pcs(rotated_p4, changed_pointcloud))
          initial_ICP_rmse.append(initial_evaluation.inlier_rmse)
          final_ICP_rmse.append(transform_process.inlier_rmse)
          print 'third case, {}: '.format(num_sel), rmse_pcs(rotated_p4, changed_pointcloud)
      
      vect_best_points.append(it)
      initial_errors_pcs.append(pc_error)
      final_errors_pcs_local.append(final_error_pcs_local)
      initial_ICP_rmses.append(initial_ICP_rmse)
      final_ICP_rmses.append(final_ICP_rmse)
      print np.mean(initial_errors_pcs), np.median(initial_errors_pcs)
      print 'Mean: ', np.mean(initial_errors_pcs), np.mean(np.array(final_errors_pcs_local), axis = 0), 
      print 'Median: ', np.median(initial_errors_pcs), np.median(np.array(final_errors_pcs_local), axis = 0)
      
      aux_data_path = copy.deepcopy(data_path)
      if evaluate_random:
        data_path = data_path[:-4] + '_random.npy'
      np.save(data_path[:-4] + 'initial_errors', initial_errors_pcs)
      np.save(data_path[:-4] + 'final_errors', final_errors_pcs_local)
      np.save(data_path[:-4] + 'initial_ICP_errors', initial_ICP_rmses)
      np.save(data_path[:-4] + 'final_ICP_errors', final_ICP_rmses)
      np.save(data_path[:-4] + 'num_av', it_av)
      np.save(data_path[:-4] + 'num_removed', num_remove)
      if not evaluate_random: np.save(data_path[:-4] + 'num_points_pcs', num_points_pcs)
      np.save(data_path[:-4] + 'num_selected', num_selected)
      np.save(data_path[:-4] + 'best_points', vect_best_points)
      data_path = copy.deepcopy(aux_data_path)
      
      
      #import pdb; pdb.set_trace()
      
      # Compute final pose
      ## Use transformation matrix and carts to get results: multiply matrices
      '''
      #plt.plot(initial_error_poses)
      plt.plot(initial_error_pcs)
      plt.show()
      import pdb; pdb.set_trace()
      
      
      plt.plot(combined_score)
      plt.plot(img_score)
      plt.show()
      best_match = N_points[np.argmax(combined_score)]
      plt.imshow(np.concatenate([img_vect[new_point],img_vect[best_match]], axis=1)); plt.show()
      

      best_match = N_points[np.argmin(initial_error_pcs)]
      print best_match, initial_error_pcs[np.argmin(initial_error_pcs)]
      plt.imshow(np.concatenate([img_vect[new_point],img_vect[best_match]], axis=1)); plt.show()
      import pdb; pdb.set_trace()
      #'''
    import pdb; pdb.set_trace()
