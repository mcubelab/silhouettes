import numpy as np
from control_robot import ControlRobot
import time, pdb
import matplotlib.pyplot as plt
import copy
from training_touch import training_touch
import os
if __name__ == "__main__":
    num_data = 2000
    
    #shapes = ['sphere', 'semicone_1', 'semicone_2', 'hollowcone_1', 'hollowcone_2', 'semipyramid', 'stamp']
    shapes = ['sphere', 'semicone_2', 'hollowcone_1', 'hollowcone_2', 'semipyramid', 'stamp']
    
    gs_ids = [1,2]
    num_data = 500
    num_test_data = 100
    for gs_id in gs_ids:
      for shape in shapes:
        if () or ('stamp' in shape):
            rotations = 180/8.0*np.arange(4)
            num_data = num_data/4
            num_test_data = num_test_data/4
        else:
            rotations = [0]
        for rotation in rotations:
            data_type = '/media/mcube/data/shapes_data/raw/' + shape + '_08-17-2018_gs_id={}_rot={}/'.format(gs_id, rotation)
            data_test_type = '/media/mcube/data/shapes_data/raw/' +shape + '_08-17-2018_test_gs_id={}_rot={}/'.format(gs_id, rotation)
            raw_input('Collecting data for: '+ data_type +'. Ready?')
            if not os.path.exists(data_type): os.makedirs(data_type)
            if not os.path.exists(data_test_type): os.makedirs(data_test_type)
            collected_data = sum(os.path.isdir(data_type + i) for i in os.listdir(data_type))
            collected_test_data = sum(os.path.isdir(data_test_type + i) for i in os.listdir(data_test_type))
            print data_test_type
            print collected_test_data
            while num_data > collected_data:
              try:
                training_touch(experiment_name = data_type, object_angle = rotation, number_of_points = num_data, gs_id = gs_id, last_touch = collected_data)
                
              except: pass
              collected_data = sum(os.path.isdir(data_type + i) for i in os.listdir(data_type))
            while num_test_data > collected_test_data:
              try:
                training_touch(experiment_name = data_test_type, object_angle = rotation, number_of_points = num_test_data, gs_id = gs_id, last_touch = collected_test_data)
              except: pass
              collected_test_data = sum(os.path.isdir(data_test_type + i) for i in os.listdir(data_test_type))
    now_test = False
    while now_test != 'y':              
      now_test = raw_input('All the automatic data is collected. Now collect the test images (keys, ...). OK?[y/n]')
