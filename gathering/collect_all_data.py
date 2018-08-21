import numpy as np
from control_robot import ControlRobot
import time, pdb
import matplotlib.pyplot as plt
import copy
from training_touch import training_touch
import os
from data_collector import DataCollector
import datetime
if __name__ == "__main__":
    num_data = 2000
    
    #shapes = ['sphere', 'semicone_1', 'semicone_2', 'hollowcone_1', 'hollowcone_2', 'semipyramid', 'stamp']
    shapes = ['semicone_1', 'sphere', 'semicone_2', 'hollowcone_1', 'hollowcone_2', 'semipyramid_2', 'stamp']
    
    
    gs_ids = [2,1]
    num_data = 500
    num_test_data = 100
    date = datetime.datetime.today().strftime('%m-%d-%Y')
    for gs_id in gs_ids:
      for shape in shapes:
        if ('semipyramid' in shape) or ('stamp' in shape):
            rotations = 180/8.0*np.arange(4)
            num_data = num_data/4
            num_test_data = num_test_data/4
        else:
            rotations = [0]
            num_data = 500
            num_test_data = 100
        for j, rotation in enumerate(rotations):
            data_type = '/media/mcube/data/shapes_data/raw/' + shape + '_{}_gs_id={}_rot={}/'.format(date,gs_id, j)
            data_test_type = '/media/mcube/data/shapes_data/raw/' +shape + '_{}_test_gs_id={}_rot={}/'.format(date,gs_id, j)
            start_again = False
            while start_again != 'y':
              start_again = raw_input('Collecting data for: '+ data_type +'. Ready?')
            if not os.path.exists(data_type): os.makedirs(data_type)
            if not os.path.exists(data_test_type): os.makedirs(data_test_type)
            collected_data = sum(os.path.isdir(data_type + i) for i in os.listdir(data_type))-1
            collected_test_data = sum(os.path.isdir(data_test_type + i) for i in os.listdir(data_test_type))-1 #remove the air folder
            print data_test_type
            print collected_test_data
            while num_data > collected_data:
              try:
                training_touch(experiment_name = data_type, object_angle = rotation, number_of_points = num_data, gs_id = gs_id, last_touch = collected_data)
                
              except: pass
              collected_data = sum(os.path.isdir(data_type + i) for i in os.listdir(data_type))-1
            while num_test_data > collected_test_data:
              try:
                training_touch(experiment_name = data_test_type, object_angle = rotation, number_of_points = num_test_data, gs_id = gs_id, last_touch = collected_test_data)
              except: pass
              collected_test_data = sum(os.path.isdir(data_test_type + i) for i in os.listdir(data_test_type))-1
      now_test = False
      print 'All the automatic data is collected. Now collect the test images (keys, bolts, also all shapes again). '
      dc = DataCollector(only_one_shot=True, save_path='/media/mcube/data/shapes_data/raw/test_{}_gs_id={}/'.format(date,gs_id))
      ite = 0
      while now_test != 'y':              
        dc.it = ite
        if gs_id == 1:
          dc.get_data(get_cart=False, get_gs1=True, get_gs2=False, get_wsg=False, iteration=0)
        else:
          dc.get_data(get_cart=False, get_gs1=False, get_gs2=True, get_wsg=False, iteration=0)
        time.sleep(1)
        ite += 1
        now_test = raw_input('Are you done?[y/n]')
            
        
