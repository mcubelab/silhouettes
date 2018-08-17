import numpy as np
from control_robot import ControlRobot
import time, pdb
import matplotlib.pyplot as plt
import copy

def training_touch(experiment_name = '/media/mcube/data/shapes_data/raw/test', object_angle = '0', number_of_points = 10, gs_id = 2, last_touch = 0):
    
    #experiment_name = "/media/mcube/data/shapes_data/raw/cross_stamp_08-15-2018_gs2_rot=" + str(object_angle)
    # Enter object main dimensions in [mm]
    length = 40 # distance from the magnetic base to the top
    # Enter area to area to explore uniformly
    height = 8
    width = 8
    

    # Enter these other params
    cr = ControlRobot(gs_ids=[gs_id], force_list=[20])

    # IMPORTANT NOTE: Make sure you start at a save position

    if gs_id == 1:
        start_cart = 895-length, 364., 660., -0.0205, -0.00833, -0.9997, 0.
    elif gs_id == 2:
        start_cart = 900-length, 387., 660., 0.01, 0.9999, -0.0102, 0.008

    # We generate all the random offsets from the center:
    rnd_x = np.random.uniform(-width/2, width/2, size=number_of_points)
    rnd_y = np.random.uniform(-height/2, height/2, size=number_of_points)

    # We compute all the movememts
    movement_list = [(0, 0, 0)]
    prev_x = 0
    prev_y = 0
    for x, y in zip(rnd_x, rnd_y):
        movement = [0, 0, 0]
        movement[0] = 0
        movement[1] = x - prev_x
        movement[2] = y - prev_y
        movement_list.append(movement)
        prev_x = copy.deepcopy(x)
        prev_y = copy.deepcopy(y)
        
    # We perform the movement
    print 'RELOCATING...'
    cr.set_cart_mm(start_cart)
    time.sleep(5)
    #
    print 'TOUCHING...'
    cr.perfrom_experiment(
        experiment_name=experiment_name,
        movement_list=movement_list,
        save_only_picture=False,
        last_touch = last_touch
    )


if __name__ == "__main__":
    num_data = 2000
    training_touch(experiment_name = '/media/mcube/data/shapes_data/raw/test', object_angle = '0', number_of_points = num_data, gs_id = 2)
