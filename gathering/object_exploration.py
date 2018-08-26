import numpy as np
from control_robot import ControlRobot
import gripper
import rospy

if __name__ == "__main__":
    
    # Enter object main dimensions in [mm]
    length = 40
    height = 20
    starting_angle = 0

    # Enter minimum step distance (d) in [mm]
    d = 10
    
    # Experiment name
    shape = 'big_semicone'
    experiment_folder = "/media/mcube/data/shapes_data/object_exploration/"
    experiment_name = experiment_folder + shape + "_l={}_h={}_d={}_rot={}".format(length, height,d, starting_angle)

    # Enter these other params
    cr = ControlRobot(gs_ids=[1, 2], force_list=[5])
    gripper_rotations = ["0", "+20", "-20"]

    # Keep track touches
    last_touch = 0
    # IMPORTANT NOTE: Make sure you start at a save position ()
    gripper_save_turning_pos = 833.6, 375., 862., 0., np.sqrt(0.5), -np.sqrt(0.5), 0.
    gripper_turned_pos = 833.6, 375., 862., 0., -0.716, -0.69, 0.
    for gripper_rotation in gripper_rotations:
        if gripper_rotation == "0":
            start_pos = 830.6, 372.88, 660., 0., np.sqrt(0.5), -np.sqrt(0.5), 0.
            restart_pos = 820, 380.65, 660., 0., -np.sqrt(0.5), -np.sqrt(0.5), 0.
        elif gripper_rotation == "+20":
            start_pos = 712.3, 377.45, 642.7, 0.10982, -0.6852, 0.71095, -0.1147  #ok            
            restart_pos = 707., 388.66, 640.7, 0.11314, 0.71263, 0.68473, 0.10245  # ok
        elif gripper_rotation == "-20":
            start_pos = 953.6, 374.88, 637.5, 0.11982, 0.69448, -0.70019, -0.11514  # ok
            restart_pos = 955.6, 378.26, 636.5, 0.1289, -0.71015, -0.67992, 0.1295  # ok
        
        gripper.open()
        # We perform the movement
        print 'RELOCATING...'
        # IF comming from ARC
        #cart = gripper_turned_pos
        #cr.set_cart_mm(cart)
        #cart = gripper_save_turning_pos
        #cr.set_cart_mm(cart)
        cart = start_pos
        cr.set_cart_mm(cart)
        rospy.sleep(2)
        cr.move_cart_mm(10, 0, height/2.0-10)
        rospy.sleep(2)
        print 'TOUCH MOTION 1'
        num_rep_x = int(float(length)/float(d))
        num_rep_y = int(float(height)/float(d))
        movement_list = []
        for i in range(num_rep_y):
            sgn = 2*(i%2) - 1
            for j in range(num_rep_x):
                movement_list += [[sgn*d, 0, 0]]
            movement_list += [[0, 0, -d]]
        #import pdb; pdb.set_trace()
        cr.perfrom_experiment(experiment_name=experiment_name, movement_list=movement_list, last_touch = last_touch, pix_threshold = 0)
        last_touch += len(movement_list) 


        # We perform the movement
        print 'RELOCATING...'
        cart = gripper_save_turning_pos
        cr.set_cart_mm(cart)
        cart = gripper_turned_pos
        cr.set_cart_mm(cart)
        cart = restart_pos
        cr.set_cart_mm(cart)
        rospy.sleep(2)
        cr.move_cart_mm(-10, 0, height/2.0-10)
        rospy.sleep(2)
        print 'TOUCH MOTION 2'
        num_rep_x = int(float(length)/float(d))
        num_rep_y = int(float(height)/float(d))
        movement_list = []
        for i in range(num_rep_y):
            sgn = 2*(i%2) - 1
            for j in range(num_rep_x):
                movement_list += [[sgn*d, 0, 0]]
            movement_list += [[0, 0, -d]]

        cr.perfrom_experiment(
            experiment_name=experiment_name,
            movement_list=movement_list,
            last_touch = last_touch, pix_threshold = 0)
        
        last_touch += len(movement_list) 
        print 'RELOCATING...'
        cart = gripper_turned_pos
        cr.set_cart_mm(cart)

        cart = gripper_save_turning_pos
        cr.set_cart_mm(cart)
        

