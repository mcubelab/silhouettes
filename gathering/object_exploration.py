import numpy as np
from control_robot import ControlRobot
import gripper
import rospy

if __name__ == "__main__":
    
    # Enter object main dimensions in [mm]
    length = 100
    height = 20
    starting_angle = 4
    # Enter minimum step distance (d) in [mm]
    dx = 2
    dz = 10
    
    cr = ControlRobot(gs_ids=[2], force_list=[5])
    gripper_rotations = ["0", "+20", "-20"]
    #for gripper_rotation in gripper_rotations:

    
    # Experiment name
    shape = 'flashlight'#'rectangle'  ##'cilinder'#'big_semicone'
    experiment_folder = "/media/mcube/data/shapes_data/object_exploration/"
    experiment_name = experiment_folder + shape + "_l={}_h={}_dx={}_dy={}_rot={}_debug".format(length, height,dx,dz, starting_angle)    
    

    # Keep track touches
    last_touch = 0
    # IMPORTANT NOTE: Make sure you start at a save position ()
    gripper_save_turning_pos = 833.6, 375., 862., 0., np.sqrt(0.5), -np.sqrt(0.5), 0.
    gripper_turned_pos = 833.6, 375., 862., 0., -0.716, -0.69, 0.
    for gripper_rotation in gripper_rotations:
        if gripper_rotation == "0":
            start_pos = 830.6, 373.13 , 660., 0., np.sqrt(0.5), -np.sqrt(0.5), 0.
            restart_pos = 830.6, 380.65, 660., 0., -np.sqrt(0.5), -np.sqrt(0.5), 0.
        elif gripper_rotation == "+20":
            start_pos = 701.7, 372.96, 641.13, 0.122787803968973,  -0.696364240320019,   0.696364240320019,  -0.122787803968973         # 0.10982, -0.6852, 0.71095, -0.1147  #ok    ## 
            restart_pos = 707., 388.66, 640.7, 0.122787803968973,  0.696364240320019,   0.696364240320019,  0.122787803968973 # 0.11314, 0.71263, 0.68473, 0.10245 
        elif gripper_rotation == "-20":
            start_pos = 958.1, 372.88, 634.5, 0.122787803968973,  0.696364240320019,   -0.696364240320019,  -0.122787803968973 #  0.11982, 0.69448, -0.70019, -0.11514  
            restart_pos = 965.6, 378.26, 636.5, 0.122787803968973,  -0.696364240320019,   -0.696364240320019,  0.122787803968973 # 0.1289, -0.71015, -0.67992, 0.1295 
        
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
        cr.move_cart_mm(15, 0, height/2.0)
        rospy.sleep(2)
        print 'TOUCH MOTION 1'
        num_rep_x = int(float(length)/float(dx))
        num_rep_y = int(float(height)/float(dz))
        movement_list = []
        for i in range(num_rep_y):
            sgn = 2*(i%2) - 1
            for j in range(num_rep_x):
                movement_list += [[sgn*dx, 0, 0]]
            if i < num_rep_y-1:
                movement_list += [[0, 0, -dz]]
        #import pdb; pdb.set_trace()
        cr.perfrom_experiment(experiment_name=experiment_name, movement_list=movement_list, last_touch = last_touch, pix_threshold = 0)
        last_touch += len(movement_list)+1 

        '''
        # We perform the movement
        print 'RELOCATING...'
        cart = gripper_save_turning_pos
        cr.set_cart_mm(cart)
        cart = gripper_turned_pos
        cr.set_cart_mm(cart)
        cart = restart_pos
        cr.set_cart_mm(cart)
        rospy.sleep(5)
        cr.move_cart_mm(-10, 0, height/2.0-10)
        rospy.sleep(5)
        print 'TOUCH MOTION 2'
        num_rep_x = 1#int(float(length)/float(d))
        num_rep_y = 2# int(float(height)/float(d))
        movement_list = []
        for i in range(num_rep_y):
            sgn = 2*(i%2) - 1
            for j in range(num_rep_x):
                movement_list += [[sgn*d, 0, 0]]
            if i < num_rep_y-1:
                movement_list += [[0, 0, -d]]

        cr.perfrom_experiment(
            experiment_name=experiment_name,
            movement_list=movement_list,
            last_touch = last_touch, pix_threshold = 0)
        
        last_touch += len(movement_list)+1 
        print 'RELOCATING...'
        cart = gripper_turned_pos
        cr.set_cart_mm(cart)

        cart = gripper_save_turning_pos
        cr.set_cart_mm(cart)
        '''

