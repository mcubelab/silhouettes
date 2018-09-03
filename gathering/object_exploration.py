import numpy as np
from control_robot import ControlRobot
import gripper
import rospy

if __name__ == "__main__":
    
    
    only_show = False
    # Enter object main dimensions in [mm]
    length = 140
    height = 50
    starting_angle = 12
    # Enter minimum step distance (d) in [mm]
    dx = 15
    dz = 20
    
    cr = ControlRobot(gs_ids=[1,2], force_list=[20])
    gripper_rotations = ["0", "+20", "-20"]
    gripper_rotations = ["0"] #, "+20", "-20"]
    #for gripper_rotation in gripper_rotations:

    
    # Experiment name
    shape = 'scissors'#'rectangle'  ##'cilinder'#'big_semicone'
    experiment_folder = "/media/mcube/data/shapes_data/object_exploration/"
    experiment_name = experiment_folder + shape + "_l={}_h={}_dx={}_dy={}_rot={}_debug".format(length, height,dx,dz, starting_angle)    
    experiment_name = experiment_folder + shape + "_l={}_h={}_dx={}_dy={}_rot={}_rotated".format(length, height,dx,dz, starting_angle)    
    initial_z = 0
    if 'scissors' in shape:
        initial_z = 25
    if 'tape' in shape:
        initial_z = 15
    if 'mentos' in shape:
        initial_z = 10
    if 'baseball' in shape:
        initial_z = 25
    if 'pen' in shape:
        initial_z = 0
    if 'brush' in shape:
        initial_z = 25
    if 'cube' in shape:
        initial_z = 10
    if 'big_cube' in shape:
        initial_z = 25
        

    # Keep track touches
    last_touch = 0
    # IMPORTANT NOTE: Make sure you start at a save position ()
    gripper_save_turning_pos = 833.6, 375., 862., 0., np.sqrt(0.5), -np.sqrt(0.5), 0.
    gripper_turned_pos = 833.6, 375., 862., 0., -0.716, -0.69, 0.
    cr.move_cart_mm(0, 0, 200)
    rospy.sleep(2)
    for gripper_rotation in gripper_rotations:
        if gripper_rotation == "0":
            start_pos = 838.6, 372.13 , 657., 0., np.sqrt(0.5), -np.sqrt(0.5), 0.
            restart_pos = 838.6, 372.13 , 657., 0., -np.sqrt(0.5), -np.sqrt(0.5), 0.
        elif gripper_rotation == "+20":
            start_pos = 710.9, 372.13, 636.63, 0.122787803968973,  -0.696364240320019,   0.696364240320019,  -0.122787803968973         # 0.10982, -0.6852, 0.71095, -0.1147  #ok    ## 
            restart_pos = 710.9, 372.13, 636.63, 0.122787803968973,  0.696364240320019,   0.696364240320019,  0.122787803968973 # 0.11314, 0.71263, 0.68473, 0.10245 
        elif gripper_rotation == "-20":
            start_pos = 965.1, 372.13, 631.5, 0.122787803968973,  0.696364240320019,   -0.696364240320019,  -0.122787803968973 #  0.11982, 0.69448, -0.70019, -0.11514  
            restart_pos = 965.1, 372.13, 631.5, 0.122787803968973,  -0.696364240320019,   -0.696364240320019,  0.122787803968973 # 0.1289, -0.71015, -0.67992, 0.1295 
        
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
        cr.move_cart_mm(25,0,initial_z)
        rospy.sleep(2)
        
        #import pdb; pdb.set_trace()
        print 'TOUCH MOTION 1'
        num_rep_x = int(float(length)/float(dx))
        num_rep_y = int(float(height)/float(dz))
        movement_list = []
        for i in range(num_rep_y):
            sgn = 2*(i%2) - 1
            for j in range(num_rep_x):
                movement_list += [[sgn*dx, 0, 0]]
                if only_show:
                    cr.move_cart_mm(sgn*dx, 0, 0)
                    rospy.sleep(0.2)
            if i < num_rep_y-1:
                movement_list += [[0, 0, -dz]]
                if only_show:
                    cr.move_cart_mm(0, 0, -dz)
                    rospy.sleep(0.5)
        #import pdb; pdb.set_trace()
        if only_show:
            cr.set_cart_mm(cart)
            rospy.sleep(2)
        else:
            cr.perfrom_experiment(experiment_name=experiment_name, movement_list=movement_list, last_touch = last_touch, pix_threshold = 0)
            last_touch += len(movement_list)+1 
        cr.move_cart_mm(0,0,200)
        rospy.sleep(2)


################ CHANGE ORI
for gripper_rotation in gripper_rotations:
        if gripper_rotation == "0":
            start_pos = 838.6, 372.13 , 657., 0., np.sqrt(0.5), -np.sqrt(0.5), 0.
            restart_pos = 838.6, 372.13 , 657., 0., -np.sqrt(0.5), -np.sqrt(0.5), 0.
        elif gripper_rotation == "+20":
            start_pos = 710.9, 372.13, 636.63, 0.122787803968973,  -0.696364240320019,   0.696364240320019,  -0.122787803968973         # 0.10982, -0.6852, 0.71095, -0.1147  #ok    ## 
            restart_pos = 710.9, 372.13, 636.63, 0.122787803968973,  0.696364240320019,   0.696364240320019,  0.122787803968973 # 0.11314, 0.71263, 0.68473, 0.10245 
        elif gripper_rotation == "-20":
            start_pos = 965.1, 372.13, 631.5, 0.122787803968973,  0.696364240320019,   -0.696364240320019,  -0.122787803968973 #  0.11982, 0.69448, -0.70019, -0.11514  
            restart_pos = 965.1, 372.13, 631.5, 0.122787803968973,  -0.696364240320019,   -0.696364240320019,  0.122787803968973 # 0.1289, -0.71015, -0.67992, 0.1295 



        # We perform the movement
        print 'RELOCATING...'
        cart = gripper_save_turning_pos
        cr.set_cart_mm(cart)
        cart = gripper_turned_pos
        cr.set_cart_mm(cart)
        cart = restart_pos
        cr.set_cart_mm(cart)
        rospy.sleep(2)
        cr.move_cart_mm(5,0,initial_z)
        rospy.sleep(2)
        
        print 'TOUCH MOTION 2'
        num_rep_x = int(float(length)/float(dx))
        num_rep_y = int(float(height)/float(dz))
        movement_list = []
        for i in range(num_rep_y):
            sgn = 2*(i%2) - 1
            for j in range(num_rep_x):
                movement_list += [[sgn*dx, 0, 0]]
                if only_show:
                    cr.move_cart_mm(sgn*dx, 0, -0)
                    rospy.sleep(0.2)
            if i < num_rep_y-1:
                movement_list += [[0, 0, -dz]]
                if only_show:
                    cr.move_cart_mm(0, 0, -dz)
                    rospy.sleep(0.5)
        if only_show:
            cr.set_cart_mm(cart)
            rospy.sleep(2)
        else: 
            cr.perfrom_experiment(
                experiment_name=experiment_name,
                movement_list=movement_list,
                last_touch = last_touch, pix_threshold = 0)
        
        last_touch += len(movement_list)+1 
        
        cr.move_cart_mm(0,0,200)
        rospy.sleep(2)

print 'RELOCATING...'
cart = gripper_turned_pos
cr.set_cart_mm(cart)

cart = gripper_save_turning_pos
cr.set_cart_mm(cart)
#'''

