import numpy as np
from control_robot import ControlRobot
import gripper
import rospy

'''

    if 'cilinder' in shape:
        length = 100  #60
        height = 30   #25
        initial_z = 10
        initial_x = 40
    if 'rectangle' in shape:
        length = 70  #60
        height = 30   #25
        initial_z = 10
        initial_x = 12
    if 'big_semicone' in shape:
        length = 110  #60
        height = 30   #25
        initial_z = 10
        initial_x = 30
    if 'double' in shape:
        length = 110  #60
        height = 30   #25
        initial_z = 10
        initial_x = 18
    if 'pyramid' in shape:
        length = 90  #60
        height = 30   #25
        initial_z = 10
        initial_x = 30
'''


if __name__ == "__main__":
    
    
    only_show = False
    only_borders = False
    grasp_test = True
    # Enter object main dimensions in [mm]
    length = 1000
    height = 20
    starting_angle = 14
    # Enter minimum step distance (d) in [mm]
    dx = 5
    dz = 20

    cr = ControlRobot(gs_ids=[1], force_list=[5])
    gripper_rotations = ["0", "+20", "-20"]
    if grasp_test: gripper_rotations = ["0"] #, "+20", "-20"]
    #for gripper_rotation in gripper_rotations:

    
    # Experiment name
    shape = 'test_grasps_location'#'big_semicone'
    experiment_folder = "/media/mcube/data/shapes_data/object_exploration/"
    experiment_name = experiment_folder + shape + "_l={}_h={}_dx={}_dy={}_rot={}_09-09-2018".format(length, height,dx,dz, starting_angle)    
    
    initial_z = 10
    initial_x = 20
    if 'flashlight' in shape:
        length = 170  #60
        height = 20   #25
        initial_z = 10
        initial_x = 40
    if 'tape' in shape:
        length = 70  #60
        height = 60   #25
        initial_z = 30
        initial_x = 16
    if 'scissors' in shape:
        length = 110  #60
        height = 70   #25
        initial_z = 30
        initial_x = 12
    if 'brush' in shape:
        length = 140  #60
        height = 20   #25
        initial_z = 10
        initial_x = 12
    if 'mentos' in shape:
        length = 70  #60
        height = 20   #25
        initial_z = 10
        initial_x = 15
    

    # Keep track touches
    last_touch = 0
    # IMPORTANT NOTE: Make sure you start at a save position ()
    gripper_save_turning_pos = 833.6, 375., 862., 0., np.sqrt(0.5), -np.sqrt(0.5), 0.
    gripper_turned_pos = 833.6, 375., 862., 0., -0.716, -0.69, 0.
    if not grasp_test: cr.move_cart_mm(0, 0, 200)
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
        
        
        # start_pos = 994, -400, 600, 0., np.sqrt(0.5), -np.sqrt(0.5), 0.
        gripper.open()
        # We perform the movement
        print 'RELOCATING...'
        # IF comming from ARC
        #cart = gripper_turned_pos
        #cr.set_cart_mm(cart)
        #cart = gripper_save_turning_pos
        #cr.set_cart_mm(cart)
        if not grasp_test:
            cart = start_pos
            cr.set_cart_mm(cart)
            rospy.sleep(2)
            cr.move_cart_mm(initial_x,0,initial_z)
            rospy.sleep(2)
        
        import pdb; pdb.set_trace()
        print 'TOUCH MOTION 1'
        num_rep_x = int(float(length)/float(dx))
        num_rep_y = int(float(height)/float(dz))
        movement_list = []
        for i in range(num_rep_y):
            sgn = 2*(i%2) - 1
            for j in range(num_rep_x):
                if not grasp_test:movement_list += [[sgn*dx, 0, 0]]
                else:movement_list += [[0, 0, 0]]
                if only_show and not only_borders:
                    cr.move_cart_mm(sgn*dx, 0, 0)
                    rospy.sleep(0.1)
            if i < num_rep_y-1:
                movement_list += [[0, 0, -dz]]
                if only_show:
                    if only_borders: 
                        cr.move_cart_mm(sgn*dx*num_rep_x, 0, 0)
                        rospy.sleep(0.5)
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

if grasp_test: assert(False)

cart = gripper_save_turning_pos
cr.set_cart_mm(cart)
cart = gripper_turned_pos
cr.set_cart_mm(cart)
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
        cart = restart_pos
        cr.set_cart_mm(cart)
        rospy.sleep(2)
        cr.move_cart_mm(initial_x-10,0,initial_z)
        rospy.sleep(2)
        
        print 'TOUCH MOTION 2'
        num_rep_x = int(float(length)/float(dx))
        num_rep_y = int(float(height)/float(dz))
        movement_list = []
        for i in range(num_rep_y):
            sgn = 2*(i%2) - 1
            for j in range(num_rep_x):
                movement_list += [[sgn*dx, 0, 0]]
                if only_show and not only_borders:
                    cr.move_cart_mm(sgn*dx, 0, -0)
                    rospy.sleep(0.5)
            if i < num_rep_y-1:
                movement_list += [[0, 0, -dz]]
                if only_show:
                    if only_borders: 
                        cr.move_cart_mm(sgn*dx*num_rep_x, 0, 0)
                        rospy.sleep(0.5)
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

