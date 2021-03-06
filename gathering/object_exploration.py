import numpy as np
from control_robot import ControlRobot

if __name__ == "__main__":
    experiment_name = "/media/mcube/data/shapes_data/pos_calib/test2"

    # Enter object main dimensions in [mm]
    length = 30
    height = 20
    starting_angle = 0

    # Enter minimum step distance (d) in [mm]
    d = 10

    # Enter these other params
    cr = ControlRobot(gs_ids=[2], force_list=[20])
    do_touch_motion = False
    gripper_rotation = "0"  # "0" "+30" "-30"  # TODO: Implement different angles

    # IMPORTANT NOTE: Make sure you start at a save position
    start_pos = 840.6, 373., 660., 0., 0.7045, -0.7097, 0.
    gripper_save_turning_pos = 833.6, 373., 862., 0., 0.7045, -0.7097, 0.
    gripper_turned_pos = 833.6, 373., 862., 0., -0.716, -0.69, 0.
    restart_pos = 840.6, 373., 660.., 0., -0.716, -0.69, 0.


    # We perform the movement
    print 'RELOCATING...'
    cart = start_pos
    cr.set_cart_mm(cart)
    cr.move_cart_mm(0, 0, height/2)

    print 'TOUCH MOTION 1'
    num_rep_x = int(float(length)/float(d))
    num_rep_y = int(float(height)/float(d))
    movement_list = []
    for i in range(num_rep_y):
        sgn = 2*(i%2) - 1
        for j in range(num_rep_x):
            movement_list += [[sgn*d, 0, 0]]
        movement_list += [[0, 0, -d]]

    cr.perfrom_experiment(
        experiment_name=experiment_name + '_1',
        movement_list=movement_list
    )


    # We perform the movement
    print 'RELOCATING...'
    cart = gripper_save_turning_pos
    cr.set_cart_mm(cart)
    cart = gripper_turned_pos
    cr.set_cart_mm(cart)
    cart = restart_pos
    cr.set_cart_mm(cart)

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
        experiment_name=experiment_name + '_2',
        movement_list=movement_list
    )
