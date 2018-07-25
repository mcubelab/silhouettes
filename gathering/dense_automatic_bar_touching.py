import numpy as np
from control_robot import ControlRobot

if __name__ == "__main__":
    experiment_name = "/media/mcube/data/shapes_data/pos_calib/full_bar_f=20_d=5_v1"
    do_touch_motion = True


    cr = ControlRobot(gs_ids=[2], force_list=[20])
    gripper_rotation = "0"  # "0" "+30" "-30"

    if gripper_rotation == "0":
        front_quat = 0.0, 0.7111, 0.7031, -0.0001
        side_quat = 0.0, -0.00872, -1., 0.
        back_quat = 0.0, 0.6846, -0.72888, 0.
    elif gripper_rotation == "+30":  # TODO: Put the right numbers here
        front_quat = 0.0, 0.7111, 0.7031, -0.0001
        side_quat = 0.0, -0.00872, -1., 0.
        back_quat = 0.0, 0.6846, -0.72888, 0.
    elif gripper_rotation == "-30":  # TODO: Put the right numbers here
        front_quat = 0.0, 0.7111, 0.7031, -0.0001
        side_quat = 0.0, -0.00872, -1., 0.
        back_quat = 0.0, 0.6846, -0.72888, 0.

    ###################### Front
    print 'RELOCATING...'
    cart = 674.69, 644.55, 748.03 + front_quat
    cr.set_cart_mm(cart)

    print 'TOUCH MOTION'
    if do_touch_motion:
        d = 5
        original_d = 10
        num_rep = 4*6*int(original_d/d)
        num_rep_2 = 2*int(original_d/d)
        movement_list = []
        for i in range(num_rep):
            movement_list += [[-d, 0, 0]]
        for i in range(num_rep_2):
            movement_list += [[0, 0, -d]]
        for i in range(num_rep):
            movement_list += [[d, 0, 0]]

        cr.perfrom_experiment(
            experiment_name=experiment_name + '_front',
            movement_list=movement_list
        )

    ###################### Side
    print 'RELOCATING...'
    cart = 722.79, 644.55, 748.03,  0.0, 0.7111, 0.7031, -0.0001
    cr.set_cart_mm(cart)

    cart = 715.71, 619.77, 748.03,  0.0, -0.00872, -1., 0.
    cr.set_cart_mm(cart)

    cart = 664.24, 612.12, 752.63 + side_quat
    cr.set_cart_mm(cart)

    print 'TOUCH MOTION'
    if do_touch_motion:
        num_rep = 4*int(original_d/d)
        num_rep_2 = 2*int(original_d/d)
        movement_list = []
        for i in range(num_rep):
            movement_list += [[0, d, 0]]
        for i in range(num_rep_2):
            movement_list += [[0, 0, -d]]
        for i in range(num_rep):
            movement_list += [[0, -d, 0]]
        for i in range(num_rep):
            movement_list += [[0, 0, d]]

        cr.perfrom_experiment(
            experiment_name=experiment_name + '_side',
            movement_list=movement_list
        )

    ###################### Back
    print 'RELOCATING...'
    cart = 765.71, 619.77, 748.03,  0.0, -0.00872, -1., 0.
    cr.set_cart_mm(cart)

    cart = 764.87, 670.99, 752.58,  0.0, 0.6846, -0.72888, 0.
    cr.set_cart_mm(cart)

    cart = 694.93, 650.38, 752.55 + back_quat
    cr.set_cart_mm(cart)

    print 'TOUCH MOTION'
    if do_touch_motion:
        num_rep = 4*6*int(original_d/d)
        num_rep_2 = 2*int(original_d/d)
        movement_list = []
        for i in range(num_rep_2):
            movement_list += [[0, 0, -d]]
        for i in range(num_rep):
            movement_list += [[-d, 0, 0]]
        for i in range(num_rep_2):
            movement_list += [[0, 0, -d]]
        for i in range(num_rep):
            movement_list += [[d, 0, 0]]

        cr.perfrom_experiment(
            experiment_name=experiment_name + '_back',
            movement_list=movement_list
        )

    cart = 764.87, 670.99, 752.58,  0.0, 0.6846, -0.72888, 0.
    cr.set_cart_mm(cart)
