import numpy as np
from control_robot import ControlRobot

if __name__ == "__main__":
    experiment_name = "/media/mcube/data/shapes_data/pos_calib/full_bar"
    do_touch_motion = True


    cr = ControlRobot(gs_ids=[2], force_list=[5])

    ###################### Front
    print 'RELOCATING...'
    cart = 674.69, 644.55, 748.03, 0.0, 0.7111, 0.7031, -0.0001
    cr.set_cart_mm(cart)

    print 'TOUCH MOTION'
    if do_touch_motion:
        d = 10
        movement_list = [
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],

            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],

            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],

            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],

            [0, 0, -d],

            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],

            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],

            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],

            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            ]
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

    cart = 664.24, 612.12, 752.63,  0.0, -0.00872, -1., 0.
    cr.set_cart_mm(cart)

    print 'TOUCH MOTION'
    if do_touch_motion:
        movement_list = [
            [0, d, 0],
            [0, d, 0],
            [0, d, 0],
            [0, d, 0],

            [0, 0, -d],
            [0, 0, -d],

            [0, -d, 0],
            [0, -d, 0],
            [0, -d, 0],
            [0, -d, 0],

            [0, 0, d],
            [0, 0, d],
            [0, 0, d],
            [0, 0, d]
            ]
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

    cart = 694.93, 650.38, 752.55,  0.0, 0.6846, -0.72888, 0.
    cr.set_cart_mm(cart)

    print 'TOUCH MOTION'
    if do_touch_motion:
        movement_list = [
            [0, 0, -d],

            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],

            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],

            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],

            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],
            [-d, 0, 0],

            [0, 0, -1.5*d],

            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],

            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],

            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],

            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            [d, 0, 0],
            ]
        cr.perfrom_experiment(
            experiment_name=experiment_name + '_back',
            movement_list=movement_list
        )

    cart = 764.87, 670.99, 752.58,  0.0, 0.6846, -0.72888, 0.
    cr.set_cart_mm(cart)
