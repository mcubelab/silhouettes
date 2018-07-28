import numpy as np
from control_robot import ControlRobot
import time

if __name__ == "__main__":
    experiment_name = "/media/mcube/data/shapes_data/raw/automatic_test"

    # Enter gs_id:
    gs_id = 2
    object_angle = 0

    # Enter object main dimensions in [mm]
    length = 40 # distance from the magnetic base to the top
    # Enter area to area to explore uniformly
    height = 20
    width = 20
    number_of_points = 3

    # Enter these other params
    cr = ControlRobot(gs_ids=[gs_id], force_list=[20])

    # IMPORTANT NOTE: Make sure you start at a save position
    if gs_id == 2:
        start_cart = 883.57-length, 380., 662.93, 0., -0.9998, 0.0185, 0.
    elif gs_id == 1:
        start_cart = 883.57-length, 358.06, 662.93, 0., -0.0185, -0.9998, 0.


    # We generate all the random offsets from the center:
    rnd_x = np.random.uniform(-width/2, width/2, size=number_of_points)
    rnd_y = np.random.uniform(-height/2, height/2, size=number_of_points)

    # We compute all the movememts
    movement_list = [(0, 0, 0)]
    for x, y in zip(rnd_x, rnd_y):
        movement = [0, 0, 0]
        movement[0] = 0
        movement[1] = x - movement_list[-1][1]
        movement[2] = y - movement_list[-1][2]
        movement_list.append(movement)

    # print zip(rnd_x, rnd_y)
    print movement_list

    # We perform the movement
    print 'RELOCATING...'
    cr.set_cart_mm(start_cart)
    time.sleep(5)
    #
    print 'TOUCHING...'
    cr.perfrom_experiment(
        experiment_name=experiment_name,
        movement_list=movement_list
    )
