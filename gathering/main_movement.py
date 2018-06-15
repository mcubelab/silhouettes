from control_robot import ControlRobot
from data_collector import DataCollector


if __name__ == "__main__":

    experiment_name = "datasets/pos_calibration_squares_color"
    d = 2
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

        [0, 0, -d],
        [0, 0, -d],
        [0, 0, -d],
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

        [0, 0, -d],
        [0, 0, -d],
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
        ]

    cr = ControlRobot(gs_ids=[2], force_list=[20])
    cr.perfrom_experiment(
        experiment_name=experiment_name,
        movement_list=movement_list
    )
