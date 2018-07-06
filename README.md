# silhouettes
Scope:
 - **Local Shape:** From raw GelSlim image to local heightmap reconstruction
 - **Global Shape:** From multiple raw GelSlim images and robot poses to global shape reconstruction
 - **GelSlim simulation:** From virtual heightmap to simulated GelSlim picture

# Things you can do:

## 1. Gather Data:

### Capture 1 Picture [No robot movement]:
Initialize pman and run: 0-roscore, 1-robotconfig-real, 5-rviz
> cd gathering

> open data_collector.py

Make sure in __name__ == "__main__":
 1. You initialize the DataCollector class with: only_one_shot=True
 2. You put the save_path you want
 3. dc.it is the number which will define the name of the image (if you want to save multiple images in the same directory, make sure you change this number each time) 
 4. You select the data you want to get in dc.get_data(...)

> python data_collector.py

You can control+c the program once you see something written in the Terminal (there is a deault delay otherwise)


### Capture Video [No robot movement]:
Initialize pman and run: 0-roscore, 1-robotconfig-real, 5-rviz
> cd gathering

> open data_collector.py

Make sure in __name__ == "__main__":
 1. You initialize the DataCollector class with: only_one_shot=False
 2. You put the save_path you want
 3. dc.it = -1
 4. You select the data you want to get in dc.get_data(...)
 5. You put the time you want to be recording in time.sleep(#seconds)

> python data_collector.py

You can control+c the program whenever you think you have enough data


### Predefined Tactile Exploration [Robot movement]:
Initialize pman and run: 0-roscore, 1-robotconfig-real, 2-abb, 3-hand, 5-rviz
> cd gathering

> open main_movement.py

In __name__ == "__main__":
 1. experiment_name is the path where all the data will be saved
 2. movement_list is a list of (x, y, z) translations that the robot will do, after each movement it will stop and close the gripper.
 3. Make sure in gs_ids you put all the gs ids from which you want to capture the image
 4. You can put more than one element in force_list and the gripper will close multiple times at each stop with the forces specified

> python main_movement.py

## 2. Calibrate the GelSlim position wrt World:
Steps:
 0. Change the robot finger for the calibraton tool and touch characteristic points in the environment. Take note of the (x, y, z) position wrt the base of the robot. Put the finger again.
 1. Using the **Predefined Tactile Exploration methodology** touch around the areas from, which you have characteristic points saved
 2. 
 > cd pos_calibration

 > open fitting_package.py

In __name__ == "__main__":
 1. If your characteristic points are squares, border or line, then the positions are precalculated, uncomment the corresponding block and run the program, otherwise, you'll need to create a custom point_list in **def get_real_point_list()**
 2. Change the touches_list to range of the number of touches you have for that particular touching set.
 3. If you want to preload from previous automatic saving put the path to already_done
 
 > python fitting_package.py
 
 Manually click the characteristic points in the same order as the point_list. For **squares** it is the reading order, for **line** is up to down, and for **border** it is an inverted "c" from top to bottom. If you can't see a point roll the mouse scroll once to cancel. You will have feedback at the terminal.
 
 4. 
 > open world_calib.py
