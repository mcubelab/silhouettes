import math, cv2, os, pickle, scipy.io, pypcd, subprocess
from processing.location import Location
from processing.raw2pxb import RAW2PXB
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import inv

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_string_pc(pointcloud):
    string = "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH " + str(pointcloud.shape[0]) + "\nHEIGHT 1\nVIEWPOINT 0.0 0.0 0.0 1.0 0.0 0.0 0.0\nPOINTS " + str(pointcloud.shape[0]) + "\nDATA ascii"
    for elem in pointcloud:
        string += '\n' + str(elem[0]) + ' ' + str(elem[1]) + ' ' + str(elem[2])
    return string

def stitch_pointclouds(fixed, moved):
    # 1. We save the pointclouds in .pcd format
    name = 'processing/c++/cloud' + str(0) + '.pcd'
    with open(name, 'w') as f:
        f.write(get_string_pc(fixed))

    name = 'processing/c++/cloud' + str(1) + '.pcd'
    with open(name, 'w') as f:
        f.write(get_string_pc(moved))

    # 2. We run the c++ program to stitch them
    command = 'cd processing/c++/; ./pairwise_incremental_registration cloud[0-1].pcd'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()

    # 3. We load and return the merged pointcloud
    path = 'processing/c++/1.pcd'
    pc = pypcd.PointCloud.from_path(path)
    # pointcloud = npPC2dictPC(pc.pc_data)
    return pc.pc_data

def get_global_pointcloud(directory, touches, global_pointcloud, processed_global_pointcloud):
    loc = Location()
    for i in touches:
        print "Processing img " + str(i) + "..."
        exp = str(i)
        local_pointcloud = loc.get_local_pointcloud(
            gs_id=1,
            directory=directory,
            num=i)

        # local_pointcloud_arr = dictPC2npPC(local_pointcloud)

        if global_pointcloud is None:
            global_pointcloud = local_pointcloud                # Dict format
            # processed_global_pointcloud = local_pointcloud_arr  # Numpy format
        else:
            global_pointcloud = loc.merge_pointclouds(global_pointcloud, local_pointcloud)
            # processed_global_pointcloud = stitch_pointclouds(processed_global_pointcloud, local_pointcloud_arr)
