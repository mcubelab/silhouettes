import math, cv2, os, pickle, scipy.io
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import inv
import pypcd


def visualize_pointcloud(pointcloud):
    ax = plt.axes(projection='3d')
    # ax.scatter3D(pointcloud['x'], pointcloud['y'], pointcloud['z'],
    #     c=pointcloud['x'], cmap='Greens')
    ax.scatter3D(pointcloud['x'], pointcloud['y'], pointcloud['z'],
        c=pointcloud['y'], cmap='Blues')

    # Set viewpoint.
    ax.azim = 135
    ax.elev = 15

    # Label axes.
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')

    def axisEqual3D(ax):
        extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:, 1] - extents[:, 0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize/2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    axisEqual3D(ax)
    plt.gca().invert_xaxis()
    plt.show()


def npPC2dictPC(np_pointcloud):
    pointcloud = {'x': [], 'y': [], 'z': []}
    for i in range(np_pointcloud.shape[0]):
        pointcloud['x'].append(np_pointcloud[i][0])
        pointcloud['y'].append(np_pointcloud[i][1])
        pointcloud['z'].append(np_pointcloud[i][2])
    return pointcloud


path = '2.pcd'
pc = pypcd.PointCloud.from_path(path)
pointcloud = npPC2dictPC(pc.pc_data)
visualize_pointcloud(pointcloud)
