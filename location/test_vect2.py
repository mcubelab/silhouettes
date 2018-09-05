from world_positioning import *
from matplotlib.path import Path
import matplotlib.pyplot as plt
import math, cv2, os, pickle, scipy.io, pypcd, subprocess, sys


def old_visualize2_pointclouds(pcs):
    pc1 = pcs[0]
    pc2 = pcs[1]
    pointcloud = {'x': [], 'y': [], 'z': []}
    a = np.random.permutation(len(pc1))
    max_points = 5000
    a =np.sort(a[0:min(max_points, len(pc1))])
    for i in a:
        pointcloud['x'].append(pc1[i][0])
        pointcloud['y'].append(pc1[i][1])
        pointcloud['z'].append(pc1[i][2])
    mean_y = np.mean(pointcloud['y'])
    std_y = np.std(pointcloud['y'])
    vmin = mean_y - 4*std_y
    vmax = mean_y + 4*std_y
    ax = plt.axes(projection='3d')

    ax.scatter3D(pointcloud['x'], pointcloud['y'], pointcloud['z'],
        c=pointcloud['y'], cmap='Blues', s=3, vmin = vmin, vmax=vmax)
    #surf = ax.plot_trisurf(pointcloud['x'], pointcloud['y'], pointcloud['z'], cmap='Blues', linewidth=0)
    pointcloud = {'x': [], 'y': [], 'z': []}
    a = np.random.permutation(len(pc2))
    max_points = 5000
    a = np.sort(a[0:min(max_points, len(pc2))])
    for i in a:
        pointcloud['x'].append(pc2[i][0])
        pointcloud['y'].append(pc2[i][1])
        pointcloud['z'].append(pc2[i][2])
    mean_y = np.mean(pointcloud['y'])
    std_y = np.std(pointcloud['y'])
    vmin = mean_y - 4*std_y
    vmax = mean_y + 2*std_y
    ax.scatter3D(pointcloud['x'], pointcloud['y'], pointcloud['z'],
        c=pointcloud['y'], cmap='Reds', s=6, vmin = vmin, vmax=vmax)

    #surf = ax.plot_trisurf(pointcloud['x'], pointcloud['y'], pointcloud['z'], cmap='Reds', linewidth=0)
    '''
    from scipy.interpolate import griddata
    X, Y = np.meshgrid(pointcloud['x'], pointcloud['z'])
    Z = griddata((pointcloud['x'], pointcloud['z']), pointcloud['y'], (X, Y), method='linear')
    ax.plot_surface(X,Z,Y, cmap=cm.jet)
    '''
    # Set viewpoint.
    ax.azim = -90
    ax.elev = 0

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
    plt.show()


data = np.load('data_1.npy')
# ['p3', 'p4', 'gripper', 'image', 'cart', 'height', 'vector']

height_map = data.item().get('height')

# cv2.imshow("height_map", height_map)
# cv2.waitKey(0)
# print data.item().get('p3')
print data.item().get('p4')
# print "@@@@@@@@@@@@@@@@@@@@"

gripper_state = {}
gripper_state['pos'] = data.item().get('cart')[0:3]
gripper_state['quaternion'] = data.item().get('cart')[-4:]
gripper_state['Dx'] = data.item().get('gripper')
gripper_state['Dz'] = 372  # Base + wsg + finger

fitting_params = (0.08086415302906927, 0.003541680189520563, -9.909789067143956e-06, 0.08074122271081786, 0.0012391849729127683, -1.2457408497040674e-05,1, 10.373751246753443, -13.893532142199785)
# fitting_params = (0.08022105934199905, 0.00343957595966786, -9.89284234213047e-06, 0.08060915211988702, 0.00043023818716175587, -1.242337728623875e-05, 2.8600379020441844, 10.373751246753443, -13.893532142199785)
a = fast_pxb2wb_3d(height_map, 1, gripper_state, fitting_params, threshold=0.001)

b = data.item().get('p4')

old_visualize2_pointclouds([a, b])
