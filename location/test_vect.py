# from location import *
import numpy as np
import math

data = np.load('data_0.npy')
# ['p3', 'p4', 'gripper', 'image', 'cart', 'height', 'vector']
# data.item().get('height')
# data.item().get('p3')
# data.item().get('p4')

def __quaternion_matrix(quaternion):
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    _EPS = 1e-5
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])


# gripper_state = {}
# print data
# print data.item().get('cart')
# gripper_state['pos'] = data["cart"][0:3]
# gripper_state['quaternion'] = data["cart"][-4:]
# gripper_state['Dx'] = data["gripper"]/2.0
# gripper_state['Dz'] = 372  # Base + wsg + finger
#
# height_map = data["height"]

# dim = height_map.size()
dim = (2, 3)

# xvalues = np.array(range(dim[0]))
# yvalues = np.array(range(dim[1]))
# pos = np.stack((np.meshgrid(yvalues, xvalues)), axis=2)

x, y = np.meshgrid(range(dim[0]), range(dim[1]), indexing='ij')
z = np.random.random(dim).astype(int)

normal = 1
k1, k2, k3,  l1, l2, l3,  dx, dy, dz = (0.07835168669290037, -4.221476330432822e-05, -1.6811485728606957e-06, 0.07768172299720112, 0.006936919318359956, -1.5701541829697096e-06, 2.856983575297971, 3.4512714390353536, -15.578807446701708)
half_y = 3
Dx, Dy, Dz = 0, 0, 0

# p1 = (x, y - half_y, z)
# p2 = (p1[0]*k1 + p1[1]*k2, p1[1]*l1 + p1[0]*l2, p1[2])
# p3 = (normal*(Dx + dx + p2[2]), p2[1] + dy, Dz + dz + p2[0])

quaternion = (0, 0.7, -0.7, 0)
w2gr_mat = __quaternion_matrix(quaternion)
# v = (p3[0], p3[1], p3[2], 1.0)
print x+10
print y+20
print z
v = np.stack((x+10, y+20, z, z+1), axis=0)
# print v
print "################"
print v[0]
# print w2gr_mat.dot(v[0][0])

def f(a):
    return w2gr_mat.dot(a)

u = np.apply_along_axis(f, 0, v)
print u
u = [u[0] + 10, u[1], u[2], u[3]]
u = np.swapaxes(u[:-1], 0, 2)
print u
print np.reshape(u, (dim[0]*dim[1], 3))

[0, 0, 0, 1, 0, 1]

# print w2gr_mat*v[0]

# print p3


# 2. We convert height_map data into world position
gripper_state = {}
gripper_state['pos'] = cart[0:3]
gripper_state['quaternion'] = cart[-4:]
gripper_state['Dx'] = wsg_list[0]['width']/2.0
gripper_state['Dz'] = 139.8 + 72.5 + 160  # Base + wsg + finger

pointcloud = []
pixel_list = []
for i in range(height_map.shape[0]):
    for j in range(height_map.shape[1]):
        if(height_map[i][j] >= 0.10):
            pixel_list.append([i,j])
            world_point = pxb_2_wb_3d(
                point_3d=(i, j, height_map[i][j]),
                gs_id=gs_id,
                gripper_state = gripper_state,
                fitting_params = params_gs
            )
            a = np.asarray(world_point)
            pointcloud.append(a)
"""
