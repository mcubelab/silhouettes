from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import math, os, sys, cv2, yaml
try:
    import tf.transformations as tfm
except Exception as e:
    pass

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
#SHAPES_ROOT = os.getcwd().split("/silhouettes/")[0] + "/silhouettes/"
#params_dict = yaml.load(open(SHAPES_ROOT + 'resources/params.yaml'))

try:
    params_dict = yaml.load(open('/home/mcube/silhouettes/' + 'resources/params.yaml'))
except:
    pass

try:
    params_dict = yaml.load(open('/home/ubuntu/silhouettes/' + 'resources/params.yaml'))
except:
    pass

try:
    params_dict = yaml.load(open('/home/oleguer/silhouettes/' + 'resources/params.yaml'))
except:
    pass

half_y = params_dict['input_shape_gs2'][1]/2.

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

def grb2wb(point, gripper_pos, quaternion):
    w2gr_mat = __quaternion_matrix(quaternion)
    #print 'w2: ', w2gr_mat
    #print 'tfm: ',
    #w2gr_mat = tfm.quaternion_matrix(quaternion)
    v = (point[0], point[1], point[2], 1.0)
    #print "v in gripper base: " + str(v)
    v = w2gr_mat.dot(v)
    #v = np.linalg.inv(w2gr_mat).dot(v)
    #print 'first: ', v[0]
    # print v[0]
    #v[0] += (0.46744-gripper_pos[0])*2000
    # print v[0]
    #import pdb; pdb.set_trace()
    #print 'after: ' , v[0]
    '''
    translate = gripper_pos*1000
    tranf_matrix = np.dot(tfm.compose_matrix(translate=translate),tfm.quaternion_matrix(quaternion))[0:3]
    print 'v: ', v
    print 'op1: ', tranf_matrix.dot(v)
    print 'op2: ', v[0:3] + 1000*gripper_pos
    '''
    #print "v in world base: " + str(v)
    #print "Gripper pos: " + str(gripper_pos*1000)

    # for i in range(3):
    #     gripper_pos[i] = gripper_pos[i]*1000
    # return v[0:3] + gripper_pos
    return v[0:3] + 1000*gripper_pos

def wb2grb(point, gripper_pos, quaternion):
    w2gr_mat = __quaternion_matrix(quaternion)
    #w2gr_mat = tfm.quaternion_matrix(quaternion)

    point = point - 1000*gripper_pos
    v = (point[0], point[1], point[2], 1.0)
    #print "v in gripper base: " + str(v)
    #v = w2gr_mat.dot(v)
    #v = np.linalg.inv(w2gr_mat).dot(v)
    v = np.transpose(w2gr_mat).dot(v)
    #print "v in world base: " + str(v)
    #print "Gripper pos: " + str(gripper_pos*1000)

    # for i in range(3):
    #     gripper_pos[i] = gripper_pos[i]*1000
    # return v[0:3] + gripper_pos
    return v[0:3]

def px2mm(point, fitting_params):
    x, y = point
    k1, k2, k3,  l1, l2, l3,  dx, dy, dz = fitting_params
    p1 = (x, y - half_y)
    p2 = (p1[0]*k1 + p1[1]*k2 + k3*p1[0]*p1[1],   p1[1]*l1 + p1[0]*l2 + l3*p1[1]*p1[0])
    return p2

def pxb2grb(point, gs_id, gripper_state, fitting_params):
    p2 = px2mm(point, fitting_params)
    if gs_id == 1:
        normal = 1
    else:
        normal = -1
    Dx = gripper_state['Dx'] # Obertura
    Dz = gripper_state['Dz']
    k1, k2, k3,  l1, l2, l3,  dx, dy, dz = fitting_params
    p3 = (normal*(Dx + dx), p2[1] + dy, Dz + dz + p2[0])

    return p3

# '''
def pxb_2_wb_3d(point_3d, gs_id, gripper_state, fitting_params):
    x, y, z = point_3d
    if gs_id == 1:
        normal = 1
    else:
        normal = -1

    pos = gripper_state['pos']
    quaternion = gripper_state['quaternion']
    Dx = gripper_state['Dx'] # Obertura
    Dz = gripper_state['Dz']

    k1, k2, k3,  l1, l2, l3,  dx, dy, dz = fitting_params

    p1 = (x, y - half_y, z)
<<<<<<< HEAD
    p2 = (p1[0]*k1 + p1[1]*k2 + k3*p1[0]*p1[1],   p1[1]*l1 + p1[0]*l2 + l3*p1[1]*p1[0],   p1[2])
=======
    #p2 = (p1[0]*k1 + p1[1]*k2, p1[1]*l1 + p1[0]*l2, p1[2])
    p2 = (p1[0]*k1 + p1[1]*k2 + k3*p1[0]*p1[1],   p1[1]*l1 + p1[0]*l2 + l3*p1[1]*p1[0], p1[2])
>>>>>>> 7cd667187c63adfa183f3016452cb6ba0c0f0434
    p3 = (normal*(Dx + dx + p2[2]), p2[1] + dy, Dz + dz + p2[0])
    p4 = grb2wb(point=p3, gripper_pos=pos, quaternion=quaternion)
    return p4
# '''
