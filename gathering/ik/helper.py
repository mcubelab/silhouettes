import os, errno, sys
rgrasp_path = os.environ['CODE_BASE'] + '/catkin_ws/src/apc_planning/src'
sys.path.append(rgrasp_path)
import collision_detection.collisionHelper
import json
import numpy as np
import rospy
import tf.transformations as tfm
import time
from visualization_msgs.msg import MarkerArray, Marker
from marker_helper import createCubeMarker2, createSphereMarker
from ik import setSpeedByName
#~ from ik.ik import*
from numpy import linalg as la
from roshelper import lookupTransform, poseTransform
import gripper
import std_msgs.msg
import collision_detection.collision_projection
#from suction_projection import suction_projection_func
import sensor_msgs.msg
from robot_comm.srv import robot_GetCartesian, robot_SetCartesian
from cv_bridge import CvBridge, CvBridgeError
import cv2

obj_dim_data = []
gripper_command_pub = rospy.Publisher('/hand_commands', sensor_msgs.msg.JointState, queue_size=10)

def get_obj_dim(objId):
    global obj_dim_data
    if len(obj_dim_data) == 0:
        jsonFilename = os.environ['CODE_BASE']+'/catkin_ws/src/apc_config/object_data/objectDictionary.json'
        with open(jsonFilename) as data_file:
            data = json.load(data_file)
        for key, value in data.iteritems():
            data[key] = [value[0]/1000.0, value[1]/1000.0, value[2]/1000.0]
        obj_dim_data = data

    return obj_dim_data[objId]

def get_obj_vol(objId):
    d = get_obj_dim(objId)
    return d[0]*d[1]*d[2]

def xyzrpy_from_xyzquat(pose):
    return pose[0:3] + list(tfm.euler_from_quaternion(pose[3:7])) # x,y,z,qx,qy,qz,qw


def matrix_from_xyzquat_np_array(arg1, arg2=None):
    if arg2 is not None:
        translate = arg1
        quaternion = arg2
    else:
        translate = arg1[0:3]
        quaternion = arg1[3:7]

    return np.dot(tfm.compose_matrix(translate=translate) ,
                   tfm.quaternion_matrix(quaternion))

def matrix_from_xyzquat(arg1, arg2=None):
    return matrix_from_xyzquat_np_array(arg1, arg2).tolist()

def transformBack(tf_xyzquat, pose):
    T_mat = tfm.concatenate_matrices( tfm.translation_matrix(tf_xyzquat[0:3]), tfm.quaternion_matrix(tf_xyzquat[3:7]))
    pose_mat = tfm.concatenate_matrices( tfm.translation_matrix(pose[0:3]),  tfm.quaternion_matrix(pose[3:7]) )
    new_pose_mat = np.dot(pose_mat, tfm.inverse_matrix(T_mat))
    return tfm.translation_from_matrix(new_pose_mat).tolist() + tfm.quaternion_from_matrix(new_pose_mat).tolist()

def transformTo(tf_xyzquat, pose):
    T_mat = tfm.concatenate_matrices( tfm.translation_matrix(tf_xyzquat[0:3]), tfm.quaternion_matrix(tf_xyzquat[3:7]))
    pose_mat = tfm.concatenate_matrices( tfm.translation_matrix(pose[0:3]),  tfm.quaternion_matrix(pose[3:7]) )
    new_pose_mat = np.dot(T_mat, pose_mat)
    return tfm.translation_from_matrix(new_pose_mat).tolist() + tfm.quaternion_from_matrix(new_pose_mat).tolist()

def posTransformTo(tf_xyzquat, pos):
    T_mat = tfm.concatenate_matrices( tfm.translation_matrix(tf_xyzquat[0:3]), tfm.quaternion_matrix(tf_xyzquat[3:7]))
    pose_mat = tfm.translation_matrix(pos)
    new_pose_mat = np.dot(T_mat, pose_mat)
    return tfm.translation_from_matrix(new_pose_mat).tolist()

def get_params_yaml(pose_name):
    params_list=['x','y','z','qx','qy','qz','qw']
    pose_list = [rospy.get_param(pose_name+'/'+p) for p in params_list]
    return pose_list

def pause():
    raw_input('Press any key to continue')

def pauseFunc(withPause):
    if withPause:
        pause()
    return

def visualizeFunc(withVisualize, plan):
    if withVisualize:
        plan.visualize()
    return

def getObjCOM(objPose, objId):
    #gives you the center of mass of the object
    # object frame is attached at com
    objPosition = objPose[0:3]
    return objPosition

def openGripper():
    # call gripper node, open
    gripper.open()
    return 1

def closeGripper(forceThreshold):
    # call gripper node, close
    gripper.close()
    return 1

def releaseGripper(release_pos, release_speed=50):
    # call gripper node, close
    gripper.release(release_pos*1000, release_speed)
    return 1

def graspGripper(grasp_pos, grasp_speed=50):
    # move_pos in meter, move_speed in mm/s
    #WE should make grasp gripper force controlled--Nikhil
    # call gripper node, grasp

    jnames = ['gripper_command']
    gripper_msgs = sensor_msgs.msg.JointState()
    gripper_msgs.name  = jnames
    gripper_msgs.position = [grasp_pos]
    gripper_msgs.velocity = [grasp_speed/1000]
    gripper_msgs.effort = [np.inf]
    gripper_command_pub.publish(gripper_msgs)

    grasp_pos=1000*grasp_pos

    gripper.grasp(grasp_pos, grasp_speed)
    return 1

def moveGripper(move_pos, move_speed=50):
    #~publis gripper state

    jnames = ['gripper_command']
    gripper_msgs = sensor_msgs.msg.JointState()
    gripper_msgs.name  = jnames
    gripper_msgs.position = [move_pos]
    gripper_msgs.velocity = [move_speed/1000]
    gripper_msgs.effort = [np.inf]
    gripper_command_pub.publish(gripper_msgs)

    # move_pos in meter, move_speed in mm/s
    # call gripper node, grasp
    move_pos=1000*move_pos
    gripper.move(move_pos, move_speed)
    return 1

def setForceGripper(force=50):
    gripper.set_force(force)
    return 1

def graspinGripper(grasp_speed=50,grasp_force=40):
    # call grasp_in function from gripper node

    jnames = ['gripper_command']
    gripper_msgs = sensor_msgs.msg.JointState()
    gripper_msgs.name  = jnames
    gripper_msgs.position = [0]
    gripper_msgs.velocity = [grasp_speed/1000]
    gripper_msgs.effort = [grasp_force]
    gripper_command_pub.publish(gripper_msgs)
    gripper.grasp_in(grasp_speed,grasp_force)
    return 1

def graspoutGripper(grasp_speed=50,grasp_force=40):
    # call grasp_out function from gripper node

    jnames = ['gripper_command']
    gripper_msgs = sensor_msgs.msg.JointState()
    gripper_msgs.name  = jnames
    gripper_msgs.position = [0.11]
    gripper_msgs.velocity = [grasp_speed/1000]
    gripper_msgs.effort = [grasp_force]
    gripper_command_pub.publish(gripper_msgs)
    gripper.grasp_in(grasp_speed,grasp_force)


    gripper.grasp_out(grasp_speed,grasp_force)
    return 1

def matrix_from_xyzrpy(arg1, arg2=None):
    if arg2 is not None:
        translate = arg1
        rpy = arg2
    else:
        translate = arg1[0:3]
        rpy = arg1[3:7]

    return np.dot(tfm.compose_matrix(translate=translate) ,
                   tfm.euler_matrix(rpy[0],rpy[1],rpy[2])).tolist()

def xyzquat_from_matrix(matrix):
    return tfm.translation_from_matrix(matrix).tolist() + tfm.quaternion_from_matrix(matrix).tolist()

def quat_from_matrix(rot_matrix):
    return (tfm.quaternion_from_matrix(rot_matrix))

def mat2quat(orient_mat_3x3):
    orient_mat_4x4 = [[orient_mat_3x3[0][0],orient_mat_3x3[0][1],orient_mat_3x3[0][2],0],
                       [orient_mat_3x3[1][0],orient_mat_3x3[1][1],orient_mat_3x3[1][2],0],
                       [orient_mat_3x3[2][0],orient_mat_3x3[2][1],orient_mat_3x3[2][2],0],
                       [0,0,0,1]]

    orient_mat_4x4 = np.array(orient_mat_4x4)
    quat=quat_from_matrix(orient_mat_4x4)
    return quat

def rotmatY(theta):
    theta_rad=(theta*np.pi)/180
    return(np.array([[np.cos(theta_rad), 0, np.sin(theta_rad)],
    [0, 1, 0],[-np.sin(theta_rad), 0, np.cos(theta_rad)]]))


def rotmatX(theta):
    theta_rad=(theta*np.pi)/180
    return(np.array([[1, 0, 0],
    [0, np.cos(theta_rad), -np.sin(theta_rad)],[0, np.sin(theta_rad), np.cos(theta_rad)]]))

def rotmatZ(theta):
    theta_rad=(theta*np.pi)/180
    return(np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0],[np.sin(theta_rad), np.cos(theta_rad), 0], [0,0,1]]))

def plotPickPoints(pick_points,viz_pub=None,special_index=None):
    if viz_pub!=None:
        markers_msg = MarkerArray()
        markers_msg.markers.append(createDeleteAllMarker('picking_points'))
        for i in range(0,len(pick_points)):
            if i==special_index:
                my_rgb=(0,1,0,1)
            else:
                my_rgb=(1,0,1,1)
	    m=createSphereMarker(offset=tuple(pick_points[i]), marker_id = (i+1), rgba=my_rgb, orientation=(0.0,0.0,0.0,1.0), scale=(.03,.03,.03), frame_id="/map", ns = 'picking_points')
	    markers_msg.markers.append(m)
	viz_pub.publish(markers_msg)
	viz_pub.publish(markers_msg)
	viz_pub.publish(markers_msg)
	viz_pub.publish(markers_msg)
	rospy.sleep(.1)

def plotBoxCorners(corner_points, viz_pub=None, color_rgb = None, namespace = 'corner_points',isFeasible = None):
    if viz_pub!=None:
        markers_msg = MarkerArray()
        #~ markers_msg.markers.append(createAllMarker('corner_points'))
        for i in range(0,len(corner_points)):
            #~ if isFeasible[i] is not:
            if color_rgb == None:
                my_rgb=(1,0,1,1)
            else:
                my_rgb=color_rgb
            m=createSphereMarker(offset=tuple(corner_points[i]), marker_id = (i+1), rgba=my_rgb, orientation=(0.0,0.0,0.0,1.0), scale=(.015,.015,.015), frame_id="/map", ns = namespace)
            markers_msg.markers.append(m)

        viz_pub.publish(markers_msg)
        viz_pub.publish(markers_msg)
        viz_pub.publish(markers_msg)
        viz_pub.publish(markers_msg)
        rospy.sleep(.1)

def createDeleteAllMarker(ns = ''):
    marker = Marker()
    marker.action = 3  # delete all
    marker.ns = ns
    return marker

def createAllMarker(ns = ''):
    marker = Marker()
    marker.ns = ns
    return marker

def deleteMarkers(viz_pub=None,ns=''):
    if viz_pub!=None:
        markers_msg = MarkerArray()
        markers_msg.markers.append(createDeleteAllMarker(ns))
        viz_pub.publish(markers_msg)
        rospy.sleep(0.1)


def vision_transform_precise_placing_with_visualization(bbox_info,viz_pub,listener):
    box_dim=bbox_info[7:10]
    (rel_pose,BoxBody)=vision_transform_precise_placing(bbox_info,listener=listener)

#    markers_msg = MarkerArray()
#    markers_msg.markers.append(createDeleteAllMarker('object_bounding_box_for_collision'))
#    m=createCubeMarker2(offset=tuple(rel_pose[0:3]), marker_id = 5, rgba=(1,0,1,1), orientation=tuple(rel_pose[3:7]), scale=tuple(box_dim[0:3]), frame_id="/link_6", ns = 'object_bounding_box_for_collision')
#    m.frame_locked = True
#    markers_msg.markers.append(m)
#    #viz_pub.publish(markers_msg)
#    #viz_pub.publish(markers_msg)
#    #viz_pub.publish(markers_msg)
#    #viz_pub.publish(markers_msg)
#    #rospy.sleep(0.1)
#
#    m=createCubeMarker2(offset=tuple(bbox_info[4:7]), marker_id = 6, rgba=(0,0,1,1), orientation=tuple(bbox_info[0:4]), scale=tuple(box_dim[0:3]), frame_id="/map", ns = 'object_bounding_box_for_collision')
#    m.frame_locked = True
#    markers_msg.markers.append(m)
#    viz_pub.publish(markers_msg)
#    viz_pub.publish(markers_msg)
#    viz_pub.publish(markers_msg)
#    viz_pub.publish(markers_msg)
#    rospy.sleep(0.1)


    return (rel_pose, BoxBody)

def visualize_placing_box(bbox_info,place_pose,viz_pub):
    if viz_pub!=None:
        markers_msg = MarkerArray()
        m=createCubeMarker2(offset=tuple(place_pose[0:3]), marker_id = 7, rgba=(0,0,1,1), orientation=tuple(place_pose[3:7]), scale=tuple(box_dim[0:3]), frame_id="/map", ns = 'object_bounding_box_for_collision')
        m.frame_locked = True
        markers_msg.markers.append(m)
        viz_pub.publish(markers_msg)
        viz_pub.publish(markers_msg)
        viz_pub.publish(markers_msg)
        viz_pub.publish(markers_msg)
        rospy.sleep(0.1)

def fake_bbox_info(listener=None):
    theta=np.random.rand()*2.0*np.pi

    orient_mat_4x4 =[[np.cos(theta),-np.sin(theta),0,0],[np.sin(theta),np.cos(theta),0,0],[0,0,1,0],[0,0,0,1]]
    orient_quat=quat_from_matrix(orient_mat_4x4)
    if listener==None:
        box_pos=[0.9949660234451294, -0.5, .2]
    else:
        hand_pose = poseTransform([0.0,0.0,0.0]+[0.0,0.0,0.0,1.0], "link_6", "map", listener)
    box_pos=[hand_pose[0]+.05*np.random.rand(),hand_pose[1]+.05*np.random.rand(),.2]

    bbox_info=orient_quat.tolist()+box_pos+[0.0,0.0,0.0]
    return bbox_info


def fake_bbox_info_1(listener=None):
    theta=0

    orient_mat_4x4 =[[np.cos(theta),-np.sin(theta),0,0],[np.sin(theta),np.cos(theta),0,0],[0,0,1,0],[0,0,0,1]]
    orient_quat=quat_from_matrix(orient_mat_4x4)
    if listener==None:
        box_pos=[0.9949660234451294, -0.5, .2]
    else:
        hand_pose = poseTransform([0.0,0.0,0.0]+[0.0,0.0,0.0,1.0], "link_6", "map", listener)
    box_pos=[hand_pose[0],hand_pose[1],.2]

    bbox_info=orient_quat.tolist()+box_pos+[0.0,0.0,0.0]
    return bbox_info


def vision_transform_precise_placing(bbox_info,listener):
    print '[vision_transform_precise_placing] bbox_info = ', bbox_info
    box_pose=bbox_info[4:7]+bbox_info[0:4]
    box_dim=bbox_info[7:10]
    rel_pose = poseTransform(box_pose, "map", "link_6", listener)
    print '[vision_transform_precise_placing] rel_pose = ', rel_pose
    box_pos=tfm.translation_matrix(rel_pose[0:3])
    box_pos=box_pos[0:3,3]
    box_rot=tfm.quaternion_matrix(rel_pose[3:7])
    box_x=box_rot[0:3,0]
    box_y=box_rot[0:3,1]
    box_z=box_rot[0:3,2]
    b = [[1.0,1.0,1.0],[1.0,1.0,-1.0],[1.0,-1.0,1.0],[1.0,-1.0,-1.0],[-1.0,1.0,1.0],[-1.0,1.0,-1.0],[-1.0,-1.0,1.0],[-1.0,-1.0,-1.0]]
    BoxBody=[]
    for i in range(0, 8):
        BoxBody.append(box_pos+box_dim[0]*box_x*b[i][0]/2.0+box_dim[1]*box_y*b[i][1]/2.0+box_dim[2]*box_z*b[i][2]/2.0)
    return (rel_pose,BoxBody)

def pose_transform_precise_placing(rel_pose,BoxBody,place_pose,base_pose,bin_pts,finger_pts,margin=0,show_plot=False,viz_pub=None):
    margin=0.0
    print 'rel_pose', rel_pose
    print 'place_pose', place_pose
    #drop_pose is the first guess at the desired pose of the hand
    drop_pose = transformBack(rel_pose, place_pose)
    base_rot=tfm.quaternion_matrix(base_pose[3:7])

    BoxBody_base_pose=[]
    for i in range(0, 8):
        BoxBody_base_pose.append(np.dot(base_rot[0:3,0:3],BoxBody[i]))

    BoxOrigin_base_pose=np.dot(base_rot[0:3,0:3],rel_pose[0:3])

    BoxBody_base_pose=np.vstack(BoxBody_base_pose)
    drop_rot=tfm.quaternion_matrix(drop_pose[3:7])

    box_pose_at_base=transformTo(base_pose,rel_pose)


    theta_mat=np.dot(tfm.quaternion_matrix(place_pose[3:7]),tfm.inverse_matrix(tfm.quaternion_matrix(box_pose_at_base[3:7])))
    theta=np.imag(np.log(theta_mat[0,0]+theta_mat[1,0]*1j))

    if abs(theta-np.pi)<=.5*np.pi:
        theta=theta-np.pi

    orient_mat_4x4 =[[np.cos(theta),-np.sin(theta),0,0],[np.sin(theta),np.cos(theta),0,0],[0,0,1,0],[0,0,0,1]]

    drop_quat=quat_from_matrix(np.dot(orient_mat_4x4,tfm.quaternion_matrix(base_pose[3:7])))

    bin_ptsXY=bin_pts[:,0:2]
    finger_ptsXY= np.concatenate((finger_pts[:,0:2],BoxBody_base_pose[:,0:2]),axis=0)

    (shape_translation,dist_val_min,feasible_solution,nearest_point)=collision_detection.collision_projection.projection_func(bin_ptsXY,finger_ptsXY,np.array(place_pose[0:2],ndmin=2),np.array(BoxOrigin_base_pose[0:2],ndmin=2),theta,show_plot,margin)

    p0=BoxBody[0]
    px=BoxBody[4]
    py=BoxBody[2]
    pz=BoxBody[1]

    box_dim=[np.linalg.norm(p0-px),np.linalg.norm(p0-py),np.linalg.norm(p0-pz)]





    #feasible_solution=False
    if feasible_solution:
        drop_pose=shape_translation.tolist()+[drop_pose[2]]+drop_quat.tolist()
    else:
        drop_pose=[np.mean(bin_pts[:,0]),np.mean(bin_pts[:,1])]+[drop_pose[2]]+drop_quat.tolist()

    actual_pose=transformTo(drop_pose,rel_pose)
    actual_pose[2]=place_pose[2]

    if False:
        markers_msg = MarkerArray()
        markers_msg.markers.append(createAllMarker('object_bounding_box_for_collision2'))
        m=createCubeMarker2(offset=tuple(place_pose[0:3]), marker_id = 4, rgba=(1,1,1,1), orientation=tuple(place_pose[3:7]), scale=tuple(box_dim), frame_id="/map", ns = 'object_bounding_box_for_collision2')
        m.frame_locked = True
        markers_msg.markers.append(m)
        m=createCubeMarker2(offset=tuple(actual_pose[0:3]), marker_id = 5, rgba=(0,1,0,1), orientation=tuple(actual_pose[3:7]), scale=tuple(box_dim), frame_id="/map", ns = 'object_bounding_box_for_collision2')
        m.frame_locked = True
        markers_msg.markers.append(m)
        viz_pub.publish(markers_msg)
        viz_pub.publish(markers_msg)
        viz_pub.publish(markers_msg)
        viz_pub.publish(markers_msg)
        rospy.sleep(0.1)

    return (drop_pose,actual_pose)

class Timer(object):
    #### Timer to time a piece of code
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '\t[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)

class shortfloat(float):
    def __repr__(self):
        return "%0.3f" % self


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def CrossProduct(a,b):
    return a[0]*b[1]-a[1]*b[0]

def SameSide(p1,p2,a,b):
    cp1 = CrossProduct(b-a, p1-a)
    cp2 = CrossProduct(b-a, p2-a)
    if cp1*cp2 > 0:
        return True
    return False

def PointInTriangle(p, a,b,c):
    if SameSide(p,a, b,c) and SameSide(p,b, a,c) and SameSide(p,c, a,b):
        return True
    else:
        return False


def in_or_out(grid_x = 0,
              grid_y = 0,
              m_per_x = 0.01,
              m_per_y = 0.01,
              position = [0, 0, 0],
              orientation = [0, 0, 0, 1],
              dim = [0.1,0.1,0.1]):

    # This function assumes we have a box with at least one of its edges
    # touching a plane parallel to the floor.
    # This also assumes that the tesselation is dense enough so that
    # we can do the check with only one point in the square.

    #Obtain obstacle dimensions using orientation
    box_sizes = dim


    p = np.array([grid_x*m_per_x, grid_y*m_per_y])
    empty_bin_height = max(-0.02, -p[0]*0.02/0.2)
    box_module = 0
    for i in range(3):
      box_module = box_module + box_sizes[i]*box_sizes[i]/4
    if (p[0]-position[0])*(p[0]-position[0])+(p[1]-position[1])*(p[1]-position[1]) > box_module:
      return empty_bin_height
    #print 'p: ', p
    #generate (0,0,0),(0,0,1)...(1,1,1)
    vertices = np.zeros([8,3])
    for i in range(8):
        ci = i
        for k in range(3):
            vertices[i,k] = ((ci%2)-0.5)*box_sizes[k] #also rescale
            ci = ci/2

    obs_pose_tfm= np.array(matrix_from_xyzquat(position, orientation))
    obs_pose_orient=obs_pose_tfm[0:3,0:3]
    for i in range(3):
      obs_pose_orient[:,i] = obs_pose_orient[:,i]/la.norm(obs_pose_orient[:,i])

    #center in position
    max_height = 0 # TODO: use vertical_dim calculated above?
    real_vertices = np.zeros((8,3))
    for i in range(8):
        for j in range(3):
            real_vertices[i,j] = position[j]
            for k in range(3):
                real_vertices[i,j] = real_vertices[i,j] + obs_pose_orient[j,k]*vertices[i,k]
        max_height = max(max_height,real_vertices[i,2])

    # Look if inside one of the triangles
    # If too slow, look at only n-2 triangles instead of n choose 3.
    result = empty_bin_height
    for a in range(8):
        for b in range(a+1,8):
            for c in range(b+1,8):
                if PointInTriangle(p, real_vertices[a,0:2],
                                   real_vertices[b,0:2], real_vertices[c,0:2]):
                    result = max(result, (real_vertices[a,2]+real_vertices[b,2]+real_vertices[c,2])/3.) #Maximum height among the 3 points in the triangle.
                #else:
                    #print 'Sometimes its a no'
    #print 'The function is going to return NO'
    return result

#~Setup reference frames (All outputs are 3x1 vectors resolved in world reference frame)
def reference_frames(listener,br):
    #global listener
    #global br

    #~ ****************** Define reference frames***********************
    (world_position, world_quaternion) = lookupTransform(homeFrame="map", targetFrame="map", listener=listener)
    (tote_position, tote_quaternion) = lookupTransform(homeFrame="map", targetFrame="tote", listener=listener)

    #~ ******************* World Frame   *******************************
    world_pose_tfm_list=matrix_from_xyzquat(world_position,world_quaternion)
    world_pose_tfm=np.array(world_pose_tfm_list)
    world_pose_orient=world_pose_tfm[0:3,0:3]
    world_pose_pos=world_pose_tfm[0:3,3]

    #Normalized axes of the shelf frame
    world_X=world_pose_orient[:,0]/la.norm(world_pose_orient[:,0])
    world_Y=world_pose_orient[:,1]/la.norm(world_pose_orient[:,1])
    world_Z=world_pose_orient[:,2]/la.norm(world_pose_orient[:,2])

    #~ ******************* Tote Frame   *******************
    tote_pose_tfm_list=matrix_from_xyzquat(tote_position,tote_quaternion)
    tote_pose_tfm=np.array(tote_pose_tfm_list)
    tote_pose_orient=tote_pose_tfm[0:3,0:3]
    tote_pose_pos=tote_pose_tfm[0:3,3]
    tote_pose = np.hstack((tote_position,tote_quaternion))

    #Normalized axes of the shelf frame
    tote_X=tote_pose_orient[:,0]/la.norm(tote_pose_orient[:,0])
    tote_Y=tote_pose_orient[:,1]/la.norm(tote_pose_orient[:,1])
    tote_Z=tote_pose_orient[:,2]/la.norm(tote_pose_orient[:,2])

    return (world_X, world_Y, world_Z, tote_X, tote_Y, tote_Z, tote_pose_pos)

def clamp(my_value, min_value, max_value):
    campled_value = max(min(my_value, max_value), min_value)
    return (campled_value)

#~Check if Body1 (list of points) has a union with Body 2
def check_collision(Body1, Body2):
    #~ print ToteBody
    collision=False
    for i in range(0, len(Body1)):
        if in_hull(Body1[i], Body2):
            collision = True
            print '*******  TRUE ******'
        #~ else:
            #~ collision = False
    return (collision)

def plot_body(Body, ax, color):
    for j in range(0,len(Body)):
        fx = Body[j][0]#randrange(n, 23, 32)
        fy = Body[j][1]#randrange(n, 0, 100)
        fz = Body[j][2]#randrange(n, -50, -30)
        ax.scatter(fx, fy, fz, c=color, marker='o')

#~Check if robot will collide with Tote
def collision_detect_fn(finger_opening, tcp_pos, hand_orient_norm, listener, br):
    #~Function to detect collision between desired position of gripper and tote
    #~Inputs:
    # 1) finger_opening (Gripper opening (m))
    # 2) tcp_pos (position of tcp in cartesian coordinates(m))
    # 3) hand_orient_norm (3x3 orientation matrix for hand frame (hand_X, hand_Y, hand_Z),
    #       where Z is in line with link 6 axis and X is the axis joining both spatulas)
    #~Outputs:
    # 1) collision (True/False)
    # 2) tcp_pos (new proposed collision free position of tcp (only the x and y positions)
    # 3) hand_orient_norm (new proposed  3x3 orientation matrix for hand frame (hand_X, hand_Y, hand_Z),
    #       where Z is in line with link 6 axis and X is the axis joining both spatulas)

    #~Initialize reference frames axes
    world_X, world_Y, world_Z, tote_X,tote_Y,tote_Z, tote_pose_pos = reference_frames(listener=listener, br=br)
    hand_X=hand_orient_norm[0:3,0]
    hand_Y=hand_orient_norm[0:3,1]
    hand_Z=hand_orient_norm[0:3,2]

    #~Initialize parameters
    collision = False
    collision_section = [False]*8

    #Get constants from yaml
    tote_height=rospy.get_param("/tote/height")
    tote_width=rospy.get_param("/tote/width")
    tote_length=rospy.get_param("/tote/length")
    finger_thickness = rospy.get_param("/finger/thickness")
    finger_width = rospy.get_param("/finger/width")
    gripper_width = rospy.get_param("/gripper/width")
    gripper_length = rospy.get_param("/gripper/length")
    bin_mid_pos = tote_pose_pos+np.array([0,0,tote_height/2.0])
    hand_fing_disp_Yoffset = -finger_width/2 + 0.01
    dist_tcp_to_intersection = rospy.get_param("/wrist/length")
    dist_tcp_to_spatula = rospy.get_param("/gripper/spatula_tip_to_tcp_dist")

    #Build points of tote
    TotePoints = []
    TotePointsExtended = []
    ToteBody = []
    for i in range(0, 8):
        a = [[1,1,1],[1,1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1],[1,-1,1],[1,-1,-1]] #~[X,Y,Z]
        TotePoints.append(bin_mid_pos + a[i][2]*(tote_height/2)*tote_Z+a[i][0]*(tote_width/2.0)*tote_X+a[i][1]*(tote_length/2.0)*tote_Y)
        TotePointsExtended.append(bin_mid_pos + a[i][2]*(tote_height/2)*tote_Z+a[i][0]*(3)*tote_X+a[i][1]*(3)*tote_Y)

    #~Define totes sections (as a collection of points/bounding box)
    for i in range(0,4):
        tempList=[]
        for j in range(0,4):
            counter = np.mod(2*(i)+1+j,8)-1
            tempList.append(TotePoints[counter]) #Tote Points
            tempList.append(TotePointsExtended[counter])#Extend list with Tote points extended
        ToteBody.append((tempList)) #Extend Tote body list with section points

    #Build gripper sections (as a collection of pointsbounding box)
    FingerBody=[]
    WristBody=[]

    for i in range(0, 4):
        b = [[1,1],[1,-1],[-1,1],[-1,-1]]
        #~Finger Body (changes with opening)
        FingerBody.append(tcp_pos+dist_tcp_to_intersection*hand_Z+b[i][0]*(finger_opening/2.0+finger_thickness)*hand_X+b[i][1]*(finger_width/2.0)*hand_Y)
        FingerBody.append(tcp_pos+dist_tcp_to_spatula*hand_Z+b[i][0]*(finger_opening/2.0+finger_thickness)*hand_X+b[i][1]*(finger_width/2.0)*hand_Y)
        #~Finger Body (changes with opening)
        WristBody.append(tcp_pos+0*hand_Z +b[i][0]*(gripper_width/2.0)*hand_X+b[i][1]*(gripper_length/2.0)*hand_Y)
        WristBody.append(tcp_pos+dist_tcp_to_intersection*hand_Z+b[i][0]*(gripper_width/2.0)*hand_X+b[i][1]*(gripper_length/2.0)*hand_Y)

    #~Collision dectection for each section
    for i in range(0,4):
        collision_finger = check_collision(FingerBody, ToteBody[i])
        collision_wrist = check_collision(WristBody, ToteBody[i])
        if collision_finger or collision_wrist:
            collision_section[i] = True

    #~Collision dectection (general) Are there any sections in collisions with tote?
    collision = any(collision_section==True for collision_section in collision_section)

    #~If multiple sections in collision, select the most important section for gripper relocation purposes
    sections_collision_sum = 0
    for i in range(0,4):
        if collision_section[i]:
            sections_collision_sum+=1

    mid_tcp_pos_list = []
    mid_tcp_pos = tcp_pos+dist_tcp_to_spatula*hand_Z
    if sections_collision_sum==1:#~ if only 1 section in collision, select that index for section collision
        for i in range(0,4):
            if collision_section[i]:
                section_index = i
    elif sections_collision_sum>1:
        for i in range(0,4):
            mid_tcp_pos_list.append(mid_tcp_pos) #~collision detection function accepts list (not arrays)
            mid_tcp_pos_collision = check_collision(mid_tcp_pos_list, ToteBody[i])
            if mid_tcp_pos_collision:
                section_index = i
                break

    if collision:
        print '[collision detection]', collision
        print '[collision detection] section:', section_index
    else:
        print '[collision detection]', collision

    #~Propose new location for gripper
    if collision:
        hand_Z = np.array([0,0,-1])
        edgeX = []#~index 1 is positive, index 0 is negative direction
        edgeY = []
        d=[-1,1]
        for i in range(0,2):
            edgeX.append(bin_mid_pos[0]+d[i]*(tote_width/2.0)*tote_X[0])
            edgeY.append(bin_mid_pos[1]+d[i]*(tote_length/2.0)*tote_Y[1])

        f_finger=[-1,1,1,-1]
        f_edge = [1,0,0,1]
        if section_index==0 or section_index==2:
            hand_X=np.array([0,f_finger[section_index],0])
            hand_Y=np.cross(hand_Z,hand_X)
            tcp_pos[0]=clamp(tcp_pos[0], edgeX[0]+gripper_width/2.0, edgeX[1]-gripper_width/2.0)
            tcp_pos[1]=edgeY[f_edge[section_index]]+f_finger[section_index]*(finger_opening+finger_thickness)
        elif section_index==1 or section_index==3:
            hand_X=np.array([f_finger[section_index],0,0])
            hand_Y=np.cross(hand_Z,hand_X)
            tcp_pos[0]=edgeX[f_edge[section_index]]+f_finger[section_index]*(finger_opening+finger_thickness)
            tcp_pos[1]=clamp(tcp_pos[1], edgeY[0]+gripper_width/2.0, edgeY[1]-gripper_width/2.0)

    #~ fig = plt.figure()md_tcp_pos)

    #~ for i in range(0,4):
        #~ plot_body(ToteBody[i],ax,'r')
    #~ plot_body(tcp_pos_list,ax,'b')
    #~ plot_body(WristBody,ax,'b')
    #~ plot_body(FingerBody[0:8],ax,'b')

    #~ plt.show()
    #~ plt.close()

    hand_orient_norm_collision_free = np.vstack([hand_X,hand_Y,hand_Z])

    return (collision, tcp_pos[0:2], hand_orient_norm_collision_free)

def get_object_properties(objId,objPose):
    ## initialize listener rospy
    # broadcast frame attached to object frame
    #~ pubFrame(br, pose=objPose, frame_id='obj', parent_frame_id='map', npub=5)

    #~ ******************* Object Frame   *******************
    obj_pose_tfm_list=matrix_from_xyzquat(objPose[0:3], objPose[3:7])
    obj_pose_tfm=np.array(obj_pose_tfm_list)
    obj_pose_orient=obj_pose_tfm[0:3,0:3]
    obj_pose_pos=obj_pose_tfm[0:3,3]

    #Normalized axes of the object frame
    obj_X=obj_pose_orient[:,0]/la.norm(obj_pose_orient[:,0])
    obj_Y=obj_pose_orient[:,1]/la.norm(obj_pose_orient[:,1])
    obj_Z=obj_pose_orient[:,2]/la.norm(obj_pose_orient[:,2])

    #Normalized object frame
    obj_pose_orient_norm=np.vstack((obj_X,obj_Y,obj_Z))
    obj_pose_orient_norm=obj_pose_orient_norm.transpose()

    obj_dim=rospy.get_param('obj')[str(objId)]
    obj_dim=obj_dim['dimensions']
    obj_dim=np.array(obj_dim)


    return (obj_dim, obj_X, obj_Y, obj_Z, obj_pose_orient_norm)

def capture_gelsight(is_node=False):
    if is_node:
        rospy.init_node('capture_gelsigth', anonymous=True)

    is_loop=True
    if rospy.get_param('have_robot'):
        while is_loop == True:
            try:
                bridge = CvBridge()
                #read ros sensor
                gel1 = rospy.wait_for_message("/rpi/gelsight/flip_raw_image", sensor_msgs.msg.Image, 1)
                gel2 = rospy.wait_for_message("/rpi/gelsight/flip_raw_image2", sensor_msgs.msg.Image, 1)
                #convert to cv2 image
                cv2_gel1 = bridge.imgmsg_to_cv2(gel1, 'rgb8') # Convert your ROS Image message to OpenCV2
                cv2_gel2 = bridge.imgmsg_to_cv2(gel2, 'rgb8') # Convert your ROS Image message to OpenCV2
                #cv2.imwrite('messigray.png',img)
                cv2.imwrite('/media/mcube/data/gelsight_calibration/calibrate_gelsight1'+str(rospy.get_time())+'.jpg',cv2_gel1)
                cv2.imwrite('/media/mcube/data/gelsight_calibration/calibrate_gelsight2'+str(rospy.get_time())+'.jpg',cv2_gel2)
                is_loop=False
            except:
                pass
    return

def get_tcp_pose(listener, tcp_offset = 0.0):
    #~get robot tcp pose (both virtual and real)
    if tcp_offset is None:
        pre_basepose = poseTransform([0.0,0.0,0.0,0.0,0.0,0.0,1.0], "link_6","map", listener)
    else:
        pre_basepose = poseTransform([0.0,0.0,tcp_offset,0.0,0.0,0.0,1.0], "link_6","map", listener)

    return pre_basepose

def get_joints():
    #~get robot joints (both virtual and real)
    for i in range(10):
        APCrobotjoints = rospy.wait_for_message("/joint_states" , sensor_msgs.msg.JointState, 3)
        if APCrobotjoints.name[0] == 'joint1':
            q0 = APCrobotjoints.position[0:6]
            break
        else: #~frank hack: if robot is real but no connection, initialize robot pose to be goARC
            q0 = [-0.0014,    0.2129,    0.3204,    0,    1.0374,   -0.0014]
    return q0

def move_cart(dx=0, dy=0, dz=0):
    #convert to mm
    dx=1000.*dx
    dy=1000.*dy
    dz=1000.*dz
    #Define ros services
    getCartRos = rospy.ServiceProxy('/robot1_GetCartesian', robot_GetCartesian)
    setCartRos = rospy.ServiceProxy('/robot1_SetCartesian', robot_SetCartesian)
    #read current robot pose
    c = getCartRos()
    #move robot to new pose
    setCartRos(c.x+dx, c.y+dy, c.z+dz, c.q0, c.qx, c.qy, c.qz)

def move_cart_hand(listener, dx=0, dy=0, dz=0, speedName = 'faste'):
    #Define ros services
    getCartRos = rospy.ServiceProxy('/robot1_GetCartesian', robot_GetCartesian)
    setCartRos = rospy.ServiceProxy('/robot1_SetCartesian', robot_SetCartesian)
    #read current robot pose in world frame
    c = getCartRos()
    #build current pose
    pose_world = np.array([c.x/1000., c.y/1000., c.z/1000., c.q0, c.qx, c.qy, c.qz])
    #convert to hand frame
    pose_hand = poseTransform(pose_world, "map", "link_6", listener)
    pose_hand[0] += dx 
    pose_hand[1] += dy
    pose_hand[2] += dz
    #convert back to world frame
    pose_world_new = poseTransform(pose_hand, "link_6", "map", listener)
    #move robot to new pose
    setSpeedByName(speedName = speedName)
    setCartRos(pose_world_new[0]*1000, pose_world_new[1]*1000, pose_world_new[2]*1000, 
            pose_world_new[3], pose_world_new[4], pose_world_new[5], pose_world_new[6])
    return


def unwrap(angle):
    #~unwrap an angle to the range [-pi,pi]
    if angle>np.pi:
        angle = np.mod(angle,2*np.pi)
        if angle>np.pi:
            angle = angle - 2*np.pi
    elif angle<-np.pi:
        angle = np.mod(angle,-2*np.pi)
        if angle<-np.pi:
            angle = angle + 2*np.pi
    return angle

def joint6_angle_list(angle):
	angle_half_range=2.0*np.pi
	angle_list = []
	index_list = [-2,-1,0,1,2]
	for term in index_list:
		angle_tmp = angle+term*2*np.pi

		if angle_tmp<=angle_half_range and angle_tmp>=-angle_half_range:
			angle_list.append(angle_tmp)
	return angle_list

def angle_shortest_dist(angle_current, angle_target_list):
    angle_target_array = np.asarray(angle_target_list)
    dist_array = abs(angle_target_array-angle_current)
    index_min = np.argmin(dist_array)
    return angle_target_array[index_min]

def record_rosbag(topics):
    #~ dir_save_bagfile = '/media/' + os.environ['HOME']+'/gelsight_grasping_data'
    dir_save_bagfile = os.environ['ARCDATA_BASE']+'/gelsight_grasping_data'
    make_sure_dir_exists(dir_save_bagfile)
    name_of_bag=time.strftime("%Y%m%d-%H%M%S")
    rosbag_proc = subprocess.Popen('rosbag record -q -O %s %s' % (name_of_bag, " ".join(topics)) , shell=True, cwd=dir_save_bagfile)
    return name_of_bag

def make_sure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def terminate_ros_node(s):
    list_cmd = subprocess.Popen("rosnode list", shell=True, stdout=subprocess.PIPE)
    list_output = list_cmd.stdout.read()
    retcode = list_cmd.wait()
    assert retcode == 0, "List command returned %d" % retcode
    for str in list_output.split("\n"):
        if (str.startswith(s)):
            os.system("rosnode kill " + str)

def get_hand_frame(obj_pose_orient_norm, obj_dim, listener, br):

    #~Define reference frames
    world_X, world_Y, world_Z, tote_X, tote_Y, tote_Z, tote_pose_pos = reference_frames(listener = listener, br=br)
    #~ Project all axes of object about Z-world axis
    proj_vecZ = np.abs(np.dot(world_Z,obj_pose_orient_norm))
    temp = proj_vecZ.argsort()

    #~sort all dimensions
    max_index=temp[2]
    secondmax_index=temp[1]
    min_index=temp[0]
    signed_proj_vecZ = np.dot(world_Z,obj_pose_orient_norm)
    if signed_proj_vecZ[max_index]<0:
        hand_Z=obj_pose_orient_norm[:,max_index]
    else:
        hand_Z=-obj_pose_orient_norm[:,max_index]

    # find smaller of the other two object dimensions
    obj_xyplane_dim_array = np.array([obj_dim[secondmax_index],obj_dim[min_index]])
    obj_smaller_xydim_index = np.argmin(np.fabs(obj_xyplane_dim_array))

    # Set hand X (finger vec) along smaller dimension vector
    if obj_smaller_xydim_index==0:
        hand_X=obj_pose_orient_norm[:,secondmax_index]
        grasp_width=obj_dim[secondmax_index]
    else:
        hand_X=obj_pose_orient_norm[:,min_index]
        grasp_width=obj_dim[min_index]

    #~define coordinate frame vectors
    hand_Y=np.cross(hand_Z, hand_X)

    return hand_X, hand_Y, hand_Z, grasp_width

def get_picking_params_from_7(objInput, objId, listener, br):
    try:
        obj_dim, obj_X, obj_Y, obj_Z, obj_pose_orient_norm = get_object_properties(objId, objInput)
    except:
        obj_dim = np.array([0.135, 0.055, 0.037])
        obj_pose_orient_norm = np.array([[ 0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    objPose = objInput
    graspPos = objPose[0:3]
    hand_X, hand_Y, hand_Z, grasp_width = get_hand_frame(obj_pose_orient_norm=obj_pose_orient_norm, obj_dim=obj_dim, listener=listener, br=br)

    return graspPos, hand_X, hand_Y, hand_Z, grasp_width

def get_picking_params_from_12(objInput):

    #~define variables
    grasp_begin_pt=np.array(objInput[0:3])
    hand_Z=np.array(objInput[3:6])

    #~define grasp pos
    grasp_depth=np.array(objInput[6])
    graspPos=grasp_begin_pt + hand_Z*grasp_depth
    grasp_width=np.array(objInput[7])

    #~define hand frame
    hand_X=np.array(objInput[8:11])
    hand_Y=np.cross(hand_Z, hand_X)

    return graspPos, hand_X, hand_Y, hand_Z, grasp_width

def drop_pose_transform(binId,rel_pose, BoxBody, place_pose, viz_pub, listener, br):
     #~define gripper home orientation
    base_pose = [0.,0.,0.,0.,1.,0.,0.] #~goarc hand pose
    matrix_base_pose= tfm.quaternion_matrix(base_pose[3:7])
    hand_orient_norm = matrix_base_pose[0:3,0:3]
    #~initialize arguments for placing functions
    finger_opening = gripper.getGripperopening()
    safety_margin=.035
    #~ get 3d bin and finger points
    finger_pts_3d = collision_detection.collisionHelper.getFingerPoints(finger_opening, [0,0,0], hand_orient_norm, False)
    bin_pts_3d = collision_detection.collisionHelper.getBinPoints(binId=binId,listener=listener, br=br)
    #~convert to 2d
    bin_pts = bin_pts_3d[:,0:2]
    finger_pts = finger_pts_3d[:,0:2]
    #~ ~perform coordinate transformation from object to gripper
    (drop_pose,final_object_pose) = pose_transform_precise_placing(rel_pose, BoxBody, place_pose, base_pose, bin_pts_3d, finger_pts, safety_margin, False, viz_pub)
    return drop_pose
