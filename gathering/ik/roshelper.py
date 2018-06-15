
import rospy
import chan    # python channel https://chan.readthedocs.org/en/latest/
import geometry_msgs.msg
import std_msgs
import sensor_msgs.msg
import tf.transformations as tfm
import tf
import traceback
import numpy as np
from wsg_50_common.srv import *
import wsg_50_common.msg
import subprocess, os, signal

from marker_helper import createMoveControls, createMeshMarker, createCubeMarker, createPointMarker2
from random import randint, uniform
from visualization_msgs.msg import Marker


from plyfile import PlyData, PlyElement
import json

def readPlyFile(obj_id):
    x = PlyData.read(open(os.environ['CODE_BASE'] + '/catkin_ws/src/apc_posest/src/models/%s.ply' % obj_id))
    return x
    

def get_obj_dim(objId):
    jsonFilename = os.environ['CODE_BASE']+'/catkin_ws/src/apc_config/object_data/objectDictionary.json'
    with open(jsonFilename) as data_file:
        data = json.load(data_file)
    
    object_dim = data[objId]
    object_dim[0]/=1000.0 #object_dim[0]=object_dim[0]/1000.0
    object_dim[1]/=1000.0
    object_dim[2]/=1000.0
    return object_dim

def terminate_ros_node(s):
    list_cmd = subprocess.Popen("rosnode list", shell=True, stdout=subprocess.PIPE)
    list_output = list_cmd.stdout.read()
    retcode = list_cmd.wait()
    assert retcode == 0, "List command returned %d" % retcode
    for str in list_output.split("\n"):
        if (str.startswith(s)):
            os.system("rosnode kill " + str)

ntfretry = 40
retryTime = 0.05
secBefore = 0.5

class ROS_Wait_For_Msg:
    #init_node=True if you call use this class outside of an rosnode
    def __init__(self, topic_name, msgtype, init_node=False, sub = None):
        self.init_node = init_node
        self.data = None
        self.channel = chan.Chan()
        self.topic_name = topic_name
        self.msgtype = msgtype
        
        if self.init_node:
            rospy.init_node('listener', anonymous=True)

    def __del__(self):
        if self.init_node:
            rospy.signal_shutdown('ROS_Wait_For_Msg done')

    def callback(self, data):
        self.channel.put(data)
        self.sub.unregister()   

    def getmsg(self):
        self.sub = rospy.Subscriber(self.topic_name, self.msgtype, self.callback)
        #print 'Waiting in ROS_Wait_For_Msg for %s...' % self.topic_name
        return self.channel.get()
        # do we need to remove the rosnode
        

def ros2matlabQuat(qros):  # qxqyqzqw -> qwqxqyqz
    qmat = [qros[3]]
    qmat.extend(qros[0:3])
    return qmat

from geometry_msgs.msg import Pose
def poselist2pose(poselist):
    pose = Pose()
    pose.position.x = poselist[0]
    pose.position.y = poselist[1]
    pose.position.z = poselist[2]
    pose.orientation.x = poselist[3]
    pose.orientation.y = poselist[4]
    pose.orientation.z = poselist[5]
    pose.orientation.w = poselist[6]
    return pose

def pose2list(pose):
    return [pose.position.x, pose.position.y, pose.position.z, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

def pubFrame(br, pose=[0,0,0,0,0,0,1], frame_id='obj', parent_frame_id='map', npub=5):
    if len(pose) == 7:
        ori = tuple(pose[3:7])
    elif len(pose) == 6:
        ori = tfm.quaternion_from_euler(*pose[3:6])
    else:
        print 'Bad length of pose'
        return 
    
    pos = tuple(pose[0:3])
    
    for j in range(npub):
        rospy.sleep(0.01)
        br.sendTransform(pos, ori, rospy.Time.now(), frame_id, parent_frame_id)

#there is a ROS library that does things for you. You can query this library
#and it will do math for you.
#pt is a point
#homeFrame is a string labeling the frame we care about
#targetFrame is a string see above
#homeFrame is a frame XYZ list (list of list, it's kind of like a matrix)
#targetFrame is also a frame XYZ list
#listener is a ROS listener
def coordinateFrameTransform(pt, homeFrame, targetFrame, listener):
    pose = geometry_msgs.msg.PoseStamped()
    pose.header.frame_id = homeFrame
    for i in range(ntfretry):
        try:
            t = rospy.Time(0)
            pose.header.stamp = t
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.position.z = pt[2]
            pose_target = listener.transformPose(targetFrame, pose)
            return pose_target
        except Exception as e: #(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print e.message
            print '[coordinateFrameTransform] failed to transform'
            print '[coordinateFrameTransform] targetFrame %s homeFrame %s, retry %d' % (targetFrame, homeFrame, i)
            if i == ntfretry - 1:
                raise e
    print traceback.format_exc()
    rospy.sleep(retryTime)


def coordinateFrameTransformList(pt, homeFrame, targetFrame, listener):
    return pose2list(coordinateFrameTransform(pt, homeFrame, targetFrame, listener).pose )[0:3]


# given a pose in list, transform to target frame and return a list
# coordinateFrameTransform return a PoseStamped
def poseTransform(pose, homeFrame, targetFrame, listener):
    _pose = geometry_msgs.msg.PoseStamped()
    _pose.header.frame_id = homeFrame
    if len(pose) == 6:
        pose.append(0)
        pose[3:7] = tfm.quaternion_from_euler(pose[3], pose[4], pose[5]).tolist()
    
    _pose.pose.position.x = pose[0]
    _pose.pose.position.y = pose[1]
    _pose.pose.position.z = pose[2]
    _pose.pose.orientation.x = pose[3]
    _pose.pose.orientation.y = pose[4]
    _pose.pose.orientation.z = pose[5]
    _pose.pose.orientation.w = pose[6]
    #while not rospy.is_shutdown():
    for i in range(ntfretry):
        try:
            t = rospy.Time(0)
            _pose.header.stamp = t
            _pose_target = listener.transformPose(targetFrame, _pose)
            p = _pose_target.pose.position
            o = _pose_target.pose.orientation
            return [p.x, p.y, p.z, o.x, o.y, o.z, o.w]
        except Exception as e: #(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print e.message
            print '[poseTransform] failed to transform'
            print '[poseTransform] targetFrame %s homeFrame %s, retry %d' % (targetFrame, homeFrame, i)
            if i == ntfretry - 1:
                raise e
            rospy.sleep(retryTime)
            
    print traceback.format_exc()
    return None

# given a ROS Pose(), transform to target frame and return as a ROS Pose()
def rosposeTransform(pose, homeFrame, targetFrame, listener):
    _pose = geometry_msgs.msg.PoseStamped()
    _pose.header.frame_id = homeFrame
    #while not rospy.is_shutdown():
    for i in range(ntfretry):
        try:
            t = rospy.Time(0)
            _pose.header.stamp = t
            _pose.pose = pose
            _pose_target = listener.transformPose(targetFrame, _pose)
            return _pose_target.pose
        except Exception as e: #(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print e.message
            print '[rosposeTransform] failed to transform'
            print '[rosposeTransform] targetFrame %s homeFrame %s, retry %d' % (targetFrame, homeFrame, i)
            if i == ntfretry - 1:
                raise e
            rospy.sleep(retryTime)
    print traceback.format_exc()
    return None

def lookupTransform(homeFrame, targetFrame, listener):
    #while not rospy.is_shutdown():
    for i in range(ntfretry):
        try:
            t = rospy.Time(0)
            (trans,rot) = listener.lookupTransform(homeFrame, targetFrame, t)
            return (trans,rot)
        except Exception as e: #(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print e.message
            print '[lookupTransform] failed to transform'
            print '[lookupTransform] targetFrame %s homeFrame %s, retry %d' % (targetFrame, homeFrame, i)
            if i == ntfretry - 1:
                raise e
            rospy.sleep(retryTime)
    print traceback.format_exc()
    return None, None


def lookupTransformList(homeFrame, targetFrame, listener):
    #while not rospy.is_shutdown():
    for i in range(ntfretry):
        try:
            t = rospy.Time(0)
            (trans,rot) = listener.lookupTransform(homeFrame, targetFrame, t)
            return list(trans) + list(rot)
        except Exception as e: #(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print e.message
            print '[lookupTransform] failed to transform'
            print '[lookupTransform] targetFrame %s homeFrame %s, retry %d' % (targetFrame, homeFrame, i)
            if i == ntfretry - 1:
                raise e
            rospy.sleep(retryTime)
    print traceback.format_exc()
    return None

def checkGraspStatus(drop_thick=0.000001 ): # finger gap =0.002m = .5 mm
    joint_topic = '/joint_states'
    rospy.sleep(0.5)
    while True:
        APCrobotjoints = ROS_Wait_For_Msg(joint_topic, sensor_msgs.msg.JointState).getmsg() 
        q0 = APCrobotjoints.position
        
        if len(q0) == 12:
            q0 = q0[6:8]    # first 6 are the robot joints, then 7 and 8 are the fingers, used for virtual
            break
        if len(q0) == 2:
            q0 = q0[0:2]    # just the first 2 for fingers, used for real robot
            break
        
    gripper_q0=np.fabs(q0)
    
    if gripper_q0[0] < drop_thick:
        print '[Grasp] ***************'
        print '[Grasp] Could not grasp'
        print '[Grasp] ***************'
        execution_possible = False
    else:
        print '[Grasp] ***************'
        print '[Grasp] Grasp Successful'
        print '[Grasp] ***************'
        execution_possible = True
    return execution_possible   

fastvirtual = rospy.get_param('/fast_virtual', False)

def getGripperopening():
    if rospy.get_param('have_robot')==False:
        while True:
            joint_topic = '/joint_states'
            APCrobotjoints = rospy.wait_for_message(joint_topic, sensor_msgs.msg.JointState, timeout=5)
            q0 = APCrobotjoints.position
            if not rospy.get_param('/have_robot'):
                q0 = q0[6:8]    # first 6 are the robot visualizeObjectsjoints, then 7 and 8 are the fingers, used for virtual
                break
            else:
                if 'wsg_50_gripper_base_joint_gripper_left' in APCrobotjoints.name:
                    q0 = q0[0:2]    # just the first 2 for fingers, used for real robot
                    break
        gripper_q0=np.fabs(q0)
        print 'gripper_q0',gripper_q0
        full_grasp_opening=2*gripper_q0[0]
    else:
        gripper_topic = '/wsg_50_driver/status'
        gripper_status = rospy.wait_for_message(gripper_topic, wsg_50_common.msg.Status)
        full_grasp_opening = gripper_status.width/1000.
    
    return full_grasp_opening
    

# Poses = [["elmers_glue", [x,y,z,qx,qy,qz,qw]], ["elmers_glue", [x,y,z,qx,qy,qz,qw]]]

vizpub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
def visualizeObjects(Poses, br, frame_id = '/map', rgba = None):
    if rgba is None:
        rgba = [uniform(0, 1), uniform(0, 1), uniform(0, 1), 1]
    
    for i, obj_pose in enumerate(Poses):
        r = randint(0,2000) # use a random number to distinguish between multiple items
        obj_id = obj_pose[0]
        m = createCubeMarker(offset = obj_pose[1][0:3], orientation = obj_pose[1][3:7], frame_id = frame_id, marker_id = r, rgba = rgba, scale = get_obj_dim(obj_id))
        vizpub.publish(m.markers[0])
        #rospy.sleep(0.1)
    
    #rospy.sleep(0.5)


def visualizeObjectsDict(Poses, br, frame_id = '/map', rgba = None):
    if rgba is None:
        rgba = [uniform(0, 1), uniform(0, 1), uniform(0, 1), 0.5]
    
    for i, obj_pose in enumerate(Poses):
        r = randint(0,20000) # use a random number to distinguish between multiple items
        
        if 'score' in obj_pose:
            rgba = (np.array([1.0,0.0,0.0,0.5]) * obj_pose['score'] + np.array([0.0,1.0,0.0,0.5]) * (1.0-obj_pose['score'])).tolist()
        
        obj_id = obj_pose['label']
        m = createCubeMarker(offset = obj_pose['pose'][0:3], orientation = obj_pose['pose'][3:7], frame_id = frame_id, marker_id = r, rgba = rgba, scale = get_obj_dim(obj_id))
        vizpub.publish(m.markers[0])
        
        #import pdb; pdb.set_trace()
        if 'pca' in obj_pose:
            if 'latent' in obj_pose:
                l = obj_pose['latent']
            else:
                l = [0.4, 0.03, 0.01] # dummy
            r = randint(0,20000) # use a random number to distinguish between multiple items
            m = createCubeMarker(offset = obj_pose['pca'][0:3], orientation = obj_pose['pca'][3:7], frame_id = frame_id, marker_id = r, 
            rgba = [rgba[0]/2.0,rgba[1]/2.0,rgba[2]/2.0,1], scale = (np.array(l)).tolist())  # show pca
            vizpub.publish(m.markers[0])
        #rospy.sleep(0.1)
    
    #rospy.sleep(0.5)


def visualizeObjectsCloud(Poses, br, frame_id = '/map'):
    
    for i, obj_pose in enumerate(Poses):
        r = randint(0,20000) # use a random number to distinguish between multiple items
        obj_id = obj_pose[0]
        obj_frame_id = obj_pose[0] + '%d' % i
        #pubFrame(br, obj_pose[1], obj_frame_id, frame_id)
        
        ply = readPlyFile(obj_id)
        m = createPointMarker2(points = ply.elements[0].data, frame_id = 'map', namespace= obj_frame_id, marker_id = r, pose = obj_pose[1]) #, rgba= [uniform(0, 1), uniform(0, 1), uniform(0, 1), 1])
        
        pub.publish(m)
        #rospy.sleep(0.5)
    
    #rospy.sleep(0.5)
    


