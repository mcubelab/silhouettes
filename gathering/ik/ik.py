 #!/usr/bin/env python

import rospy
import geometry_msgs.msg, std_msgs.msg, sensor_msgs.msg
import visualization_msgs.msg, trajectory_msgs.msg,  moveit_msgs.msg
import json
from roshelper import ROS_Wait_For_Msg
import roshelper
from visualization_msgs.msg import *
import telnetlib
import tf.transformations as tfm
import numpy as np
import roslib; roslib.load_manifest("robot_comm")
from robot_comm.srv import *
import math
import subprocess, os, time, socket
import copy
from std_msgs.msg import Int32, String,Bool, Float64
from pr_msgs.msg import gelsightresult
import datetime
import helper
import spatula

impact_pub=rospy.Publisher('/impact_time', Bool, queue_size = 10, latch = False)
toviz = rospy.get_param('/toviz', True)
haverobot = rospy.get_param('/have_robot', True)
fastvirtual = rospy.get_param('/fast_virtual', False)

from ctypes import cdll, c_void_p, c_int
_dll = cdll.LoadLibrary(os.environ['CODE_BASE'] + '/catkin_ws/devel/lib/libikfast_python.so')
_ikfastAndFindBest = _dll['ikfastAndFindBest']
_ikfastAndFindBest.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
_fkfast = _dll['fkfastPython']
_fkfast.argtypes = [c_void_p, c_void_p]


enforceSlow = False

def shake(speed_name='superSaiyan'):
 #convert to mm
 dx=0. #mm
 dy=10. #mm
 dz=10. #mm
 #Define ros services
 getCartRos = rospy.ServiceProxy('/robot1_GetCartesian', robot_GetCartesian)
 setCartRos = rospy.ServiceProxy('/robot1_SetCartesian', robot_SetCartesian)
 setSpeed = rospy.ServiceProxy('/robot1_SetSpeed', robot_SetSpeed)
 #read current robot pose
 c = getCartRos()
 #move robot to new pose
 name2vel = {'gripperRotation' : (150,300), 'superSaiyan': (1200,120),'yolo': (800,90),'fastest': (400,70),'faster': (200,60),
             'fast': (100,30), 'slow': (50,15)}
 setSpeed(name2vel[speed_name][0], name2vel[speed_name][1])

 # if (c.x+dx) < 1.2*1000 and (c.x-dx)>0.8*1000 and (c.y+dy) < .55*1000 and (c.y-dy)>-.55*1000 and (c.z+dz) < 1.*1000 and (c.z-dz)>0.6*1000:
 for i in xrange(6):
     setCartRos(c.x+dx, c.y+dy, c.z, c.q0, c.qx, c.qy, c.qz)
     setCartRos(c.x-dx, c.y-dy, c.z, c.q0, c.qx, c.qy, c.qz)
     setCartRos(c.x, c.y, c.z+dz, c.q0, c.qx, c.qy, c.qz)
     setCartRos(c.x, c.y, c.z-dz, c.q0, c.qx, c.qy, c.qz)
 setCartRos(c.x, c.y, c.z, c.q0, c.qx, c.qy, c.qz)
 # else:
 #     print ('[Helper] Current position dangerous for shaking')
 #     print (c.x, c.y, c.z, c.q0, c.qx, c.qy, c.qz)
# p.setSpeedByName('fast')

def GetJoints(joint_topic):
    # get robot joints from published topic  # sensor_msgs/JointState
    while True:
        arc_robot_joints = ROS_Wait_For_Msg(joint_topic, sensor_msgs.msg.JointState).getmsg()
        if arc_robot_joints.name[0] == 'joint1':
            return arc_robot_joints.position[0:6]
class IK:
    # target_hand_pos = [x, y, z] (required)
    # target_hand_ori = [qx, qy, qz, qw] (required)
    # tip_hand_transform = [x,y,z, r,p,y]
    # target_joint_bb = [[joint_ind, lb, ub], [joint_ind, lb, ub], ...]  joint_ind: 1-6, don't use inf, use a big number instead.
    # from what time should it follow the orientation constraint of target

    # q0 = 6 joint angles in rad
    # qnom = nominal joint angles in rad  (will solve ik to find joint closest to qnom, if not specified q0 = qnom)
    def __init__(self, joint_topic = '/joint_states', target_tip_pos = None, target_tip_ori = None, q0 = None, qnom = None,
                 straightness = 0, pos_tol = 0.001, ori_tol = 0.01,
                 tip_hand_transform = [0,0,0, 0,0,0], inframebb = None,
                 target_link = 'link_6', target_joint_bb = None, weight = [1,1,1,2,1,1],
                 N = 10, ik_only = False, useMoveit = False, useFastIK = False):
        self.q0 = q0
        self.qnom = qnom

        self.target_tip_pos = target_tip_pos
        self.target_tip_ori = target_tip_ori
        self.pos_tol = pos_tol
        self.ori_tol = ori_tol
        self.tip_hand_transform = tip_hand_transform
        self.straightness = straightness
        self.inframebb = inframebb
        self.joint_topic = joint_topic
        self.target_link = target_link
        self.target_joint_bb = target_joint_bb

        self.N = N
        self.ik_only = ik_only
        self.ikServerAddress = ('localhost', 30000)

        self.useMoveit = useMoveit
        self.useFastIK = useFastIK
        self.weight = weight

    def plan(self):
        # plan() return an object includes attributes
        #   q_traj,
        #   snopt_info_iktraj,
        #   infeasible_constraint_iktraj
        #   snopt_info_ik

        # 1. Get a handle for the service
        #    todo: makes warning if ikTrajServer does not exist

        argin = {}
        # 2. prepare input arguments
        if self.q0 is not None:
            argin['q0'] = self.q0
        else:
            argin['q0'] = GetJoints(self.joint_topic)

        if self.qnom is None:
            self.qnom = argin['q0']
        argin['qnom'] = self.qnom

        # 2.1 transform tip pose to hand pose
        if self.target_tip_pos is not None:
            tip_hand_tfm_mat = np.dot( tfm.compose_matrix(translate=self.tip_hand_transform[0:3]),
                                       tfm.compose_matrix(angles=self.tip_hand_transform[3:6]) )
            tip_world_rot_mat = tfm.quaternion_matrix(self.target_tip_ori)
            tip_world_tfm_mat = np.dot(tfm.compose_matrix(translate=self.target_tip_pos) , tip_world_rot_mat)
            hand_tip_tfm_mat = np.linalg.inv(tip_hand_tfm_mat)
            hand_world_tfm_mat = np.dot(tip_world_tfm_mat, hand_tip_tfm_mat)
            target_hand_pos = tfm.translation_from_matrix(hand_world_tfm_mat)

            argin['target_hand_pos'] =  target_hand_pos.tolist()

        if self.target_tip_ori is not None:
            tip_hand_tfm_mat = tfm.compose_matrix(angles=self.tip_hand_transform[3:6])
            tip_world_rot_mat = tfm.quaternion_matrix(self.target_tip_ori)
            hand_tip_tfm_mat = np.linalg.inv(tip_hand_tfm_mat)
            hand_world_tfm_mat = np.dot(tip_world_tfm_mat, hand_tip_tfm_mat)
            target_hand_ori = tfm.quaternion_from_matrix(hand_world_tfm_mat)

            argin['target_hand_ori'] = roshelper.ros2matlabQuat(target_hand_ori.tolist())  # tolist: so that its serializable

        # 2.2 prepare other options
        argin['straightness'] = self.straightness
        argin['target_link'] = self.target_link
        argin['pos_tol'] = self.pos_tol
        argin['ori_tol'] = self.ori_tol

        if self.inframebb is not None:
            argin['inframebb'] = self.inframebb

        if self.target_joint_bb is not None:
            argin['target_joint_bb'] = self.target_joint_bb

        argin['N'] = self.N

        argin['tip_hand_transform'] = self.tip_hand_transform

        if self.ik_only:
            argin['ik_only'] = 1
        else:
            argin['ik_only'] = 0


        if self.useFastIK:
            js = fastik(argin['target_hand_pos']+argin['target_hand_ori'], list(argin['qnom'][0:6]), self.weight)

            if js is None:  # nosolution
                ret = {}
                ret['q_traj'] = []
                ret['snopt_info_iktraj'] = 20 ## hack
                ret['snopt_info_ik'] = 20 ## hack
                return Plan(ret)
            else:
                p = IKJoint(js, q0 = list(argin['q0'][0:6])).plan()
                return p

def _convertToFloatList(lst):
    return [float(x) for x in lst]

def fastik(targetpose, q0, weight = [1,1,1,2,1,1]):
    _targetpose = np.array(_convertToFloatList(targetpose))
    _q0 = np.array(_convertToFloatList(q0))
    _weight = np.array(_convertToFloatList(weight))
    _solution = np.array([0.0] * 6)
    _hassol = np.array([0])

    _ikfastAndFindBest(_solution.ctypes.data, _targetpose.ctypes.data, _weight.ctypes.data, _q0.ctypes.data, _hassol.ctypes.data)

    if _hassol[0] == 0:
        return None
    else:
        return _solution.tolist()

def fastfk(q0):
    _q0 = np.array(q0)
    _pose = np.array([0.0] * 7)
    _fkfast(_pose.ctypes.data, _q0.ctypes.data)
    return _pose.tolist()

def fastfk_python(q0):
    cmd = os.environ['APC_BASE']+'/catkin_ws/devel/lib/apc_planning/ikfast fk %.10f %.10f %.10f %.10f %.10f %.10f' % tuple(q0)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE) # x,y,z,qx,qy,qz,q0
    candidate_js = []

    line = proc.stdout.readline()
    r = [t(s) for t,s in zip((float,float,float,float,float,float,float),line.split())]

    return r

def isWildTraj(traj):
    if math.fabs(traj[0][0] - traj[-1][0]) > 3.14 / 2 or math.fabs(traj[0][2] - traj[-1][2]) > 3.14 / 2:
        return True
    return False

class IKGuarded:
    # target_hand_pos = [x, y, z] (required)
    # target_hand_ori = [qx, qy, qz, qw] (required)
    # tip_hand_transform = [x,y,z, r,p,y]
    # from what time should it follow the orientation constraint of target

    # here we'll always use IKFast
    # we assume it is always a straight path

    # q0 = 6 joint angles in rad
    def __init__(self, joint_topic = '/joint_states', target_tip_pos = None, target_tip_ori = None, q0 = None,
                 tip_hand_transform = [0,0,0, 0,0,0], step = 0.001, guarded_obj = None, useWarp = False, binId = 0, weight = [1,1,1,2,1,1]):
        self.q0 = q0

        if useWarp and haverobot:
            self.target_tip_pos = robot_calibration.cart_warp(target_tip_pos, binId)
        else:
            self.target_tip_pos = target_tip_pos

        self.target_tip_ori = target_tip_ori
        self.tip_hand_transform = tip_hand_transform
        self.joint_topic = joint_topic
        self.step = step
        self.guarded_obj = guarded_obj
        self.weight = weight

    def plan(self):
        q0 = None
        ret = {}
        # 2. prepare input arguments
        if self.q0 is not None:
            q0 = self.q0
        else:
            # get robot joints from published topic  # sensor_msgs/JointState
            q0 = GetJoints(self.joint_topic)

        # hack, need to refine
        tip_hand_tfm_mat = np.dot( tfm.compose_matrix(translate=self.tip_hand_transform[0:3]),
                                   tfm.compose_matrix(angles=self.tip_hand_transform[3:6]) )
        tip_world_rot_mat = tfm.quaternion_matrix(self.target_tip_ori)
        tip_world_tfm_mat = np.dot(tfm.compose_matrix(translate=self.target_tip_pos) , tip_world_rot_mat)
        hand_tip_tfm_mat = np.linalg.inv(tip_hand_tfm_mat)
        hand_world_tfm_mat = np.dot(tip_world_tfm_mat, hand_tip_tfm_mat)
        target_hand_pos = tfm.translation_from_matrix(hand_world_tfm_mat).tolist()

        tip_hand_tfm_mat = tfm.compose_matrix(angles=self.tip_hand_transform[3:6])
        tip_world_rot_mat = tfm.quaternion_matrix(self.target_tip_ori)
        hand_tip_tfm_mat = np.linalg.inv(tip_hand_tfm_mat)
        hand_world_tfm_mat = np.dot(tip_world_tfm_mat, hand_tip_tfm_mat)
        target_hand_ori = tfm.quaternion_from_matrix(hand_world_tfm_mat).tolist() # x,y,z,q1,q2,q3,q0
        target_hand_ori = roshelper.ros2matlabQuat(target_hand_ori)    # x,y,z,q0,q1,q2,q3

        qN = fastik(target_hand_pos + target_hand_ori, q0, self.weight)
        if qN is None:
            # no solution
            ret['q_traj'] = [];   ret['snopt_info_iktraj'] = 20 ; ret['snopt_info_ik'] = 20
            return Plan(ret)

        p0 = fastfk(q0)  # output x,y,z,q0,q1,q2,q3
        pN = target_hand_pos + target_hand_ori

        # flip the q0 quaternion if flipped is closer to qN
        if np.linalg.norm(np.array(p0[3:7])-np.array(pN[3:7])) > np.linalg.norm(np.array(p0[3:7])+np.array(pN[3:7])):
            p0[3:7] = (-np.array(p0[3:7])).tolist()

        N = int(np.linalg.norm(np.array(pN[0:3]) - np.array(p0[0:3])) / self.step)+1
        delta = (np.array(pN) - np.array(p0)) / N
        lastq = q0
        q_traj = []
        for i in xrange(0,N+1):
            pi = np.array(p0) + delta * float(i)
            pi[3:7] = pi[3:7] / np.linalg.norm(pi[3:7])
            qi = fastik(pi, lastq)
            if qi is None:
                ret['q_traj'] = [];   ret['snopt_info_iktraj'] = 20 ; ret['snopt_info_ik'] = 20
                return Plan(ret)
            q_traj.append(qi)
            lastq = qi

        ret['snopt_info_iktraj'] = 1;   ret['snopt_info_ik'] = 1 ## hack

        ret['q_traj'] = np.array(q_traj).transpose().tolist()
        return GuardedPlan(ret, self.guarded_obj)

class IKJoint:
    def __init__(self, target_joint_pos, joint_topic = '/joint_states', q0 = None):
        self.target_joint_pos = target_joint_pos
        self.q0 = q0
        if self.q0 is not None:
            self.q0 = q0[0:6]
        self.joint_topic = joint_topic

    def plan(self):
        N = 30
        p = Plan()

        if self.q0 is None:
            self.q0 = GetJoints(self.joint_topic)

        for i in xrange(N):
            t = (i*1.0)/(N-1)
            q_t = (np.array(self.target_joint_pos)*t + np.array(self.q0)*(1-t)).tolist()
            p.q_traj.append(q_t)

        p.setExecPoints(1)
        return p

class EvalPlan:
    def __init__(self, script = None):
        self.script = script

    def execute(self):
        exec(self.script)
        return True

    def setSpeedByName(self, speedName):
        pass

    def visualize(self,hand_param=.055):
        pass

    def visualizeBackward(self,hand_param=.055):
        pass

    def executeBackward(self):
        return True

class Plan:
    def __init__(self, data = None):
        self.q_traj = []
        self.data = None
        if data is not None:
            self.data = data
            self.q_traj = np.array(data['q_traj']).transpose().tolist()
        self.ikTrajServer_pub = rospy.Publisher('/move_group/display_planned_path_2', moveit_msgs.msg.DisplayTrajectory, queue_size=10)
        self.exec_joint_pub = rospy.Publisher('/virtual_joint_states', sensor_msgs.msg.JointState, queue_size=10)
        self.speedName = None
        self.speed = None  # should be a tuple: (tcpSpeed, oriSpeed); None means don't change speed
        self.speedNameBackwards = None
        self.speedBackwards = None  # should be a tuple: (tcpSpeed, oriSpeed); None means don't change speed
        self.numExecPoints = 10

    def setExecPoints(self, n):
        self.numExecPoints = n

    def is_guarded(self):
        return False

    def setSpeedByName(self, speedName = 'fast'):
        # superSaiyan, yolo, fastest, faster, fast, slow
        name2vel = {'gripperRotation' : (150,300), 'superSaiyan': (1200,120),'yolo': (800,90),'fastest': (400,70),'faster': (200,60),
                    'fast': (100,30), 'slow': (50,15)}
        self.speedName = speedName
        self.speed = name2vel[speedName]

    def setSpeedByNameBackwards(self, speedName = 'fast'):
        # superSaiyan, yolo, fastest, faster, fast, slow
        name2vel = {'gripperRotation' : (150,300),'superSaiyan': (1200,120),'yolo': (800,90),'fastest': (400,70),'faster': (200,60),
                    'fast': (100,30), 'slow': (50,15)}
        self.speedNameBackwards = speedName
        self.speedBackwards = name2vel[speedName]

    def success(self):
        if self.data is not None:
            return (self.data['snopt_info_iktraj'] < 10 and self.data['snopt_info_ik'] < 10)
        return True

    def _visualize(self, backward=False, hand_param=None):
        if not toviz:
            return

        extra_joint_names = []
        extra_joint_pos = []
        if hand_param is None:
            joint_topic = '/joint_states'
            hand_param = 0.055

            for i in range(100):
                APCrobotjoints = ROS_Wait_For_Msg(joint_topic, sensor_msgs.msg.JointState).getmsg()
                q0 = APCrobotjoints.position
                if len(q0) == 8 or len(q0) == 2:   # need to write better
                    hand_q = q0[-2:]
                    hand_param = math.fabs(hand_q[0])
                    break

                if len(q0) == 12:
                    hand_q = q0[6:8]
                    hand_param = math.fabs(hand_q[0])
                    extra_joint_names = APCrobotjoints.name[8:]
                    extra_joint_pos = q0[8:]
                    break

        jointTrajMsg = trajectory_msgs.msg.JointTrajectory()
        q_traj = self.q_traj  # a N-by-6 matrix

        jointTrajMsg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4',
        'joint5', 'joint6', 'wsg_50_gripper_base_joint_gripper_left', 'wsg_50_gripper_base_joint_gripper_right']

        jointTrajMsg.joint_names.extend(extra_joint_names)
        speedup = 30
        delta = int(len(q_traj) / 30)
        if delta < 1:
            delta = 1

        if backward:
            rng = range(len(q_traj)-1, -1, -delta)
            if len(q_traj) % delta != 0: rng += [0]
        else:
            rng = range(0, len(q_traj), delta)
            if len(q_traj) % delta != 0: rng += [len(q_traj)-1]

        for i in rng:
            pt = trajectory_msgs.msg.JointTrajectoryPoint()
            for j in range(6):
                pt.positions.append(q_traj[i][j])
            pt.positions.append(-hand_param)  # open gripper, these two numbers should be planned somewhere
            pt.positions.append(hand_param)
            pt.positions.extend(extra_joint_pos)
            jointTrajMsg.points.append(pt)

        robotTrajMsg = moveit_msgs.msg.RobotTrajectory()
        robotTrajMsg.joint_trajectory = jointTrajMsg

        dispTrajMsg = moveit_msgs.msg.DisplayTrajectory()
        dispTrajMsg.model_id = 'irb_1600id'
        dispTrajMsg.trajectory.append(robotTrajMsg)

        self.ikTrajServer_pub.publish(dispTrajMsg)

    def visualize(self,hand_param=None):
        self._visualize(backward=False,hand_param=hand_param)

    def visualizeForward(self,hand_param=None):
        self._visualize(backward=False,hand_param=hand_param)

    def visualizeBackward(self,hand_param=None):
        self._visualize(backward=True,hand_param=hand_param)

    def _execute(self, backward=False):
        if backward and self.speedBackwards is not None:
            setSpeed(self.speedBackwards[0], self.speedBackwards[1])
        if self.speed is not None:
            setSpeed(self.speed[0], self.speed[1])

        clearBuffer = rospy.ServiceProxy('/robot1_ClearJointPosBuffer', robot_ClearJointPosBuffer)   # should move service name out
        addBuffer = rospy.ServiceProxy('/robot1_AddJointPosBuffer', robot_AddJointPosBuffer)
        executeBuffer = rospy.ServiceProxy('/robot1_ExecuteJointPosBuffer', robot_ExecuteJointPosBuffer)

        q_traj = self.q_traj  # a N-by-6 matrix

        R2D = 180.0 / math.pi

        try:
            if not haverobot:
                raise Exception('No robot')
            rospy.wait_for_service('/robot1_ClearJointPosBuffer', timeout = 0.5)
            clearBuffer()

            for j in getrng(len(q_traj), self.numExecPoints, backward):
                addBuffer(q_traj[j][0]*R2D, q_traj[j][1]*R2D, q_traj[j][2]*R2D,
                 q_traj[j][3]*R2D, q_traj[j][4]*R2D, q_traj[j][5]*R2D)
            ret = executeBuffer()

            if ret.ret == 1:
                return True
            else:
                print '[Robot] executeBuffer() error:', ret.ret, ret.msg
                return False
        except Exception as e:
            if e.message == 'No robot' or e.message.startswith( 'timeout exceeded while waiting for service' ):
                print e.message
                print '[Robot] Robot seeems not connected, skipping execute()'

                jnames = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
                js = sensor_msgs.msg.JointState()
                for j in getrng(len(q_traj), 10, backward):
                    js.name  = jnames
                    js.position = q_traj[j]
                    self.exec_joint_pub.publish(js)
                    if not fastvirtual:
                        rospy.sleep(0.05)

                return True  # should be something else, set True so that primitive can run without robot
            else:
                print e.message
                print '[Robot] Error in motion execution. Likely a collision. Waiting for controller comeback.'
                wait_controller_comeback()
                print '[Robot] Controller back'
                return False  # possibly due to motion supervision and then service timeout

    def execute(self):
        return self._execute(backward=False)

    def executeForward(self):
        return self._execute(backward=False)

    def executeBackward(self):
        return self._execute(backward=True)

class GraspingGuard(): #Gelsight Sensor
     def __init__(self):
         from multiprocessing import Lock
         self.mutex = Lock()
         self.value = None
         self.init_msg = None
         self.sub = rospy.Subscriber('/rpi/gelsight/deflection', Int32, self.callback)


         if rospy.has_param('/rpi_params'):
             self.threshold = rospy.get_param("/rpi_params/gelsight_contact_index")
         else:
             self.threshold = 1 #CHANGE to correct index
             ##self.threshold_high = 260

         print "[GraspingGuard] Grasping guard initiated"

     def callback(self, x):
         with self.mutex:
             self.value = x.data

     def prep(self):  # will be call when guarded move starts
         assert haverobot
         rospy.sleep(0.5)
         prep_done = False

         while (not prep_done):
             data1 = rospy.wait_for_message("/rpi/gelsight/deflection", Int32, 0.5).data
             data2 = rospy.wait_for_message("/rpi/gelsight/deflection", Int32, 0.5).data
             data3 = rospy.wait_for_message("/rpi/gelsight/deflection", Int32, 0.5).data
             if data1 == data2 == data3:
                 self.init_msg = data1
                 prep_done = True

         print "[Grasping Guard] Read initial index: ", self.threshold

     def test(self):  # will be called for each step of guarded move
          with self.mutex:
             print  (abs(self.value-self.init_msg) > self.threshold)
             return (abs(self.value-self.init_msg) > self.threshold) #and (self.value < self.threshold_high)

class SuctionGuard(): #Hall Effect sensor
    def __init__(self):
        from multiprocessing import Lock
        self.mutex = Lock()
        self.value = None
        self.init_msg = None
        self.sub = rospy.Subscriber('/rpi/hall_level', Float64, self.callback)
        if rospy.has_param('/rpi_params'):
            self.threshold = rospy.get_param("/rpi_params/hall_threshold")
        else:
            self.threshold = 50 #CHANGE
        print "[SuctionGuard] Suction guard initiated"

    def callback(self, x):
        with self.mutex:
            self.value = x.data

    def prep(self):  # will be call when guarded move starts
        rospy.sleep(0.5)
        self.init_msg = rospy.wait_for_message("rpi/hall_level", Float64, 3).data
        print "[Suction Guard] Read initial value: ", self.init_msg

    def test(self):  # will be called for each step of guarded move
        with self.mutex:
            return abs(self.value - self.init_msg) > self.threshold

class WeightGuard(): #Loadstar Weight
    def __init__(self, binNum, threshold = None):
        from multiprocessing import Lock
        self.mutex = Lock()
        self.value = [None, None, None]
        self.init_msg = [None, None, None]
        self.guard_stream_name = ['ws_stream'+str(binNum - 1), 'ws_stream'+str(binNum), 'ws_stream'+str(binNum + 1)]
        self.sub = [None, None, None]
        if binNum > 0 and binNum <= 3:
            self.sub[0] = rospy.Subscriber(self.guard_stream_name[0], Float64, self.callback0)
        self.sub[1] = rospy.Subscriber(self.guard_stream_name[1], Float64, self.callback1)
        if binNum < 3:
            self.sub[2] = rospy.Subscriber(self.guard_stream_name[2], Float64, self.callback2)
        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = 100.0
        print "[WeightGuard] Weight guard initiated"
    def callback0(self, x):
        with self.mutex:
            self.value[0] = x.data
    def callback1(self, x):
        with self.mutex:
            self.value[1] = x.data
    def callback2(self, x):
        with self.mutex:
            self.value[2] = x.data

    def prep(self):  # will be call when guarded move starts
        for i in range(0, 3):
            if self.sub[i] is not None:
                self.init_msg[i] = rospy.wait_for_message(self.guard_stream_name[i], Float64, 3).data
        print "[Weight Guard] Read initial value: ", self.init_msg


    def test(self):  # will be called for each step of guarded move
        with self.mutex:
            for i in range(0, 3):
                if self.value[i] is not None and abs(self.value[i]-self.init_msg[i]) > self.threshold:
                    return True
            return False

def wait_controller_comeback():
    ping = rospy.ServiceProxy('/robot1_Ping', robot_Ping)
    i = 0
    while not rospy.is_shutdown():
        try:
            ping()
            return
        except:
            print '[Robot] Waiting for robot to come back (%d).' % i
            i += 1
            rospy.sleep(1)

class GuardedPlan(Plan):
    def __init__(self, data = None, guard_obj = None):
        Plan.__init__(self, data)
        self.numExecPoints = len(self.q_traj)
        self.j_stopped = 0 # where it stopped because of touch
        self.guard_obj = guard_obj
        self.guarded_speed = (40,40)

    def setExecPoints(self, n):
        print "setExecPoints is not supported for GuardedPlan."

    def is_guarded(self):
        return True

    def plan_finished(self):
        return self.j_stopped == len(self.q_traj)-1

    def setSpeedByName(self, speedName = 'fast'):
        rospy.logwarn('Setting speed in guarded move, only backwards speed will be modified')
        self.setSpeedByNameBackwards(speedName)

    def _execute(self, backward=False):
        if backward and self.speedBackwards is not None:
            setSpeed(self.speedBackwards[0], self.speedBackwards[1])
        if self.speed is not None:
            setSpeed(self.guarded_speed[0], self.guarded_speed[1])


        setSpeed(self.guarded_speed[0], self.guarded_speed[1])

        setJoints = rospy.ServiceProxy('/robot1_SetJoints', robot_SetJoints)   # should move service name out

        q_traj = self.q_traj  # a N-by-6 matrix
        R2D = 180.0 / math.pi

        try:
            rospy.set_param('is_contact', False)
            if not haverobot:
                raise Exception('No robot')
            setZone(1)
            rospy.wait_for_service('/robot1_SetJoints', timeout = 0.5)

            if self.guard_obj is not None:
                self.guard_obj.prep()
            if not backward:  #move forward
                for j in getrng(len(q_traj), self.numExecPoints, False):
                    if self.guard_obj is None or self.guard_obj.test() == False:
                        ret = setJoints(q_traj[j][0]*R2D, q_traj[j][1]*R2D, q_traj[j][2]*R2D,
                              q_traj[j][3]*R2D, q_traj[j][4]*R2D, q_traj[j][5]*R2D)

                        if ret.ret != 1:
                            self.j_stopped = max(0, j - 10)  # hack: start from more back when executeBackward
                            print '[Robot] setJoints() error:', ret.ret, ret.msg
                            setZone(1)
                            return False
                        self.j_stopped = j
                    else:
                        self.j_stopped = j - 1
                        if self.j_stopped < 0: self.j_stopped = 0
                        print '[Guarded Move] Contact detected!'
                        impact_msgs = Bool()
                        impact_msgs.data = True
                        impact_pub.publish(impact_msgs)
                        rospy.set_param('is_contact', True)
                        break
                setZone(1)
                return True
            else:             #move backward
                setSpeed(100, 30) # fast
                # setSpeed(200, 60) # faster
                for j in xrange(self.j_stopped, -1, -10):
                    ret = setJoints(q_traj[j][0]*R2D, q_traj[j][1]*R2D, q_traj[j][2]*R2D,
                          q_traj[j][3]*R2D, q_traj[j][4]*R2D, q_traj[j][5]*R2D)

                    if ret.ret != 1:
                        print '[Robot] setJoints() error:', ret.ret, ret.msg
                        setZone(1)
                        return False
                if j != 0:
                    ret = setJoints(q_traj[j][0]*R2D, q_traj[j][1]*R2D, q_traj[j][2]*R2D,
                          q_traj[j][3]*R2D, q_traj[j][4]*R2D, q_traj[j][5]*R2D)

                    if ret.ret != 1:
                        print '[Robot] setJoints() error:', ret.ret, ret.msg
                        setZone(1)
                        return False

                setZone(1)
                return True


        except Exception as e:
            if e.message == 'No robot' or e.message.startswith( 'timeout exceeded while waiting for service' ):
                print e.message
                print '[Robot] Robot seeems not connected, skipping execute()'

                jnames = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
                js = sensor_msgs.msg.JointState()
                for j in getrng(len(q_traj), 10, backward):
                    js.name  = jnames
                    js.position = q_traj[j]
                    self.exec_joint_pub.publish(js)
                    if not fastvirtual:
                        rospy.sleep(0.05)
                return True  # should be something else, set True so that primitive can run without robot
            else:
                print e.message
                print '[Robot] Error in motion execution. Likely a collision. Waiting for controller comeback.'
                #rospy.sleep(3)  # hack to wait the controller to come back
                wait_controller_comeback()
                print '[Robot] Controller back'
                return False  # possibly due to motion supervision and then service timeout


    def _execute_forward_more(self, step=1):
        if self.speed is not None:
            setSpeed(self.speed[0], self.speed[1])

        setJoints = rospy.ServiceProxy('/robot1_SetJoints', robot_SetJoints)   # should move service name out

        q_traj = self.q_traj  # a N-by-6 matrix
        R2D = 180.0 / math.pi

        try:
            if not haverobot:   raise Exception('No robot')

            setZone(1)
            rospy.wait_for_service('/robot1_SetJoints', timeout = 0.5)

            for j in range(self.j_stopped,min(len(q_traj),self.j_stopped+step+1)):
                print 'Executing forward a bit more'
                ret = setJoints(q_traj[j][0]*R2D, q_traj[j][1]*R2D, q_traj[j][2]*R2D,
                          q_traj[j][3]*R2D, q_traj[j][4]*R2D, q_traj[j][5]*R2D)
                self.j_stopped = j
            setZone(1)
            return True


        except Exception as e:
            if e.message == 'No robot' or e.message.startswith( 'timeout exceeded while waiting for service' ):
                print '[Robot] Robot seeems not connected, skipping execute_forward_more()'

                jnames = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
                js = sensor_msgs.msg.JointState()
                for j in range(self.j_stopped,min(len(q_traj),self.j_stopped+step+1)):
                    js.name  = jnames
                    js.position = q_traj[j]
                    self.exec_joint_pub.publish(js)
                    if not fastvirtual:
                        rospy.sleep(0.05)

                return True  # should be something else, set True so that primitive can run without robot
            else:
                print '[Robot] Error in motion execution (more). Likely a collision. Waiting for controller comeback.'
                #rospy.sleep(3)  # hack to wait the controller to come back
                wait_controller_comeback()
                print '[Robot] Controller back'
                return False  # possibly due to motion supervision and then service timeout

def getrng(length, numExecPoints, backward):
    if numExecPoints > 1:
        delta = int(length / numExecPoints)
        if delta < 1:
            delta = 1

        if backward:
            rng = range(length-1, -1, -delta)
            rng += [0]
        else:
            rng = range(0, length, delta)
            rng += [length-1]
    elif numExecPoints == 1:
        if backward:
            rng = [0]
        else:
            rng = [-1]
    else:
        print 'Invalid numExecPoints', numExecPoints
    return rng

def setSpeedByName(speedName = 'fast'):
    if enforceSlow:
        speedName = 'slow'  # hack
    # superSaiyan, yolo, fastest, faster, fast, slow
    name2vel = {'superSaiyan': (1200,120),'yolo': (800,90),'fastest': (400,70),'faster': (200,60),
                'fast': (100,30), 'slow': (50,15)}
    speed = name2vel[speedName]
    setSpeed(speed[0], speed[1])

def setSpeed(tcp=100, ori=30):
    if not haverobot:
        return

    setSpeed_ = rospy.ServiceProxy('/robot1_SetSpeed', robot_SetSpeed)

    print '[Robot] setSpeed(%.1f, %.1f)' % (tcp, ori)
    try:
        rospy.wait_for_service('/robot1_SetSpeed', timeout = 0.5)
        setSpeed_(tcp, ori)
        return True
    except Exception as e:
        print e.message
        print '[Robot] Robot seeems not connected, skipping setSpeed()'
        return False

def setZone(zone = 1):
    if not haverobot:
        return

    setZone_ = rospy.ServiceProxy('/robot1_SetZone', robot_SetZone)

    print '[Robot] setZone(%d)' % (zone)
    try:
        rospy.wait_for_service('/robot1_SetZone', timeout = 0.5)
        setZone_(zone)
        return True
    except Exception as e:
        e.message
        print '[Robot] Robot seems not connected, skipping setZone()'
        return False


def setMotionSupervision(value = 50):
    if not haverobot:
        return

    setMotionSupervision_ = rospy.ServiceProxy('/robot1_SetMotionSupervision', robot_SetMotionSupervision)

    print '[Robot] setMotionSupervision(%d)' % (value)
    try:
        rospy.wait_for_service('/robot1_SetMotionSupervision', timeout = 0.5)
        setMotionSupervision_(value)
        return True
    except Exception as e:
        print e.message
        print '[Robot] Robot seems not connected, skipping setMotionSupervision()'
        return False


def setMotionSupervisionByName(name = 'normal'):
    name2val = {'high': 150,'normal': 90, 'low': 40}
    return setMotionSupervision(name2val[name])


def setAcc(acc=1, deacc=1):
    if not haverobot:
        return
    setAcc_ = rospy.ServiceProxy('/robot1_SetAcc', robot_SetAcc)

    print '[Robot] setAcc(%.1f, %.1f)' % (acc, deacc)
    try:
        rospy.wait_for_service('/robot1_SetAcc', timeout = 0.5)
        setAcc_(acc, deacc)
        return True
    except:
        print '[Robot] Robot seeems not connected, skipping setAcc()'
        return False

def generatePlan(q0, target_tip_pos, target_tip_ori, tip_hand_transform, speed, guard_on=None, backwards_speed=None, plan_name = None):
    joint_topic = '/joint_states'
    execution_possible = True
    if guard_on != None:
        planner = IKGuarded(q0 = q0, target_tip_pos = target_tip_pos, target_tip_ori = target_tip_ori, tip_hand_transform=tip_hand_transform, joint_topic=joint_topic, guarded_obj = guard_on)
        plan = planner.plan()
    else:
        planner = IK(q0 = q0, target_tip_pos = target_tip_pos, target_tip_ori = target_tip_ori, tip_hand_transform=tip_hand_transform, joint_topic=joint_topic, useFastIK=True)
        plan = planner.plan()
        plan.setSpeedByName(speed)

    if backwards_speed is not None:
        plan.setSpeedByNameBackwards(backwards_speed)

    s = plan.success()
    if not s:
        rospy.logerr('[Generate Plan] Plan Failed')
        execution_possible = False
        qf = None
    else:
		#~unwrap angle of joint 6
		#~ angle_list = helper.joint6_angle_list(plan.q_traj[-1][5])
		#~ best_angle = helper.angle_shortest_dist(q0[5], angle_list)
		#~ plan.q_traj[-1][5] = best_angle
		qf = plan.q_traj[-1]

    if plan_name is not None:
        rospy.logdebug('[Generate Plan] Build plan %s',plan_name)
        rospy.logdebug('[Generate Plan] Target position %s',target_tip_pos)
        rospy.logdebug('[Generate Plan] Target orientation %s',target_tip_ori)
        rospy.logdebug('[Generate Plan] Tip hand transform %s',tip_hand_transform)
        rospy.logdebug('[Generate Plan] Initial joint state %s',q0)
        rospy.logdebug('[Generate Plan] Computed joint state %s',qf)
        rospy.logdebug('[Generate Plan] Execution possible %s', execution_possible)

    return plan, qf, execution_possible

def executePlanForward(plans, withPause, withViz = False):
    for numOfPlan in range(0, len(plans)):
        if withViz:
            plans[numOfPlan].visualize()
        helper.pauseFunc(withPause)
        plans[numOfPlan].execute()

def executePlanBackward(plans, withPause, withViz = False):
    for numOfPlan in range(0, len(plans)):
        if withViz:
            plans[len(plans)-numOfPlan-1].visualizeBackward()
        helper.pauseFunc(withPause)
        plans[len(plans)-numOfPlan-1].executeBackward()

def setCartJ(pos, ori):
    if not haverobot:
        return
    param = (np.array(pos) * 1000).tolist() + ori[3:4] + ori[0:3]
    print '[Robot] setCartJ(%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f)' % tuple(param)

    setCartJRos_ = rospy.ServiceProxy('/robot1_SetCartesianJ', robot_SetCartesian)

    try:
        rospy.wait_for_service('/robot1_SetCartesianJ', timeout = 0.5)
        setCartJRos_(*param)
        return True
    except Exception as e:
        if e.message == 'No robot' or e.message.startswith( 'timeout exceeded while waiting for service' ):
            print '[Robot] Robot seeems not connected, skipping setCartJ()'
            return True
        return False

def setJoint(q):
    if not haverobot:
        return
    param = q
    print '[Robot] setJoint(%.1f, %.1f, %.1f, %.1f, %.1f, %.1f)' % tuple(q)

    setJointRos_ = rospy.ServiceProxy('/robot1_SetJoints', robot_SetJoints)

    try:
        rospy.wait_for_service('/robot1_SetJoints', timeout = 0.5)
        setJointRos_(*param)
        return True
    except Exception as e:
        if e.message == 'No robot' or e.message.startswith( 'timeout exceeded while waiting for service' ):
            print '[Robot] Robot seeems not connected, skipping setJoint()'
            IKJoint(target_joint_pos = np.deg2rad(q).tolist()).plan().execute()
            return True
        return False

def setCart(pos, ori):
    param = (np.array(pos) * 1000).tolist() + ori[3:4] + ori[0:3]
    print '[Robot] setCart(%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f)' % tuple(param)

    setCartRos_ = rospy.ServiceProxy('/robot1_SetCartesian', robot_SetCartesian)

    try:
        if not haverobot:   raise Exception('No robot')

        rospy.wait_for_service('/robot1_SetCartesian', timeout = 0.5)
        setCartRos_(*param)
        return True
    except Exception as e:
        if e.message == 'No robot' or e.message.startswith( 'timeout exceeded while waiting for service' ):
            print '[Robot] Robot seeems not connected, skipping setCart()'
            # show it virtually
            IKGuarded(target_tip_pos= pos, target_tip_ori =ori, step = 0.01).plan().execute()
            return True
        return False

def getCart():
    if not haverobot:
        return

    getCartRos_ = rospy.ServiceProxy('/robot1_GetCartesian', robot_GetCartesian)

    try:
        rospy.wait_for_service('/robot1_GetCartesian', timeout = 0.5)
        cart_pose=getCartRos_()
        cart_pose_list=[cart_pose.x/1000.0,cart_pose.y/1000.0,cart_pose.z/1000.0,cart_pose.qx,cart_pose.qy,cart_pose.qz,cart_pose.q0]
        return cart_pose_list
    except:
        print '[Robot] Robot seeems not connected, skipping setCart()'
        return False

def setTool(pos, ori):
    if not haverobot:
        return
    param = (np.array(pos) * 1000).tolist() + ori[3:4] + ori[0:3]
    print '[Robot] setTool(%f, %f, %f, %f, %f, %f, %f)' % tuple(param)

    setTool_ = rospy.ServiceProxy('/robot1_SetTool', robot_SetTool)

    try:
        rospy.wait_for_service('/robot1_SetTool', timeout = 0.5)
        setTool_(*param)
        return True
    except:
        print '[Robot] Robot seeems not connected, skipping setTool()'
        return False



# activateCSS
  # * @param refFrame The coordinate system the soft direction is related to.
  # *                 CSS_REFFRAME_TOOL: 1 Softness direction will be in relation to current tool.
                    # CSS_REFFRAME_WOBJ: 2 Softness direction will be in relation to current work object.
  # * @param refOrient This argument gives the possibility to rotate the coordinate system described by RefFrame.
  # * @param softDir  The Cartesian direction in which the robot will be soft. Soft direction is in relation
  # *                 to RefFrame.
  # *                 CSS_X := 1;
                    # CSS_Y := 2;
                    # CSS_Z := 3;
                    # CSS_XY := 4;
                    # CSS_XZ := 5;
                    # CSS_YZ := 6;
                    # CSS_XYZ := 7;
                    # CSS_XYRZ := 8;
  # * @param stiffness This argument describes how strongly the robot tries to move back to the reference
  # *                  point when it is pushed away from that point. It is a percentage of a configured
  # *                  value where 0 gives no spring effect of going back to the reference point. 50 mean 50 %
  # * @param stiffnessNonSoftDir This argument sets the softness for all directions that are not defined as soft by the argument SoftDir.
  # * @param allowMove When this switch is used movement instructions will be allowed during the activated
  # *                  soft mode. Note that using \AllowMove will internally increase the value of the
  # *                  stiffness parameter.
  # * @param ramp This argument defines how fast the softness is implemented, as a percentage of
  # *             the value set by the system parameter Activation smoothness time. Can be set to
  # *             between 1 and 500%.

CSS_REFFRAME_TOOL = 1
CSS_REFFRAME_WOBJ = 2

CSS_X = 1
CSS_Y = 2
CSS_Z = 3
CSS_XY = 4
CSS_XZ = 5
CSS_YZ = 6
CSS_XYZ = 7
CSS_XYRZ = 8

CSS_REFFRAME_name = {1: 'CSS_REFFRAME_TOOL', 2: 'CSS_REFFRAME_WOBJ'}
CSS_DIR_name = {1: 'CSS_X', 2: 'CSS_Y', 3: 'CSS_Z', 4: 'CSS_XY', 5: 'CSS_XZ', 6: 'CSS_YZ', 7: 'CSS_XYZ', 8: 'CSS_XYRZ'}

def activateCSS(refFrame = 1, softDir = 3, stiffness = 50,
                stiffnessNonSoftDir = 50, allowMove = True, ramp = 100, refOrient = (0,0,0,1)):
    if not haverobot:
        return
    activateCSS_ = rospy.ServiceProxy('/robot1_ActivateCSS', robot_ActivateCSS)

    print '[Robot] activateCSS(%s, %s)' % (CSS_REFFRAME_name[refFrame], CSS_DIR_name[softDir])
    try:
        rospy.wait_for_service('/robot1_ActivateCSS', timeout = 0.5)
        activateCSS_(refFrame, refOrient[3], refOrient[0], refOrient[1], refOrient[2],
                     softDir, stiffness, stiffnessNonSoftDir, allowMove, ramp)   # quaternion is qw,qx,qy,qz for abb_node
        return True
    except:
        print '[Robot] Robot seeems not connected, skipping activateCSS()'
        return False

# pose [x,y,z, qx,qy,qz,qw] is the tcp pose we want to move to after CSS ended. x,y,z in meter.
# if None then will be the current tcp pose

def deactivateCSS(pose = None):
    if not haverobot:
        return
    deactivateCSS_ = rospy.ServiceProxy('/robot1_DeactivateCSS', robot_DeactivateCSS)
    if pose is None:
        getCart_ = rospy.ServiceProxy('/robot1_GetCartesian', robot_GetCartesian)

    print '[Robot] deactivateCSS()'
    try:
        rospy.wait_for_service('/robot1_DeactivateCSS', timeout = 0.5)
        rospy.wait_for_service('/robot1_GetCartesian', timeout = 0.5)

        if pose is None:
            p = getCart_()
            pose = [p.x, p.y, p.z, p.qx, p.qy, p.qz, p.q0]
        else:
            pose = [pose[0]*1000.0, pose[1]*1000.0, pose[2]*1000.0] + pose[3:7]
        print '[Robot] deactivateCSS(mm):', pose
        deactivateCSS_(roshelper.poselist2pose(pose))
        return True
    except:
        print '[Robot] Robot seeems not connected, skipping deactivateCSS()'
        return False
