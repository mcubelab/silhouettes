#!/usr/bin/env python

import rospy
from wsg_50_common.srv import *
from std_srvs.srv import *
from std_msgs.msg import Bool
import time
import sensor_msgs.msg
from ik.roshelper import ROS_Wait_For_Msg
from ik.roshelper import *
# from ik.helper import *

class NoGripperException(Exception):
    pass

ErrorMessage = 'Gripper not connected, skipping gripper command:'

exec_joint_pub = rospy.Publisher('/virtual_joint_states', sensor_msgs.msg.JointState, queue_size=10)

havegripper = rospy.get_param('/have_gripper', True)
toRetry = False  # keep retry when motion failed

def move(move_pos=110, move_speed=50):
    move_pos=max(min(move_pos,110),0)
    #move pos range: 0 - 110 mm
    #move speed range: 5- 450 mm/s
    ack()
    command = 'move'
    srv=rospy.ServiceProxy('/wsg_50_driver/%s' % command, Move)

    #~publish command to impact_time is contact was not detected
    if (rospy.get_param('is_contact') == False) and (rospy.get_param('is_record') == True) and havegripper:
        impact_pub=rospy.Publisher('/impact_time', Bool, queue_size = 10, latch = False)
        impact_msgs = Bool()
        impact_msgs.data = False
        impact_pub.publish(impact_msgs)

    while True:
        try:
            if not havegripper:
                raise NoGripperException('No gripper')

            error = srv(move_pos, move_speed)
            print '[Gripper] move, return:', error
            break
        except NoGripperException:

            print '[Gripper] move,', ErrorMessage, command

            # publish to joint state publisher for visualization without real hand
            jnames = ['wsg_50_gripper_base_joint_gripper_left', 'wsg_50_gripper_base_joint_gripper_right']

            js = sensor_msgs.msg.JointState()
            js.name  = jnames
            js.position = [-move_pos / 2.0 / 1000.0, move_pos / 2.0 / 1000.0]
            exec_joint_pub.publish(js)
        if not toRetry: break
        time.sleep(0.5)

def open(speed=100):

    #we assume that grippers opens fully with speed = 100
    ack()
    if havegripper: time.sleep(0.1)
    move(109, speed)
    if havegripper: time.sleep(0.1)
    if havegripper: move(109, speed)
    if havegripper: time.sleep(0.1)
    if havegripper: move(109, speed)


def close(speed=100):

    #we assume that grippers closes fully with speed = 100
    ack()
    if havegripper: time.sleep(0.1)
    move(3,speed)

def grasp(move_pos=108, move_speed=80, grasp_type = 'default'):
    ack()
    command = 'grasp'
    srv=rospy.ServiceProxy('/wsg_50_driver/%s' % command, Move)

    while True:
        try:
            if not havegripper:
                raise NoGripperException('No gripper')

            error = srv(move_pos, move_speed)

            print '[Gripper] grasp %d, return:' % move_pos, error
            break
        except NoGripperException:
            print '[Gripper] grasp,', ErrorMessage, command

            # publish to joint state publisher for visualization without real hand
            jnames = ['wsg_50_gripper_base_joint_gripper_left', 'wsg_50_gripper_base_joint_gripper_right']

            js = sensor_msgs.msg.JointState()
            js.name  = jnames
            if grasp_type == 'default':
                js.position = [-move_pos / 2.0 / 1000.0, move_pos / 2.0 / 1000.0]
            elif grasp_type == 'in':
                js.position = [0,0]
            elif grasp_type == 'out':
                move_pos = 108
                js.position = [-move_pos / 2.0 / 1000.0, move_pos / 2.0 / 1000.0]

            exec_joint_pub.publish(js)
        if not toRetry: break
        time.sleep(5)

def release(move_pos=109, move_speed=50):
    ack()
    command = 'release'
    srv=rospy.ServiceProxy('/wsg_50_driver/%s' % command, Move)

    while True:
        try:
            if not havegripper:
                raise NoGripperException('No gripper')
            error = srv(move_pos, move_speed)
            print '[Gripper] release, return:', error
            break
        except NoGripperException:
            print '[Gripper] release,', ErrorMessage, command

            # publish to joint state publisher for visualization without real hand
            jnames = ['wsg_50_gripper_base_joint_gripper_left', 'wsg_50_gripper_base_joint_gripper_right']

            js = sensor_msgs.msg.JointState()
            js.name  = jnames
            js.position = [-move_pos / 2.0 / 1000.0, move_pos / 2.0 / 1000.0]
            exec_joint_pub.publish(js)

        if not toRetry: break
        time.sleep(5)

def homing():
    ack()
    command = 'homing'
    srv=rospy.ServiceProxy('/wsg_50_driver/%s' % command, Empty)
    try:
        error = srv()
    except:
        print '[Gripper] homing,', ErrorMessage, command

def ack():
    command = 'ack'
    srv=rospy.ServiceProxy('/wsg_50_driver/%s' % command, Empty)
    while True:
        try:
            if not havegripper:   raise NoGripperException('No gripper')
            error = srv()
            break
        except NoGripperException:
            print '[Gripper] ack,', ErrorMessage, command

        if not toRetry: break
        time.sleep(0.5)


def set_force(val = 5):
    command = 'set_force'
    srv=rospy.ServiceProxy('/wsg_50_driver/%s' % command, Conf)

    while True:
        try:
            error = srv(val)
            break
        except:
            print '[Gripper] set_force,', ErrorMessage, command

        if not toRetry: break
        time.sleep(0.5)


def nailControl(openNail=True, maxOpen=False):
    # openNail: pointing straight forward if true, else fold away
    # maxOpen: max opening possible

    # TODO: remove or replace next 3 lines once we have an interface with the nails
    ack()
    command = 'move'
    srv=rospy.ServiceProxy('/wsg_50_driver/%s' % command, Move)
    #time.sleep(0.05)

    while True:
        try:
            error = srv(move_pos, move_speed)
            print '[Nails] move, return:', error
            break
        except:

            print '[Nails] move,', ErrorMessage, command

            # publish to joint state publisher for visualization without real hand
            jnames = ['joint_nail_right', 'joint_nail_left']

            js = sensor_msgs.msg.JointState()
            js.name  = jnames

            if openNail:
                js.position = [0.0, 0.0]
            if maxOpen:
                js.position = [-0.17453292519, -0.17453292519]
            if not openNail:
                js.position = [3.14159265359, 3.14159265359]

            exec_joint_pub.publish(js)
        if not toRetry: break
        time.sleep(0.5)

def grasp_in(grasp_speed=50,grasp_force=10):

    #~publish command to impact_time is contact was not detected
    if (rospy.get_param('is_contact') == False) and (rospy.get_param('is_record') == True) and havegripper:
        impact_pub=rospy.Publisher('/impact_time', Bool, queue_size = 10, latch = False)
        impact_msgs = Bool()
        impact_msgs.data = False
        impact_pub.publish(impact_msgs)


    ack()
    gripper_opening=(getGripperopening())*1000
    print gripper_opening
    if havegripper: time.sleep(0.1)
    if havegripper: set_force(grasp_force)
    if havegripper: time.sleep(0.1)
    grasp_pos = max(gripper_opening - 2, 0)
    #grasp_pos = 0
    grasp(grasp_pos, grasp_speed, grasp_type = 'in')
    if havegripper: time.sleep(0.5)

def grasp_out(grasp_speed=50,grasp_force=40):
    ack()
    gripper_opening=(getGripperopening())*1000
    if havegripper: time.sleep(0.1)
    if havegripper: set_force(grasp_force)
    if havegripper: time.sleep(0.1)
    grasp(gripper_opening+2, grasp_speed, grasp_type = 'out')
    if havegripper: time.sleep(0.5)

def flush_open():
    ack()
    grasp_out(grasp_force=5)
    gripper_opening=(getGripperopening())*1000
    #print gripper_opening
    move(gripper_opening-8,70)

# Use this function
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

if __name__=='__main__':
    homing()

    for i in range(1000):
        print 'Trial:', i
        move(100,50)
        move(20,50)
        #grasp()
    # time.sleep(10)
    # release()
# time.sleep(1)
