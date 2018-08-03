from data_collector import DataCollector
from gripper import *
from ik.helper import *
from visualization_msgs.msg import *
from robot_comm.srv import *
from visualization_msgs.msg import *
from wsg_50_common.msg import Status
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge, CvBridgeError
import rospy, math, cv2, os, pickle
import numpy as np
import time

class ControlRobot():
    def __init__(self, gs_ids=[2], force_list=[1, 10, 20, 40]):
        self.gs_id = gs_ids
        self.force_list = force_list

        rospy.init_node('listener', anonymous=True) # Maybe we should only initialize one general node


    def move_cart_mm(self, dx=0, dy=0, dz=0):
        #Define ros services
        getCartRos = rospy.ServiceProxy('/robot1_GetCartesian', robot_GetCartesian)
        setCartRos = rospy.ServiceProxy('/robot1_SetCartesian', robot_SetCartesian)
        #read current robot pose
        c = getCartRos()
        #move robot to new pose
        setCartRos(c.x+dx, c.y+dy, c.z+dz, c.q0, c.qx, c.qy, c.qz)

    def set_cart_mm(self, cart):
        x, y, z, q0, qx, qy, qz = cart
        setCartRos = rospy.ServiceProxy('/robot1_SetCartesian', robot_SetCartesian)
        setCartRos(x, y, z, q0, qx, qy, qz)

    def move_joint(self, dj1=0, dj2=0, dj3=0, dj4=0, dj5=0, dj6=0, print_cart=False):
        getJointRos = rospy.ServiceProxy('/robot1_GetJoints', robot_GetJoints)
        setJointRos = rospy.ServiceProxy('/robot1_SetJoints', robot_SetJoints)
        j = getJointRos()
        j.j1 += dj1
        j.j2 += dj2
        j.j3 += dj3
        j.j4 += dj4
        j.j5 += dj5
        j.j6 += dj6
        setJointRos(j.j1, j.j2, j.j3, j.j4, j.j5, j.j6)

        if print_cart:
            a = raw_input("Press enter to get cartesian (wait until done):")
            getCartRos = rospy.ServiceProxy('/robot1_GetCartesian', robot_GetCartesian)
            c = getCartRos()
            print c


    def close_gripper_f(self, grasp_speed=50, grasp_force=40):
        graspinGripper(grasp_speed=grasp_speed, grasp_force=grasp_force)

    def open_gripper(self):
        open(speed=100)

    def palpate(self, speed=80, force_list=[1, 10, 20, 40], save=False, path='', save_only_picture=False, i=0):
        # 0. We create the directory
        if save is True and not os.path.exists(path): # If the directory does not exist, we create it
            os.makedirs(path)

        # 1. We get and save the cartesian coord.
        dc = DataCollector(only_one_shot=False, automatic=True, save_only_picture=save_only_picture)
        cart = dc.getCart()
        if save and not save_only_picture:
            np.save(path + '/cart.npy', cart)

        # 2. We get wsg forces and gs images at every set force and store them
        for force in force_list:
            self.close_gripper_f(grasp_speed=speed, grasp_force=force)
            print "Applying: " + str(force)
            time.sleep(0.2)
            dc.get_data(get_cart=False, get_gs1=(1 in self.gs_id), get_gs2=(2 in self.gs_id), get_wsg=True, save=save, directory=path, iteration=i)
            self.open_gripper()
            i += 1

    def perfrom_experiment(self, experiment_name='test', movement_list=[], save_only_picture=False):
        # 1. We save the background image:
        dc = DataCollector(only_one_shot=False, automatic=True, save_only_picture=save_only_picture)
        dc.get_data(get_cart=False, get_gs1=(1 in self.gs_id), get_gs2=(2 in self.gs_id), get_wsg=False, save=True, directory=experiment_name+'/air', iteration=-1)
        print "Air data gathered"

        # 2. We perfomr the experiment:
        ini = time.time()
        i = 0
        if not os.path.exists(experiment_name): # If the directory does not exist, we create it
            os.makedirs(experiment_name)
        for movement in movement_list:
            if i>0:
                print "Done: " + str(i) + "/" + str(len(movement_list)) + ", Remaining minutes: " + str(((len(movement_list)-i)*(time.time()-ini)/i)/60.)
            if save_only_picture:
                path = experiment_name + '/'
                j = i
            else:
                path = experiment_name + '/p_' + str(i) + '/'
                j = 0
            self.palpate(speed=40, force_list=self.force_list, save=True, path=path, save_only_picture=save_only_picture, i=j)
            self.move_cart_mm(movement[0], movement[1], movement[2])
            print "moved"
            i += 1
        if save_only_picture:
            path = experiment_name + '/'
            j = i
        else:
            path = experiment_name + '/p_' + str(i) + '/'
            j = 0
        self.palpate(speed=40, force_list=self.force_list, save=True, path=path, save_only_picture=save_only_picture, i=j)


if __name__ == "__main__":
    cr = ControlRobot()
    #cr.close_gripper_f()
    #cr.close_gripper_f(grasp_speed=40, grasp_force=10)
    #print 1
    #time.sleep(1)
    #cr.close_gripper_f(grasp_speed=40, grasp_force=15)
    #print 2
    #time.sleep(1)
    #cr.close_gripper_f(grasp_speed=40, grasp_force=20)
    #print 3
    #time.sleep(1)

    #cr.move_cart_mm(dx=5, dy=0, dz=0)
    #cr.palpate(speed=40, force_list=[10, 20, 30], save=True, path='air_palpate_test')


    # cr.move_cart_mm(0, 0, 5)
    cr.move_joint(dj6=-180, print_cart=True)
    print 'done!'
