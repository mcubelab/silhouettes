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
import glob
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
SHAPES_ROOT = os.getcwd().split("/silhouettes/")[0] + "/silhouettes/"




def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def make_kernal(n):
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
    return kernal

def calibration(img,background, gs_id=2, mask_bd=None):
    # print "len: " + str(len(img[0, 0, :]))
    def apply_mask_bd(im):
        if (mask_bd is not None):
            if (len(im[0, 0, :]) > 0):
                for i in range(len(im[0, 0, :])):
                    im[:, :, i] = im[:, :, i]*mask_bd
            else:
                im = im*mask_bd
        return im

    if gs_id == 1:
        M = np.load(SHAPES_ROOT + 'resources/M_GS1.npy')
        rows, cols, cha = img.shape
        imgw = cv2.warpPerspective(img, M, (cols, rows))
        imgw = apply_mask_bd(imgw)
        imgwc = imgw[10:, 65:572]
        # print "New gs1 shpae: " + str(imgwc.shape)

        bg_imgw = cv2.warpPerspective(background, M, (cols, rows))
        bg_imgw = apply_mask_bd(bg_imgw)
        bg_imgwc = bg_imgw[10:, 65:572]
        img_blur = cv2.GaussianBlur(bg_imgwc.astype(np.float32), (25, 25), 30)
        img_bs = imgwc.astype(np.int32) - img_blur.astype(np.int32) + np.mean(img_blur)

    elif gs_id == 2:
        M = np.load(SHAPES_ROOT + 'resources/M_GS1.npy')
        rows, cols, cha = img.shape
        imgw = cv2.warpPerspective(img, M, (cols, rows))
        imgw = apply_mask_bd(imgw)
        # print "GS2 after warp shpae: " + str(imgw.shape)
        imgwc = imgw[10:, 65:572]
        # print "New gs2 shpae: " + str(imgwc.shape)

        bg_imgw = cv2.warpPerspective(background, M, (cols, rows))
        bg_imgw = apply_mask_bd(bg_imgw)
        bg_imgwc = bg_imgw[10:, 65:572]
        img_blur = cv2.GaussianBlur(bg_imgwc.astype(np.float32), (25, 25), 30)
        img_bs = imgwc.astype(np.int32) - img_blur.astype(np.int32) + np.mean(img_blur)
    return img_bs.astype(np.uint8), imgwc


def contact_detection(im,im_ref,low_bar,high_bar):
    im_sub = im/(im_ref+1e-6)*70
    im_gray = im_sub.astype(np.uint8)
    im_canny = cv2.Canny(im_gray, low_bar, high_bar)

    kernal1 = make_kernal(10)
    kernal2 = make_kernal(10)
    kernal3 = make_kernal(30)
    kernal4 = make_kernal(20)
    img_d = cv2.dilate(im_canny, kernal1, iterations=1)
    img_e = cv2.erode(img_d, kernal1, iterations=1)
    img_ee = cv2.erode(img_e, kernal2, iterations=1)
    
    img_dd = cv2.dilate(img_ee, kernal3, iterations=1)
    img_eee = cv2.erode(img_dd, kernal4, iterations=1)
    img_label = np.stack((np.zeros(img_dd.shape),np.zeros(img_dd.shape),img_eee),axis = 2).astype(np.uint8)
    return img_label



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


    def close_gripper_f(self, grasp_speed=200, grasp_force=40):
        graspinGripper(grasp_speed=grasp_speed, grasp_force=grasp_force)

    def open_gripper(self):
        open(speed=200)

    def palpate(self, speed=200, force_list=[1, 10, 20, 40], save=False, path='', save_only_picture=False, i=0):
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
            time.sleep(1)
            dc.get_data(get_cart=False, get_gs1=(1 in self.gs_id), get_gs2=(2 in self.gs_id), get_wsg=True, save=save, directory=path, iteration=i)
            time.sleep(0.5)
            self.open_gripper()
            i += 1


    def check_patch_enough(self, path_img, path_ref):
        ref = cv2.imread(path_ref + 'GS{}'.format(self.gs_id[0]) + '.png')
        mask_bd = np.load(SHAPES_ROOT + 'resources/mask_GS{}.npy'.format(self.gs_id[0]))
        ref_bs, ref_warp = calibration(ref, ref, self.gs_id[0], mask_bd)
        im_temp = cv2.imread(path_img)
        im_bs, im_wp = calibration(im_temp, ref, self.gs_id[0], mask_bd)
        ## Remove background to the iamge and place its minimum to zero
        im_diff = im_wp.astype(np.int32) - ref_warp.astype(np.int32)
        im_diff_show = ((im_diff - np.min(im_diff))).astype(np.uint8)
        im_diff = im_diff - np.min(im_diff)
        ## Take into account different channels and create masks for the contact patch
        mask1 = (im_diff[:, :, 0]-im_diff[:, :, 1]) > 15
        mask2 = (im_diff[:, :, 1]-im_diff[:, :, 0]) > 15
        mask = ((mask1 + mask2)*255).astype(np.uint8)
        mask = rgb2gray(contact_detection(rgb2gray(im_wp).astype(np.float32), rgb2gray(ref_warp).astype(np.float32),20,50))
        ## Apply eroding to the image and dilatation
        kernal1 = make_kernal(30)
        kernal2 = make_kernal(10)
        mask = cv2.erode(mask, kernal2, iterations=1)
        mask = cv2.dilate(mask, kernal2, iterations=1)
        mask = cv2.dilate(mask, make_kernal(35), iterations=1)
        mask_color = cv2.erode(mask, kernal1, iterations=1).astype(np.uint8)   
        mask_pixels = np.sum(mask_color)/255
        print 'Number pixels: {}'.format(mask_pixels)
        return mask_pixels > 1000
    
    
    
    def perfrom_experiment(self, experiment_name='test', movement_list=[], save_only_picture=False, last_touch = 0, original_x = None,
        original_y = None):
        if last_touch == 0:
            # 1. We save the background image:
            dc = DataCollector(only_one_shot=False, automatic=True, save_only_picture=save_only_picture)
            dc.get_data(get_cart=False, get_gs1=(1 in self.gs_id), get_gs2=(2 in self.gs_id), get_wsg=False, save=True, directory=experiment_name+'/air', iteration=-1)
            print "Air data gathered"

        # 2. We perfomr the experiment:
        ini = time.time()
        if not os.path.exists(experiment_name): # If the directory does not exist, we create it
            os.makedirs(experiment_name)
        print 'last ', last_touch
        print 'left ', len(movement_list)
        for i in range(last_touch, len(movement_list)+last_touch):
            if i>last_touch:
                print path
                print "Done: " + str(i) + "/" + str(len(movement_list)) + ", Remaining minutes: " + str(((len(movement_list)-i)*(time.time()-ini)/i)/60.)
            if save_only_picture:
                path = experiment_name + '/'
                j = i
            else:
                path = experiment_name + '/p_' + str(i) + '/'
                j = i
            movement = movement_list[i-last_touch]
            is_good = False
            while not is_good:
                print 'hi'
                self.palpate(speed=200, force_list=self.force_list, save=True, path=path, save_only_picture=save_only_picture, i=j)
                print 'hi2'
                is_good = self.check_patch_enough(path_img = path + '/GS{}_{}'.format(self.gs_id[0],j)+ '.png', path_ref = experiment_name+'/air/')
                print 'hi3'
                if not is_good: print 'Not enough pixels in the mask'
            self.move_cart_mm(movement[0], movement[1], movement[2])
            print "moved"
            print 'movement: ', movement
            if original_x is not None:
                print ('Motion in gelsight: ' , original_x[i-last_touch], original_y[i-last_touch])
            i += 1
        if save_only_picture:
            path = experiment_name + '/'
            j = i
        else:
            path = experiment_name + '/p_' + str(i) + '/'
            j = 0
        is_good = False
        while not is_good:
            self.palpate(speed=200, force_list=self.force_list, save=True, path=path, save_only_picture=save_only_picture, i=j)
            is_good = check_patch_enough(path_img = path  + '/GS{}_{}'.format(self.gs_id[0],j) + '.png', path_ref = experiment_name+'/air/')
            if not is_good: print 'Not enough pixels in the mask'

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


    # cr.move_cart_mm(0, 0, 100)
    cr.move_joint(dj6=90, print_cart=False)
    print 'done!'
