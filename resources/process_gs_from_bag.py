#!/usr/bin/env python


import sys
import os, time
import optparse
import json
import numpy as np
import json
import rospy
#sys.path.append(os.environ['PUSHING_BENCHMARK_BASE'] + '/Data')
#sys.path.append('../helper')
import scipy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy
import pdb
import tf
import rospy
import tf.transformations as tfm
import matplotlib.pyplot as plt
import cv2
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from shapely.geometry import Point, Polygon
import time # for sleep
import json
from ik.helper import *
from sensor_msgs.msg import *
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
import tf
import tf.transformations as tfm
import h5py
import optparse
from bisect import bisect
import sys
import subprocess
import numpy as np
from scipy.io import loadmat
from os import listdir
from os.path import isfile, join
import glob
import time
import pickle
import shutil


def get_tranf_matrix(extrinsics, camera_matrix):
    # Matrix for extrinsics
    translate = extrinsics[0:3]
    quaternion = extrinsics[3:7]
    extrinsics_matrix = np.dot(tfm.compose_matrix(translate=translate),tfm.quaternion_matrix(quaternion))[0:3]
    
    # Transformation matrix
    transformation_matrix = np.dot(camera_matrix,extrinsics_matrix)
    
    return transformation_matrix

def load_json_file(filename):
    
    with open(filename) as json_data:
        data = json.load(json_data)
        
    return data



if __name__=='__main__':
    
    final_time = 0
    bridge = CvBridge()
    data = {}
    data['images_0'] = []; data['t_images_0'] = []; data['xc'] = []; data['timeJSON'] = []
    data['images_1'] = []; data['t_images_1'] = []; data['xc2'] = []; data['timeJSON2'] = []
    data['vicon_to_world'] = []
    
    #   Load from BAG
    print('Loading bag files')
    folder_path = '/media/mcube/data/shapes_data/graps/tape_v2/' #.format(shape)
    onlyfiles = filter(os.path.isfile, glob.glob(folder_path + "*.bag")) #[f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    onlyfiles.sort(key=lambda x: os.path.getmtime(x))    
    
    
    
    for data_filename in onlyfiles: 
        initial_time_traj = None
        data_filename = data_filename[:-4]
        print data_filename        
        it1 = 0
        it0 = 0
        with rosbag.Bag(data_filename+'.bag', 'r') as bag:
            for topic, msg, t in bag.read_messages():
                if topic == '/rpi/gelsight/flip_raw_image':
                    cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
                    data['images_0'].append(cv_image)
                    if not initial_time_traj: initial_time_traj = t.to_sec()
                    data['t_images_0'].append(t.to_sec()-initial_time_traj+final_time)
                    final_time_traj = t.to_sec()-initial_time_traj+final_time
                    #cv_image = cv2.cvtColor(cv_image,cv2.COLOR_RGB2BGR) 
                    #cv2.imwrite(data_filename+'gs_{}.png'.format(data['t_images_0'][-1]),cv_image)
                    #directory = '/media/mcube/data/shapes_data/object_exploration/flashlight_track/p_{}/'.format(it0)
                    #if not os.path.exists(directory): os.makedirs(directory)
                    #cv2.imwrite(directory +'GS1_{}.png'.format(it0),cv_image); it0 +=1
                    #shutil.copy(folder_path + 'grasp_data/cart.npy', directory + 'cart.npy')
                    #shutil.copy(folder_path + 'grasp_data/wsg_0.pkl', directory + 'wsg_0.pkl')
                if topic == '/arc_1/rgb_bin0': 
                    cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
                    data['images_1'].append(cv_image)
                    if not initial_time_traj: initial_time_traj = t.to_sec()
                    data['t_images_1'].append(t.to_sec()-initial_time_traj+final_time)
                    final_time_traj = t.to_sec()-initial_time_traj+final_time
                    cv_image = cv2.cvtColor(cv_image,cv2.COLOR_RGB2BGR) 
                    #cv2.imwrite(data_filename+'_rgb_{}.png'.format(it1),cv_image); it1 +=1
            final_time = final_time_traj
            final_time = final_time_traj
    
    print('Done!')
    
    # Video time
    fps = len(data['t_images_0'])/(data['t_images_0'][-1]-data['t_images_0'][0])
    fps1 = len(data['t_images_1'])/(data['t_images_1'][-1]-data['t_images_1'][0])
    time_steps = np.arange(data['t_images_0'][0], data['t_images_0'][-1], 1.0/max(fps,fps1))
    
    print 'fps: ', fps
    print 'fps1: ', fps1
    
    aux_img2 = data['images_0'][0]
    for i,t in enumerate(time_steps):
        print i
        # find iterator closest to
        index0 = bisect(data['t_images_0'], t)
        img0 = data['images_0'][index0]
        index1 = bisect(data['t_images_1'], t)
        img1 = data['images_1'][index1]
        if i > 700:
            pc_path = '/media/mcube/data/shapes_data/processed_pointclouds/flashlight_track/new_{}_global.png'.format(index0)
        else:
            pc_path = '/media/mcube/data/shapes_data/processed_pointclouds/flashlight_track/{}_global.png'.format(index0)
            pc_path = '/home/mcube/Downloads/tape_global2.png'.format(index0)
        img2 = cv2.imread(pc_path)
        
        if img2 is not None:
            aux_img2 = copy.deepcopy(img2)
        else:
            img2 = copy.deepcopy(aux_img2)
        img2 = cv2.resize(img2, (640,480))
        if i< 313: #305:
            img2 = img2*0+255
        if i > 550: #941:
            break
        cv_image = np.concatenate([img1, img0], axis = 1)
        cv_image = np.concatenate([cv_image, img2], axis = 1)
        cv_image = cv2.cvtColor(cv_image,cv2.COLOR_RGB2BGR)
        cv2.imwrite(data_filename+'_comb_6_{}.png'.format(i),cv_image)
        cv_image = np.concatenate([img1, img2], axis = 1)
        
        cv_image = cv2.cvtColor(cv_image,cv2.COLOR_RGB2BGR)
        cv2.imwrite(data_filename+'_comb_7_{}.png'.format(i),cv_image)
    
    pdb.set_trace()
    for i in range(1): #range(2):
        fps = len(data['t_images_{}'.format(i)])/(data['t_images_{}'.format(i)][-1]-data['t_images_{}'.format(i)][0])
        cv2.destroyAllWindows()
        plt.close()
        # Initialize time and figure
        fig = plt.figure()
        it_im = 0
        t_im = np.array(data['t_images_{}'.format(i)]) - data['t_images_{}'.format(i)][0]
        time_steps -= time_steps[0]
        
        # For each time steps / frame
        for it in range(time_steps.shape[0]): #range(x_act.shape[0]):
            #if np.mod(it,5) > 0: continue;
            print 'Building frame:', it
            # Get real image
            while t_im[it_im]< time_steps[it]: #What is the right image from the sequence to plot?
                it_im += 1
            resize_rgb = cv2.resize(data['images_1'][it_im], (640,480))
            rgb_gs = np.concatenate([resize_rgb,data['images_0'][it_im]], axis=1)
            #plt.imshow(rgb_gs)
            plt.imshow(data['images_0'][it_im])
            
            plt.axis('off')
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            print 'Video in frame:', it
            # Convert canvas to image
            #fig.set_dpi(300)
            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            #   Img is rgb, convert to opencv's default bgr
            
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) 
            #[500:1400,0:2200] #[500:1400,500:2200] #[120:380,150:600] #[90:410,220:820]
            
            # Initialize video
            if it == 0:
                video_name = data_filename + '_rgb.avi'
                height, width, layers = img.shape
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                print 'video_name', video_name, fps, width, height
                video = cv2.VideoWriter(video_name,fourcc, fps, (width,height)) 
            
            # Record video
            video.write(img)
            print 'hi'

            
            
            '''
            # Create new figure
            img_name = data_filename + '_image_{}.png'.format(it)
            plt.savefig(img_name,bbox_inches='tight',pad_inches = 0)
            plt.close()
            fig = plt.figure()
            '''
        cv2.destroyAllWindows()
        video.release()
    
    
