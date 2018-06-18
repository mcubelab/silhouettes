import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from labeller import Labeller
from depth_helper import *

import pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os, sys
import cv2
import scipy
from keras import backend as K
import scipy.io
import cPickle as pickle
import h5py
import deepdish as dd
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
SHAPES_ROOT = os.getcwd().split("/silhouettes/")[0] + "/silhouettes/"

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def make_kernal(n):
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
    return kernal

def calibration(img,background, gs_id=2):
    if gs_id == 1:
        return None, None
    elif gs_id == 2:
        M = np.load(SHAPES_ROOT + 'resources/GS2_M_color.npy')
        rows,cols,cha = img.shape
        imgw = cv2.warpPerspective(img, M, (cols, rows))
        imgwc = imgw[:,63:-80,:]
        bg_imgw = cv2.warpPerspective(background, M, (cols, rows))
        bg_imgwc = bg_imgw[:,63:-80,:]
        img_blur = cv2.GaussianBlur(bg_imgwc.astype(np.float32),(25,25),30)
        img_bs = imgwc.astype(np.int32) - img_blur.astype(np.int32) + np.mean(img_blur)
    return img_bs.astype(np.uint8), imgwc

def contact_detection(im,im_ref,low_bar,high_bar):
    im_sub = im/im_ref*70
    im_gray = im_sub.astype(np.uint8)
    im_canny = cv2.Canny(im_gray,low_bar,high_bar)

    kernal1 = make_kernal(10)
    kernal2 = make_kernal(10)
    kernal3 = make_kernal(30)
    kernal4 = make_kernal(20)
    img_d = cv2.dilate(im_canny, kernal1, iterations=1)
    img_e = cv2.erode(img_d, kernal1, iterations=1)
    img_ee = cv2.erode(img_e, kernal2, iterations=1)
    #    plt.figure()
    #    plt.imshow(img_ee)
    #    plt.show()
    img_dd = cv2.dilate(img_ee, kernal3, iterations=1)
    img_eee = cv2.erode(img_dd, kernal4, iterations=1)
    img_label = np.stack((np.zeros(img_dd.shape),np.zeros(img_dd.shape),img_eee),axis = 2).astype(np.uint8)
    return img_label

def creat_mask(im_gray):
    mask = (im_gray > 100).astype(np.uint8)
    kernal1 = make_kernal(10)
    kernal2 = make_kernal(30)
    kernal3 = make_kernal(100)
    kernal4 = make_kernal(80)
    img_d = cv2.dilate(mask, kernal1, iterations=1)
    img_e = cv2.erode(img_d, kernal2, iterations=1)
    img_dd = cv2.dilate(img_e, kernal3, iterations=1)
    img_ee = cv2.erode(img_dd, kernal4, iterations=1)
    return img_ee

def corner_mask(im):
    plt.imshow(im.astype(np.uint8))
    coor = plt.ginput(4)
    print("clicked", coor)
    x = np.array((coor[0][0],coor[1][0],coor[2][0],coor[3][0]),dtype = np.float32)
    y = np.array((coor[0][1],coor[1][1],coor[2][1],coor[3][1]),dtype = np.float32)
    m1 = (y[1]-y[0])/(x[1]-x[0])
    m2 = (y[3]-y[2])/(x[3]-x[2])
    b1 = y[0]-m1*x[0]
    b2 = y[2]-m2*x[2]
    col,row,cha = im.shape
    x_mesh, y_mesh = np.meshgrid(range(row),range(col))
    mask1 = (x_mesh*m1+b1)<y_mesh
    mask2 = (x_mesh*m2+b2)<y_mesh
    return mask1,mask2

def check_center(center,radius,col,row):
    if center[0] < radius + 5 or center[0] + radius +5 > col or center[1] < radius + 5 or center[1]+radius+5> row :
        return False
    else:
        return True


#%%

if __name__ == "__main__":

    ## Paths to obtain Gelsight raw images
    load_path = "/media/mcube/data/shapes_data/semicone_1/"
    # load_path = "/media/mcube/data/shapes_data/ball_D28.5/"
    # load_path = "/media/mcube/data/shapes_data/color3/"
    root, dirs, files = os.walk(load_path).next()

    ## Path to save new images and gradients
    # save_path = "data/semicone_1_processed_h_mm/"
    # save_path = "data/processed_color_D28.5_h_mm/"
    save_path = "data/test_semicone1_augmented/"
    #save_path = "data/test_sphere/"

    ## Select shape!
    # geometric_shape = 'sphere'
    geometric_shape = 'semicone_1'
    sphere_R_mm = 28.5/2  # Only used if geometric_shape == 'sphere'

    ## Basic parameters
    save_data = True
    show_data = False

    ## Augmented data params
    augmented_data_copies = 4  # Number of copies of augmented data that will be created and saved, 0 if you don't want it
    weight_mean = 1.
    weight_dev = 0.05
    biass_mean = 0.
    biass_dev = 9.

    ## Create labeler so that we can find circles on the images
    labeller = Labeller()

    ## Assumption: first image has no contact and can be used as background
    ref = cv2.imread(root+'/'+'GS2_1.png')
    ref_bs,ref_warp = calibration(ref,ref)
    mask_bd = np.load(SHAPES_ROOT + 'resources/GS2_mask_color.npy')
    kernal1 = make_kernal(30)
    kernal2 = make_kernal(10)
    col,row = ref_warp[:,:,1].shape
    x_mesh, y_mesh = np.meshgrid(range(row),range(col))


    ## Go through each raw gelsight image
    index = 0
    n =len(files)
    for i in range(n):
        if 'GS' in files[i]:
            print 'Progress made: ' + str(100.*float(i)/float(n)) + ' %'

            ## Apply calibration and mask to the raw image
            im_temp = cv2.imread(root+'/'+files[i])
            im_bs,im_wp = calibration(im_temp,ref)
            im_wp = im_wp*mask_bd
            im_wp_save = im_wp.copy()

            ## Remove background to the iamge and place its minimum to zero
            im_diff = im_wp.astype(np.int32) - ref_warp.astype(np.int32)
            im_diff_show = ((im_diff - np.min(im_diff))).astype(np.uint8)
            im_diff = im_diff - np.min(im_diff)

            ## Take into account different channels and create masks for the contact patch
            mask1 = (im_diff[:,:,0]-im_diff[:,:,1])>15
            mask2 = (im_diff[:,:,1]-im_diff[:,:,0])>12
                # cv2.imshow('mask1', mask1.astype(np.uint8)*255)
                # cv2.imshow('mask2', mask2.astype(np.uint8)*255)
            mask = ((mask1 + mask2)*255).astype(np.uint8)
            mask = mask*mask_bd[:,:,0]
                # cv2.imshow('maskA', mask)
                # cv2.waitKey(0)

            ## Apply eroding to the image and dilatation
            mask = cv2.erode(mask, kernal2, iterations=1)
            mask = cv2.dilate(mask, kernal2, iterations=1)
            mask = cv2.dilate(mask, make_kernal(35), iterations=1)
            mask_color = cv2.erode(mask, kernal1, iterations=1)


            ## Detect circles if any exists
            if np.sum(mask_color)/255 > 225:  #Checks if the contact patch is big enough
                im2,contours,hierarchy = cv2.findContours(mask_color, 1, 2)
                (x,y),radius = cv2.minEnclosingCircle(contours[0])
                center = (int(x),int(y))
                if geometric_shape == 'sphere':
                    center = (int(x)-2,int(y))  #TODO: why are we doing this?
                    radius = int(radius*0.77)#TODO: this numbers seems a bit of a hack
                elif geometric_shape == 'semicone_1':
                    radius = int(radius*0.85)
                else:
                    radius = int(radius)   #TODO: this numbers seems a bit of a hack

                ## Checks if the circle found matches well with contact patch
                mask_circle = ((x_mesh-center[0])**2 + (y_mesh-center[1])**2) < (radius)**2
                contact = contact_detection(rgb2gray(im_wp).astype(np.float32),rgb2gray(ref_warp).astype(np.float32),20,50)
                contact_mask = contact[:,:,2]*mask_circle
                if np.sum(contact_mask)/255 < 50 and len(contours)> 1:
                    (x,y),radius = cv2.minEnclosingCircle(contours[1])
                    center = (int(x)-2,int(y))
                    radius = int(radius*0.71)
                    mask_circle = ((x_mesh-center[0])**2 + (y_mesh-center[1])**2) < (radius)**2
                    contact = contact_detection(rgb2gray(im_wp).astype(np.float32),rgb2gray(ref_warp).astype(np.float32),20,50)
                    contact_mask = contact[:,:,2]*mask_circle

                center_ok = center[0] > 100 or center[1] > 100
                if center_ok and radius > 30 and radius < 80 and check_center(center,radius,col,row) and np.sum(contact_mask)/255 > 50:
                    contact = contact_detection(rgb2gray(im_wp).astype(np.float32),rgb2gray(ref_warp).astype(np.float32),20,40)
                    contact_mask = contact[:,:,2]*mask_circle
                    cv2.circle(im_wp,center,radius,(0,0,255),1)
                    # print 'get gradient params: ', center, radius
                    ## Compute gradients given center and radius
                    grad_x, grad_y = labeller.get_gradient_matrices(center,radius, shape=geometric_shape, sphere_R_mm=sphere_R_mm)
                    if (grad_x is not None) and (grad_y is not None):

                        ## Uncomment this to check gradients and heightmap

                        #print "Max: " + str(np.amax(grad_x))
                        #print "Min: " + str(np.amin(grad_x))

                        # cv2.imshow('gx', grad_x)
                        # cv2.imshow('gy', grad_y)
                        # cv2.waitKey(0)
                        depth_map = poisson_reconstruct(grad_y, grad_x)
                        print "Max: " + str(np.amax(depth_map))

                        def plot(depth_map):
                            fig = plt.figure()
                            ax = fig.gca(projection='3d')
                            X = np.arange(depth_map.shape[0], step=1)
                            Y = np.arange(depth_map.shape[1], step=1)
                            X, Y = np.meshgrid(X, Y)
                            surf = ax.plot_surface(X, Y, np.transpose(depth_map), rstride=1, cstride=1, cmap=cm.BuPu, linewidth=0, antialiased=False)
                            ax.set_zlim(0, 2)
                            ax.view_init(elev=90., azim=0)
                            #ax.axes().set_aspect('equal')
                            # plt.savefig(path + "img_" + str(img_number) + "_semicone_obj_weights.png")
                            plt.show()
                        depth_map = cv2.resize(depth_map, dsize=(100, 166), interpolation=cv2.INTER_LINEAR)
                        #plot(depth_map)
                        # '''

                        # We show/save the augmented data copies
                        for i in range(augmented_data_copies+1):
                            if i == 0:
                                noise_coefs = [(1, 0), (1, 0), (1, 0)]
                            else:
                                noise_coefs = get_rgb_noise(weight_mean, weight_dev, biass_mean, biass_dev)


                            if show_data:
                                cv2.imshow('contact_mask', contact_mask.astype(np.uint8))
                                cv2.imshow('image', cv2.cvtColor(introduce_noise(im_wp, noise_coefs)*mask_bd, cv2.COLOR_BGR2RGB))
                                cv2.imshow('grad_x',grad_x)
                                cv2.imshow('grad_y',grad_y)
                                cv2.waitKey(100)
                            if save_data:
                                index = index + 1
                                if not os.path.exists(save_path + 'image/'):
                                    os.makedirs(save_path + 'image/')
                                if not os.path.exists(save_path + 'image_circled/'):
                                    os.makedirs(save_path + 'image_circled/')
                                if not os.path.exists(save_path + 'gradient/'):
                                    os.makedirs(save_path + 'gradient/')
                                if not os.path.exists(save_path + 'heightmap/'):
                                    os.makedirs(save_path + 'heightmap/')

                                ## Save the depth map with the maximum height in the name
                                depth_map = poisson_reconstruct(grad_y, grad_x)
                                name = str(np.amax(depth_map))
                                cv2.imwrite(save_path + 'heightmap/'+str(index) + '_' + name + '.png', depth_map*1000)

                                ## Save raw image, raw_image with circle and the gradients
                                cv2.imwrite(save_path + 'image/img_'+str(index)+ '.png',cv2.cvtColor(introduce_noise(im_wp_save, noise_coefs)*mask_bd, cv2.COLOR_BGR2RGB))
                                cv2.imwrite(save_path + 'image_circled/img_'+str(index)+ '.png',cv2.cvtColor(introduce_noise(im_wp, noise_coefs)*mask_bd, cv2.COLOR_BGR2RGB))
                                np.save(save_path + 'gradient/gx_'+ str(index) + '.npy', grad_x)
                                np.save(save_path + 'gradient/gy_'+ str(index) + '.npy', grad_y)

                                ## Save the raw image blended with the depth map
                                io = Image.open(save_path + 'image_circled/img_'+str(index)+ '.png').convert("RGB") # image_for_input
                                ii = Image.open(save_path + 'heightmap/'+str(index) + '_' + name + '.png').resize(io.size).convert("RGB") #depth map
                                result = Image.blend(io, ii, alpha=0.5)
                                result.save(save_path + 'heightmap/'+str(index) + '_blend.png')
