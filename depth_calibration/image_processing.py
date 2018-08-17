import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from labeller import *
from depth_helper import *

import pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os, sys
import cv2
import scipy
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
_EPS = 1e-5

sphere_R_mm = 0
hollow_r_mm = 0
hollow_R_mm = 0
semicone_r_mm = 0
hollowcone_slope = 0
semicone_slope = 0

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
    im_sub = im/(im_ref+_EPS)*70
    im_gray = im_sub.astype(np.uint8)
    im_canny = cv2.Canny(im_gray, low_bar, high_bar)

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
    # cv2.imshow('img_d', img_d)
    # cv2.imshow('img_e', img_e)
    # cv2.imshow('img_ee', img_ee)
    # cv2.imshow('img_ee', img_dd)
    # cv2.imshow('img_ee', img_eee)
    # cv2.imshow('img_label', img_label)
    # cv2.waitKey(0)
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

def get_files(load_path, only_pictures=True):
    if only_pictures:
        root, dirs, files = os.walk(load_path).next()
    else:
        files = []
        root, ds, fs = os.walk(load_path).next()
        for d in ds:
            root2, ds2, fs2 = os.walk(load_path + "/" + d).next()
            for f in fs2:
                if "GS" in f:
                    files.append("/" + d + "/" + f)
    return root, files

if __name__ == "__main__":
    # NOTE: remember we assume GSx_0 is air picture

        ## Select GS Id!!!!!!!!
    gs_id = 2

        ## Select shape!!!!!!!
    #shape = 'sphere'
    #shape = 'semicone_1'
    #shape = 'semicone_2'
    #shape = 'hollowcone_1'
    shape = 'hollowcone_2'
    # shape = 'test1'
    # shape = 'semipyramid'
    '''
    gs_ids = [1,2]
    shapes = ['sphere', 'semicone_1', 'semicone_2', 'hollowcone_1', 'hollowcone_2', 'semipyramid', 'stamp']
    for gs_id in gs_ids:
        for shape in shapes:
    '''
    print shape
    # IMPORTANT NOTE: MAKE SURE YOU UNCOMMENT THE MESURES OF THE ONE YOU ARE PROCESSING!!!!!!!!!!!!!
    # Sphere
    sphere_R_mm = 28.5/2  # Only used if geometric_shape == 'sphere'

    
    if shape == 'hollowcone 1':
        hollow_r_mm = 9.8/2
        hollow_R_mm = 16.5/2
        hollowcone_slope = 10
    else:    
        # # hollowcone 2
        hollow_r_mm = 9.8/2
        hollow_R_mm = 16.5/2
        hollowcone_slope = 20

    if shape == 'semicone 1':
        semicone_r_mm = 10.2/2
        semicone_slope = 30
    else:
        # # semicone 2
        semicone_r_mm = 16.5/2
        semicone_slope = 10



    # Don't touch this
    geometric_shape = shape
    shape_params = (sphere_R_mm, hollow_r_mm, hollow_R_mm, semicone_r_mm, hollowcone_slope, semicone_slope)
    
    ## Paths to obtain Gelsight raw images
    folder_data_name = shape + '_08-15-2018_gs2_rot=0/'
    load_path = "/media/mcube/data/shapes_data/raw/" + folder_data_name
    # load_path = "sample_data/"
    only_pictures = False  # If in the load_path there are only the pictures and not folders with each cartesian info
    root, files = get_files(load_path, only_pictures=only_pictures)

    ## Path to save new images and gradients
    save_path = "/media/mcube/data/shapes_data/processed/"+ folder_data_name
    # save_path = "sample_data/"

    ## Basic parameters
    save_data = True
    show_data = False

    ## Augmented data params
    augmented_data_copies = 4  # Number of copies of augmented data that will be created and saved, 0 if you don't want it
    weight_mean = 1.
    weight_dev = 0.05
    biass_mean = 0.
    biass_dev = 9.

    ## Create labeler so that we can get the ground truth of the gradients and height of the pictures
    labeller = Labeller()

    ## NOTE: ASSUMPTION: first image has no contact and can be used as background, if not only image we assume that there exists air folder
    ## TODO: THIS SPECIAL CASE FOR THE HOLLOWCONE HAS TO BE REMOVED
    if only_pictures:
        ref = cv2.imread(load_path + 'GS' + str(gs_id) + '_0.png')
    else:
        ref = cv2.imread(load_path + '/air/GS' + str(gs_id) + '.png')
    # if geometric_shape == 'hollowcone':
    #     ref = cv2.imread(load_path+'GS' + str(gs_id) + '_0.png')
    # else:
    #     ref = cv2.imread(root+'/'+'GS' + str(gs_id) + '_1.png')

    # Load a different mask for each gelsight
    if gs_id == 1:
        mask_bd = np.load(SHAPES_ROOT + 'resources/mask_GS1.npy')
    elif gs_id == 2:
        mask_bd = np.load(SHAPES_ROOT + 'resources/mask_GS2.npy')

    ref_bs, ref_warp = calibration(ref, ref, gs_id, mask_bd)

    ## Create x_mesh and y_mesh
    col, row = ref_warp[:, :, 1].shape
    x_mesh, y_mesh = np.meshgrid(range(row), range(col))

    ## Max_rad param:
    if 'hollowcone' in geometric_shape:
        max_rad = 200
    elif geometric_shape == 'semicone_1':
        max_rad = 80
    else:
        max_rad = 250

    ## Go through each raw gelsight image
    index = 0
    n = len(files)
    for i in range(n):
	print files[i]
        if (((gs_id == 1) and ('GS1' in files[i])) or ((gs_id == 2) and ('GS2' in files[i]))):
            print 'Progress made: ' + str(100.*float(i)/float(n)) + ' %'

            ## Apply calibration and mask to the raw image
            im_temp = cv2.imread(root+'/'+files[i])
	    #print root+'/'+files[i]

            im_bs, im_wp = calibration(im_temp, ref, gs_id, mask_bd)
            # print "Shape im_bs: " + str(im_bs.shape)
            # print "Shape im_wp: " + str(im_wp.shape)

            # cv2.imshow("im_temp", im_temp)
            # cv2.imshow("im_bs", im_bs)
            # cv2.imshow("im_wp", im_wp)
            # cv2.waitKey(0)
            # print "im_wp: ", im_wp.shape
            im_wp_save = im_wp.copy()

            ## Remove background to the iamge and place its minimum to zero
            im_diff = im_wp.astype(np.int32) - ref_warp.astype(np.int32)
            # cv2.imshow("im_wp", im_wp)
            # cv2.imshow("ref_warp", ref_warp)
            # cv2.waitKey(0)

            im_diff_show = ((im_diff - np.min(im_diff))).astype(np.uint8)
            # cv2.imshow("im_diff_show", im_diff_show)
            # cv2.waitKey(0)
            # print "MAX dif: " + str(np.amax(im_diff))
            # print "MAX dif: " + str(np.amin(im_diff))
            im_diff = im_diff - np.min(im_diff)

            ## Take into account different channels and create masks for the contact patch
            mask1 = (im_diff[:, :, 0]-im_diff[:, :, 1]) > 15
            mask2 = (im_diff[:, :, 1]-im_diff[:, :, 0]) > 12
            # cv2.imshow('mask1', ((mask1)*255).astype(np.uint8))
            # cv2.imshow('mask2', ((mask2)*255).astype(np.uint8))
            mask = ((mask1 + mask2)*255).astype(np.uint8)
            # cv2.imshow("mask", mask)
            # cv2.waitKey(0)

            mask = rgb2gray(contact_detection(rgb2gray(im_wp).astype(np.float32), rgb2gray(ref_warp).astype(np.float32),20,50))
            # mask = mask*mask_bd[:,:,0]
            # mask = mask*mask_bd

            # cv2.imshow('maskA', mask)
            # cv2.waitKey(0)

            ## Apply eroding to the image and dilatation
            kernal1 = make_kernal(30)
            kernal2 = make_kernal(10)
            mask = cv2.erode(mask, kernal2, iterations=1)
            mask = cv2.dilate(mask, kernal2, iterations=1)
            mask = cv2.dilate(mask, make_kernal(35), iterations=1)
            mask_color = cv2.erode(mask, kernal1, iterations=1).astype(np.uint8)
            # cv2.imshow('mask_color', mask_color)
            # cv2.waitKey(0)

            ## Detect circles if any exists
            if np.sum(mask_color)/255 > 225:  #Checks if the contact patch is big enough
                im2, contours, hierarchy = cv2.findContours(mask_color, 1, 2)
                
                #import pdb; pdb.set_trace()                
                (x, y), radius = (0, 0), 0
                biggest_area = 0
                biggest_rect = None
                for c in contours:
                    if geometric_shape == 'semipyramid':
                        rect = cv2.minAreaRect(c)
                        area = float(rect[1][0])*float(rect[1][1])
                        if biggest_area < area:
                            biggest_rect = rect
                            biggest_area = area
                    else:
                        (x_, y_), radius_ = cv2.minEnclosingCircle(c)
                        if radius_ > radius:
                            (x, y), radius = (x_, y_), radius_

                # print (x, y), radius
                if geometric_shape != 'semipyramid':
                    center = (int(x), int(y))
                    if geometric_shape == 'sphere':
                        # center = (int(x)-2, int(y))  # TODO: why are we doing this?
                        radius = int(radius*0.8)  # TODO: this numbers seems a bit of a hack
                    elif 'semicone' in geometric_shape:
                        radius = int(radius*0.9)
                    else:
                        radius = int(radius)  # TODO: this numbers seems a bit of a hack

                    #raw_input("Press Enter to continue...")

                    ## Checks if the circle found matches well with contact patch
                    mask_circle = ((x_mesh-center[0])**2 + (y_mesh-center[1])**2) < (radius)**2

                    contact = contact_detection(rgb2gray(im_wp).astype(np.float32), rgb2gray(ref_warp).astype(np.float32),20,50)
                    contact_mask = contact[:, :, 2]*mask_circle
                    # cv2.imshow('contact_mask', contact_mask)
                    # cv2.waitKey(0)
                    if np.sum(contact_mask)/255 < 50 and len(contours) > 1:
                        (x, y), radius = cv2.minEnclosingCircle(contours[1])
                        center = (int(x)-2, int(y))
                        radius = int(radius*0.71)
                        mask_circle = ((x_mesh-center[0])**2 + (y_mesh-center[1])**2) < (radius)**2
                        contact = contact_detection(rgb2gray(im_wp).astype(np.float32), rgb2gray(ref_warp).astype(np.float32), 20, 50)
                        contact_mask = contact[:, :, 2]*mask_circle

                    center_ok = center[0] > 100 or center[1] > 100
                    if center_ok and radius > 30 and radius < max_rad and check_center(center, radius, col, row) and np.sum(contact_mask)/255 > 50:
                        contact = contact_detection(rgb2gray(im_wp).astype(np.float32), rgb2gray(ref_warp).astype(np.float32), 20, 40)
                        contact_mask = contact[:, :, 2]*mask_circle
                        cv2.circle(im_wp, center, radius, (0, 0, 255), 1)
                        # print 'get gradient params: ', center, radius
                        ## Compute gradients given center and radius
                        # cv2.imshow('im', im_wp)
                        # cv2.waitKey(0)
                        # print "im_wp_save shape: ", im_wp_save.shape
                        grad_x, grad_y = labeller.get_gradient_matrices(center, radius, shape=geometric_shape, shape_params=shape_params)
                        if (grad_x is not None) and (grad_y is not None):

                            ## Uncomment this to check gradients and heightmap

                            #print "Max: " + str(np.amax(grad_x))
                            #print "Min: " + str(np.amin(grad_x))

                            # cv2.imshow('gx', grad_x)
                            # cv2.imshow('gy', grad_y)
                            # cv2.waitKey(0)
                            depth_map = poisson_reconstruct(grad_y, grad_x)

                            # print "Max: " + str(np.amax(depth_map))

                            def plot(depth_map):
                                fig = plt.figure()
                                ax = fig.gca(projection='3d')
                                X = np.arange(depth_map.shape[0], step=1)
                                Y = np.arange(depth_map.shape[1], step=1)
                                X, Y = np.meshgrid(X, Y)
                                surf = ax.plot_surface(X, Y, np.transpose(depth_map), rstride=1, cstride=1, cmap=cm.BuPu, linewidth=0, antialiased=False)
                                ax.set_zlim(0, 2)
                                ax.view_init(elev=90., azim=0)
                                # ax.axes().set_aspect('equal')
                                # plt.savefig(path + "img_" + str(img_number) + "_semicone_obj_weights.png")
                                plt.show()
                            depth_map = cv2.resize(depth_map, dsize=(100, 166), interpolation=cv2.INTER_LINEAR)
                            # plot(depth_map)
                            # '''

                            # We show/save the augmented data copies
                            for iii in range(augmented_data_copies+1):
                                if iii == 0:
                                    noise_coefs = [(1, 0), (1, 0), (1, 0)]
                                else:
                                    noise_coefs = get_rgb_noise(weight_mean, weight_dev, biass_mean, biass_dev)
                                    # print '#####Noise coefs:'
                                    # print noise_coefs
                                    # print '#####'

                                noise_mask = mask_bd[10:, 65:572]

                                if show_data:
                                    cv2.imshow('contact_mask', contact_mask.astype(np.uint8))
                                    cv2.imshow('image', cv2.cvtColor(introduce_noise(im_wp, noise_coefs, mask=noise_mask), cv2.COLOR_BGR2RGB))
                                    cv2.imshow('grad_x', grad_x)
                                    cv2.imshow('grad_y', grad_y)
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
                                    cv2.imwrite(save_path + 'image/img_'+str(index)+ '.png',cv2.cvtColor(introduce_noise(im_wp_save, noise_coefs, mask=noise_mask), cv2.COLOR_BGR2RGB))
                                    cv2.imwrite(save_path + 'image_circled/img_'+str(index)+ '.png',cv2.cvtColor(introduce_noise(im_wp, noise_coefs, mask=noise_mask), cv2.COLOR_BGR2RGB))
                                    np.save(save_path + 'gradient/gx_'+ str(index) + '.npy', grad_x)
                                    np.save(save_path + 'gradient/gy_'+ str(index) + '.npy', grad_y)
                                    np.save(save_path + 'gradient/gx_angle_'+ str(index) + '.npy', np.arctan(grad_x))
                                    np.save(save_path + 'gradient/gy_angle_'+ str(index) + '.npy', np.arctan(grad_y))

                                    ## Save the raw image blended with the depth map
                                    io = Image.open(save_path + 'image_circled/img_'+str(index)+ '.png').convert("RGB") # image_for_input
                                    ii = Image.open(save_path + 'heightmap/'+str(index) + '_' + name + '.png').resize(io.size).convert("RGB") #depth map
                                    result = Image.blend(io, ii, alpha=0.5)
                                    result.save(save_path + 'heightmap/'+str(index) + '_blend.png')
                    else:
                        print "[ERROR]: Circle out of defined limits"

                elif geometric_shape == "semipyramid":
                    box = cv2.boxPoints(biggest_rect)
                    box = np.int0(box)
                    mask_color = mask_color*255

                    center_px = biggest_rect[0]
                    sides_px = biggest_rect[1]
                    angle = biggest_rect[2]
                    # im_wp = copy.deepcopy(im_wp_save)
                    cv2.drawContours(im_wp, [box], 0, (100, 100, 100), 2)

                    # cv2.drawContours(mask_color, [box], 0, (100, 100, 100), 2)
                    # cv2.imshow('mask_color', mask_color)
                    # cv2.waitKey(0)

                    grad_x, grad_y = labeller.get_gradient_matrices(center_px=center_px, angle=angle, sides_px=sides_px, shape=geometric_shape, shape_params=shape_params)

                    if (grad_x is not None) and (grad_y is not None):
                        # print "NOT none"
                        ## Uncomment this to check gradients and heightmap

                        #print "Max: " + str(np.amax(grad_x))
                        #print "Min: " + str(np.amin(grad_x))

                        # cv2.imshow('gx', grad_x)
                        # cv2.imshow('gy', grad_y)
                        # cv2.waitKey(0)
                        # depth_map = poisson_reconstruct(grad_y, grad_x)
                        # print "Max: " + str(np.amax(depth_map))

                        # cv2.imshow('depth_map', depth_map)
                        # cv2.waitKey(0)

                        # def plot(depth_map):
                        #     fig = plt.figure()
                        #     ax = fig.gca(projection='3d')
                        #     X = np.arange(depth_map.shape[0], step=1)
                        #     Y = np.arange(depth_map.shape[1], step=1)
                        #     X, Y = np.meshgrid(X, Y)
                        #     surf = ax.plot_surface(X, Y, np.transpose(depth_map), rstride=1, cstride=1, cmap=cm.BuPu, linewidth=0, antialiased=False)
                        #     ax.set_zlim(0, 2)
                        #     ax.view_init(elev=90., azim=0)
                        #     # ax.axes().set_aspect('equal')
                        #     # plt.savefig(path + "img_" + str(img_number) + "_semicone_obj_weights.png")
                        #     plt.show()
                        # depth_map = cv2.resize(depth_map, dsize=(100, 166), interpolation=cv2.INTER_LINEAR)
                        # plot(depth_map)
                        # '''

                        # We show/save the augmented data copies
                        for iii in range(augmented_data_copies+1):
                            if iii == 0:
                                noise_coefs = [(1, 0), (1, 0), (1, 0)]
                            else:
                                noise_coefs = get_rgb_noise(weight_mean, weight_dev, biass_mean, biass_dev)
                                # print '#####Noise coefs:'
                                # print noise_coefs
                                # print '#####'

                            noise_mask = mask_bd[10:, 65:572]
                            if show_data:
                                # cv2.imshow('contact_mask', contact_mask.astype(np.uint8))
                                cv2.imshow('image', cv2.cvtColor(introduce_noise(im_wp, noise_coefs, mask=noise_mask), cv2.COLOR_BGR2RGB))
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
                                # print "depth_map: ", depth_map.shape
                                # print "im_wp: ", im_wp.shape
                                # a = raw_input('aa')
                                name = str(np.amax(depth_map))
                                cv2.imwrite(save_path + 'heightmap/'+str(index) + '_' + name + '.png', depth_map*1000)

                                ## Save raw image, raw_image with circle and the gradients
                                cv2.imwrite(save_path + 'image/img_'+str(index)+ '.png',cv2.cvtColor(introduce_noise(im_wp_save, noise_coefs, mask=noise_mask), cv2.COLOR_BGR2RGB))
                                cv2.imwrite(save_path + 'image_circled/img_'+str(index)+ '.png',cv2.cvtColor(introduce_noise(im_wp, noise_coefs, mask=noise_mask), cv2.COLOR_BGR2RGB))
                                np.save(save_path + 'gradient/gx_'+ str(index) + '.npy', grad_x)
                                np.save(save_path + 'gradient/gy_'+ str(index) + '.npy', grad_y)

                                ## Save the raw image blended with the depth map
                                io = Image.open(save_path + 'image_circled/img_'+str(index)+ '.png').convert("RGB") # image_for_input
                                ii = Image.open(save_path + 'heightmap/'+str(index) + '_' + name + '.png').resize(io.size).convert("RGB") #depth map
                                result = Image.blend(io, ii, alpha=0.5)
                                result.save(save_path + 'heightmap/'+str(index) + '_blend.png')
