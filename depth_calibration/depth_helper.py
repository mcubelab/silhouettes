from image_processing import *

try:
    from keras.models import Sequential
    from keras.layers import *
    from keras.models import model_from_json
    from keras import optimizers
    from keras.callbacks import ModelCheckpoint
    import keras.losses
    from keras.models import load_model
    from keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    from keras import backend as K
except Exception as e:
    print "Not importing keras"

import pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import cv2
import scipy
import scipy.io
import cPickle as pickle
import h5py
import deepdish as dd
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import matplotlib.image as mpimg
import math
import scipy, scipy.fftpack
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
SHAPES_ROOT = os.getcwd().split("/silhouettes/")[0] + "/silhouettes/"
label_mult_factor = 1.0

def preprocess_image(img):
    img = img.astype('float32')
    img[:, :, 0] = img[:, :, 0] - 82.53
    img[:, :, 1] = img[:, :, 1] - 82.61
    img[:, :, 2] = img[:, :, 2] - 82.76
    img = img/255.
    return img

def posprocess_image(img):
    img = img.astype('float32')
    img = img*255.
    img[:, :, 0] = img[:, :, 0] + 82.53
    img[:, :, 1] = img[:, :, 1] + 82.61
    img[:, :, 2] = img[:, :, 2] + 82.76
    return img

def preprocess_label(arr):
    def f(x):
        x = max(-20, x)
        x = min(20, x)
        return float(x)*label_mult_factor
    f = np.vectorize(f, otypes=[np.float])
    return f(arr)

def preprocess_grad_for_simulation(grad_x, grad_y, gs_id=2, include_depth_chanel=False):
    # We get the network input dims
    params_dict = yaml.load(open(SHAPES_ROOT + 'resources/params.yaml'))
    if gs_id == 1:
        dim = params_dict['input_image_hw_gs1'][0:2]
    else:
        dim = params_dict['input_shape_gs2'][0:2]

    # We get the pos matrices
    xvalues = np.array(range(dim[0])).astype('float32')/float(dim[0]) - 0.5  # Normalized
    yvalues = np.array(range(dim[1])).astype('float32')/float(dim[1]) - 0.5 # Normalized
    pos = np.stack((np.meshgrid(yvalues, xvalues)), axis = 2)

    # We compute input: gx, gy, posx, posy
    grad_x = cv2.resize(grad_x, dsize=(dim[1], dim[0]), interpolation=cv2.INTER_LINEAR)
    grad_y = cv2.resize(grad_y, dsize=(dim[1], dim[0]), interpolation=cv2.INTER_LINEAR)

    # cv2.imshow('arx', grad_x)
    # cv2.imshow('ary', grad_y)
    # cv2.waitKey(0)

    if include_depth_chanel:
        depth_map = poisson_reconstruct(grad_y, grad_x)
        depth_map = np.expand_dims(depth_map, axis=2)

    grad_x = np.expand_dims(grad_x, axis=2)
    grad_y = np.expand_dims(grad_y, axis=2)
    grad2 = np.concatenate((grad_x, grad_y), axis=2)

    if include_depth_chanel:
        grad2 = np.concatenate((grad2, depth_map), axis=2)

    return np.concatenate((grad2, pos), axis=2) # We add pos channels

def get_rgb_noise(weight_mean, weight_dev, biass_mean, biass_dev):
    noise_coefs = []
    for i in range(3):
        weight = np.random.normal(weight_mean, weight_dev, 1)
        biass = np.random.normal(biass_mean, biass_dev, 1)
        noise_coefs.append((weight, biass))
    return noise_coefs

def introduce_noise(img, noise_coefs, mask=None):
    # Applies: channel = weight*channel + biass
    # to each channel, where weight, biass are random normal values with the given mean, dev
    for i in range(3):
        weight, biass = noise_coefs[i]
        img[...,i] = weight*img[...,i] + biass

    if mask is not None:
        if (len(img[0, 0, :]) > 0):
            for i in range(len(img[0, 0, :])):
                img[:, :, i] = img[:, :, i]*mask
        else:
            img = img*mask
    return img

def posprocess_label(arr):
    def f(x):
        return float(x)/label_mult_factor
    f = np.vectorize(f, otypes=[np.float])
    return f(arr)

def plot_depth_map(depth_map, show=True, save=False, path='', img_number = '', top_view=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(depth_map.shape[0], step=1)
    Y = np.arange(depth_map.shape[1], step=1)
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, np.transpose(depth_map), rstride=1, cstride=1, cmap=cm.BuPu, linewidth=0, antialiased=False)
    ax.set_zlim(0, 5)
    ax.view_init(elev=45., azim=5)
    if top_view:
        ax.view_init(elev=90., azim=0)
    # ax.axes().set_aspect('equal')

    if save:
        plt.savefig(path + "img_" + str(img_number) + ".png")
    if show:
        plt.show()

def poisson_reconstruct(grady, gradx):
    # Thanks to Dr. Ramesh Raskar for providing the original matlab code from which this is derived
    # Dr. Raskar's version is available here: http://web.media.mit.edu/~raskar/photo/code.pdf

    # We set boundary conditioons
    # print gradx.shape
    boundarysrc = np.zeros(gradx.shape)

    # Laplacian
    gyy = grady[1:,:-1] - grady[:-1,:-1]
    gxx = gradx[:-1,1:] - gradx[:-1,:-1]
    f = np.zeros(boundarysrc.shape)
    f[:-1,1:] += gxx
    f[1:,:-1] += gyy

    # Boundary image
    boundary = boundarysrc.copy()
    boundary[1:-1,1:-1] = 0;

    # Subtract boundary contribution
    f_bp = -4*boundary[1:-1,1:-1] + boundary[1:-1,2:] + boundary[1:-1,0:-2] + boundary[2:,1:-1] + boundary[0:-2,1:-1]
    f = f[1:-1,1:-1] - f_bp

    # Discrete Sine Transform
    # print f.shape
    tt = scipy.fftpack.dst(f, norm='ortho')
    fsin = scipy.fftpack.dst(tt.T, norm='ortho').T

    # Eigenvalues
    (x,y) = np.meshgrid(range(1,f.shape[1]+1), range(1,f.shape[0]+1), copy=True)
    denom = (2*np.cos(math.pi*x/(f.shape[1]+2))-2) + (2*np.cos(math.pi*y/(f.shape[0]+2)) - 2)

    f = fsin/denom

    # Inverse Discrete Sine Transform
    tt = scipy.fftpack.idst(f, norm='ortho')
    img_tt = scipy.fftpack.idst(tt.T, norm='ortho').T

    # New center + old boundary
    result = boundary
    result[1:-1,1:-1] = img_tt
    positive_mask = (result > 0).astype(np.float32)
    result = result*positive_mask
    return result

def custom_loss(y_true, y_pred):
    return K.sum(K.square(y_true-y_pred))

def raw_gs_to_depth_map(gs_id=2, test_image=None, ref=None, model_path=None, plot=False, save=False, path='', img_number='',
                            output_type='grad', test_depth = None, model = None):
    if ref == None:
        ref = test_image


    im_bs, im_wp = calibration(test_image,ref)
#    mask_color = np.load(SHAPES_ROOT + 'resources/GS2_mask_color.npy')
    mask_color = np.load(SHAPES_ROOT + 'resources/mask_GS2.npy')
    mask_color = np.repeat(np.expand_dims(mask_color, axis=2), 3,axis=2)
    mask_color = cv2.resize(mask_color, dsize=(im_wp.shape[1], im_wp.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    test_image = im_wp*mask_color
    test_image = test_image[...,[2,1,0]]

    keras.losses.custom_loss = custom_loss
    if model is None:
        model = load_model(model_path)

    params_dict = yaml.load(open(SHAPES_ROOT + 'resources/params.yaml'))
    if gs_id == 1:
        dim = params_dict['input_image_hw_gs1'][0:2]
        pass
    else:
        dim = params_dict['input_shape_gs2'][0:2]

    test_image = cv2.resize(test_image, dsize=(dim[1], dim[0]), interpolation=cv2.INTER_LINEAR)
    xvalues = np.array(range(dim[0])).astype('float32')/float(dim[0]) - 0.5 # Normalized
    yvalues = np.array(range(dim[1])).astype('float32')/float(dim[1]) - 0.5 # Normalized
    pos = np.stack((np.meshgrid(yvalues, xvalues)), axis = 2)

    test_image = preprocess_image(test_image)
    test_image = np.concatenate((test_image, pos),axis = 2)
    test_image = np.expand_dims(test_image, axis=0)
    start = time.time()
    if output_type != 'height':
        grad = model.predict(test_image)
        grad_x = grad[...,0]
        grad_y = grad[...,1]
        
        grad_x = posprocess_label(grad_x)
        grad_y = posprocess_label(grad_y)
        
        depth_map = poisson_reconstruct(np.squeeze(grad_y), np.squeeze(grad_x))
        if output_type == 'angle':
            depth_map = poisson_reconstruct(np.squeeze(np.tan(grad_y)), np.squeeze(np.tan(grad_x)))
    else:
        depth_map = model.predict(test_image)[0,:,:,0]        

    print "Max: " + str(np.amax(depth_map))

    print "Time used:"
    print time.time() - start

    if plot or save:
        plot_depth_map(depth_map, show=plot, save=save, path=path, img_number=img_number)
    
    if test_depth is not None:
        loss = np.sum(np.square(depth_map - test_depth))  #TODO: assumption that this is the custom loss
        return depth_map, loss
    return depth_map

def grad_to_gs(model_path, gx, gy, gs_id=2):
    inp = copy.deepcopy(preprocess_grad_for_simulation(gx, gy, gs_id=2, include_depth_chanel=True))
    inp = np.expand_dims(inp, axis=0)

    keras.losses.custom_loss = custom_loss
    model = load_model(model_path)

    gs_sim = model.predict(inp)
    gs_sim = np.squeeze(gs_sim)

    gs_sim = posprocess_image(gs_sim)

    return gs_sim

def plot_model_history(filename):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    legend_list = []

    # summarize history for accuracy
    # filename = "weights/weights.color.xy.hdf5".replace("hdf5", "hist")
    history_list = [dd.io.load(filename+'.h5')]
    for i, model_history in enumerate(history_list):
        axs[0].plot(range(1,len(model_history['acc'])+1),model_history['acc'])
        # legend_list.append('Train:' + self.program_list[i].name)
        legend_list.append('Train:' + 'color xy')
        axs[0].plot(range(1,len(model_history['val_acc'])+1),model_history['val_acc'], '--')
        # legend_list.append('Val:' + self.program_list[i].name)
        legend_list.append('Val:' + 'color xy')

    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(history_list[0]['acc'])+1),len(history_list[0]['acc'])/10)
    axs[0].legend(legend_list, loc='best')
    # summarize history for loss
    for model_history in history_list:
        axs[1].plot(range(1,len(model_history['loss'])+1),model_history['loss'])
        axs[1].plot(range(1,len(model_history['val_loss'])+1),model_history['val_loss'], '--')

    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(history_list[0]['loss'])+1),len(history_list[0]['loss'])/10)
    axs[1].legend(legend_list, loc='best')
    plt.show()
