from image_process_v2 import *

from keras.models import Sequential
from keras.layers import *
from keras.models import model_from_json
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import keras.losses
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
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
import matplotlib.image as mpimg
import math
import scipy, scipy.fftpack
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
SHAPES_ROOT = os.getcwd().split("/shapes/")[0] + "/shapes/"
label_mult_factor = 1.0
def preprocess_image(img):
    img = img.astype('float32')
    img[:, :, 0] = img[:, :, 0] - 82.53
    img[:, :, 1] = img[:, :, 1] - 82.61
    img[:, :, 2] = img[:, :, 2] - 82.76
    img = img/255.
    return img

def preprocess_label(arr):
    def f(x):
        x = max(-20, x)
        x = min(20, x)
        return float(x)*label_mult_factor
    f = np.vectorize(f, otypes=[np.float])
    return f(arr)

def posprocess_label(arr):
    def f(x):
        return float(x)/label_mult_factor
    f = np.vectorize(f, otypes=[np.float])
    return f(arr)

def plot_depth_map(depth_map, show=True, save=False, path=''):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(depth_map.shape[0], step=1)
    Y = np.arange(depth_map.shape[1], step=1)
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, np.transpose(depth_map), rstride=1, cstride=1, cmap=cm.BuPu, linewidth=0, antialiased=False)
    #ax.set_zlim(0, 5)
    ax.view_init(elev=90., azim=0)
    # ax.axes().set_aspect('equal')
    if save:
        plt.savefig(path + "img_" + str(img_number) + "color_3_objs_1000_points_each_same_h.png")
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
    return result

def custom_loss(y_true, y_pred):
    return K.sum(K.square(y_true-y_pred))

def raw_gs_to_depth_map(test_image, ref = None, model_path = None, plot=False, save=False, path=''):
    start = time.time()

    if ref == None:
        ref = test_image

    mask_color = np.load(SHAPES_ROOT + 'resources/GS2_mask_color.npy')
    im_bs, im_wp = calibration(test_image,ref)
    test_image = im_wp*mask_color
    test_image = test_image[...,[2,1,0]]

    plt.imshow(test_image)
    plt.show()

    keras.losses.custom_loss = custom_loss
    model = load_model(model_path)

    dim = (480, 497)
    test_image = cv2.resize(test_image, dsize=(497,480), interpolation=cv2.INTER_LINEAR)
    xvalues = np.array(range(dim[0])).astype('float32')/float(dim[0]) - 0.5 # Normalized
    yvalues = np.array(range(dim[1])).astype('float32')/float(dim[1]) - 0.5 # Normalized
    pos = np.stack((np.meshgrid(yvalues, xvalues)), axis = 2)

    test_image = preprocess_image(test_image)
    test_image = np.concatenate((test_image, pos),axis = 2)
    test_image = np.expand_dims(test_image, axis=0)

    grad = model.predict(test_image)
    grad_x = grad[...,0]
    grad_y = grad[...,1]

    grad_x = posprocess_label(grad_x)
    grad_y = posprocess_label(grad_y)

    depth_map = poisson_reconstruct(np.squeeze(grad_y), np.squeeze(grad_x))
    print "Max: " + str(np.amax(depth_map))

    print "Time used:"
    print time.time() - start

    if plot or save:
        plot_depth_map(depth_map, show=plot, save=save, path=path)

    return depth_map

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
