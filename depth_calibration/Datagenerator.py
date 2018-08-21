
# from grad_to_depth import *
from depth_helper import *

import numpy as np
import keras
import os, sys
import cv2
import scipy
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
SHAPES_ROOT = os.getcwd().split("/silhouettes/")[0] + "/silhouettes/"

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size, dim_in, dim_out, n_channels=5, shuffle=True, simulator=False, output_type='grad'):
        'Initialization'
        self.dim = dim_in
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs

        self.n_channels = n_channels
        self.out_channels = 2
        if simulator:
            self.n_channels = 5  # HACK
            self.out_channels = 3  # HACK
        if output_type == 'height':
            self.out_channels = 1  # HACK

        self.shuffle = shuffle
        self.on_epoch_end()
        self.xvalues = np.array(range(self.dim[0])).astype('float32')/float(self.dim[0]) -0.5  # Normalized
        self.yvalues = np.array(range(self.dim[1])).astype('float32')/float(self.dim[1])  -0.5 # Normalized
        self.pos = np.stack((np.meshgrid(self.yvalues, self.xvalues)), axis = 2)
        self.dim_out = dim_out

        self.simulator = simulator
        self.output_type = output_type

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        labels_temp = [self.labels[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp,labels_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self,list_IDs_temp,labels_temp):
        X = np.empty((self.batch_size, self.dim[0],self.dim[1], self.n_channels))
        y = np.empty((self.batch_size, self.dim_out[0],self.dim_out[1], self.out_channels))
        lenofname = len(list_IDs_temp[0])
        # print list_IDs_temp[0]
        # print len(list_IDs_temp)
        # for i, ID in enumerate(list_IDs_temp):

        for i in range(len(list_IDs_temp)):
            im_temp = cv2.imread(list_IDs_temp[i])
            if im_temp is None:
                print list_IDs_temp[i]
            im_temp = preprocess_image(im_temp)

            grad_x = np.load(labels_temp[i])
            grad_y = np.load(labels_temp[i].replace("gx_","gy_"))
            if self.output_type == 'angle': #In order to consider angle
                grad_x = np.load(labels_temp[i].replace("gy_","gx_angle_"))
                grad_y = np.load(labels_temp[i].replace("gx_","gy_"))

            if self.simulator:
                # We compute input: gx, gy, posx, posy
                img = copy.deepcopy(preprocess_grad_for_simulation(grad_x, grad_y, gs_id=2, include_depth_chanel=True))
                im_temp = cv2.resize(im_temp, dsize=(self.dim_out[1], self.dim_out[0]), interpolation=cv2.INTER_LINEAR)
                grad = copy.deepcopy(im_temp)
            else:
                # grad = skimage.measure.block_reduce(grad, (2,2), np.max)
                grad_x = cv2.resize(grad_x, dsize=(self.dim_out[1], self.dim_out[0]), interpolation=cv2.INTER_LINEAR)
                grad_y = cv2.resize(grad_y, dsize=(self.dim_out[1], self.dim_out[0]), interpolation=cv2.INTER_LINEAR)

                grad_x = preprocess_label(grad_x)
                grad_y = preprocess_label(grad_y)

                if self.output_type == 'height':
                    grad = poisson_reconstruct(grad_y, grad_x)
                    grad = np.expand_dims(grad, axis=2)
                else:
                    grad_x = np.expand_dims(grad_x, axis=2)
                    grad_y = np.expand_dims(grad_y, axis=2)
                    grad = np.concatenate((grad_x, grad_y), axis = 2)

                img = np.concatenate((im_temp, self.pos),axis = 2)
            X[i,] = np.array(img)
            y[i,] = np.array(grad)
        return X, y
