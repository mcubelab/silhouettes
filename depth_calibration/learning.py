from keras import backend as K
from keras.models import Sequential
from keras.layers import *
from keras.models import model_from_json
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.model_selection import train_test_split
from Datagenerator import DataGenerator
from random import shuffle
import numpy as np
import os, sys, yaml
import cv2
import scipy
import math
import keras.losses
from depth_helper import custom_loss
keras.losses.custom_loss = custom_loss

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
SHAPES_ROOT = os.getcwd().split("/silhouettes/")[0] + "/silhouettes/"

def createModel(input_shape, simulator=False, output_type = 'grad'):
    print input_shape
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu', input_shape=input_shape)) #,kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu')) #,kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu')) #,kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu')) #,kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu')) #,kernel_initializer='he_normal'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(64, (3, 3), activation='relu')) #,kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation='relu')) #,kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation='relu')) #,kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation='relu')) #,kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation='relu')) #,kernel_initializer='he_normal'))

    model.add(Conv2D(64, (1, 1), activation='relu')) #,kernel_initializer='he_normal'))

    model.add(Dropout(0.5))
    if simulator:
        model.add(Conv2D(3, (1, 1), activation='tanh')) #,kernel_initializer='he_normal'))
    else:
        if output_type == 'height':
            model.add(Conv2D(1, (1, 1), activation='tanh')) #,kernel_initializer='he_normal'))
        else:
            model.add(Conv2D(2, (1, 1), activation='tanh')) #,kernel_initializer='he_normal'))

    # Compile model
    ad = optimizers.Adam(lr=0.0001)
    model.compile(loss = custom_loss, optimizer=ad)
    model.summary()
    return model

def get_data_paths(paths, gradient, val_fraction=0.2, max_data_points=99999):
    combined = []

    def get_pictures_from_path(path, max):
        inpu = []
        lab = []
        root_aux, dirs_aux, inputs_raw = os.walk(path).next()
        for inp in np.sort(inputs_raw):
            if '.png' in inp and 'img_' in inp:
                inpu.append(path + inp)
                lab.append(path.replace('image/', "gradient/g") + gradient + '_' + inp.replace('.png', '.npy').replace('img_', ''))
        comb = list(zip(inpu, lab))
        shuffle(comb)
        return comb[:max]

    n_folders = len(paths)
    for path in paths:
        combined += get_pictures_from_path(path, max_data_points/n_folders)

    # print combined
    shuffle(combined)
    inputs, labels = zip(*combined)
    m = int(len(inputs)*val_fraction)
    return inputs[m:], labels[m:], inputs[:m+1], labels[:m+1]

def train(pretrain = False):
    # Params:
    simulator = False
    output_type =  'grad' #'grad' #'height', 'angle'
    weights_filepath = "weights/weights_semipyramid_2_08-17-2018_gs_id=1_rot=all_out_type={}.hdf5".format(output_type)

    paths = [
    # "/media/mcube/data/shapes_data/processed/ball_D6.35/image/",
    # "/media/mcube/data/shapes_data/processed/ball_D28.5/image/",
    # "/media/mcube/data/shapes_data/processed/hollowcone/image/",
    #"/media/mcube/data/shapes_data/processed/semicone_augmented/image/",
    #"/media/mcube/data/shapes_data/processed/semipyramid_augmented/image/",
    #'/media/mcube/data/shapes_data/processed/sphere_08-17-2018_gs_id=1_rot=0/image/',
    #'/media/mcube/data/shapes_data/processed/semicone_2_08-17-2018_gs_id=1_rot=0/image/',
    '/media/mcube/data/shapes_data/processed/semipyramid_2_08-17-2018_gs_id=1_rot=0/image/',
    '/media/mcube/data/shapes_data/processed/semipyramid_2_08-17-2018_gs_id=1_rot=1/image/',
    '/media/mcube/data/shapes_data/processed/semipyramid_2_08-17-2018_gs_id=1_rot=2/image/',
    '/media/mcube/data/shapes_data/processed/semipyramid_2_08-17-2018_gs_id=1_rot=3/image/'
    ]
    gs_id = 1
    # Datasets
    inputs_train, labels_train, inputs_val, labels_val = get_data_paths(paths=paths, gradient='x', val_fraction=0.2, max_data_points=500)
    print "Train size: " + str(len(inputs_train))
    print "Validation size: " + str(len(inputs_val))

    # Input/output shape (change it in resources/params.yaml)
    params_dict = yaml.load(open(SHAPES_ROOT + 'resources/params.yaml'))
    input_shape = params_dict['input_shape_gs{}'.format(gs_id)]
    if simulator:
        input_shape[2] = 5  # HACK
    
    input_image_shape = params_dict['input_shape_gs{}'.format(gs_id)][0:2]
    output_shape = params_dict['output_shape_gs{}'.format(gs_id)][0:2]

    # Generators
    train_batch_size = 8
    val_batch_size = 8

    training_generator = DataGenerator(inputs_train, labels_train, batch_size=train_batch_size, dim_in=input_image_shape, dim_out=output_shape, simulator=simulator, output_type = output_type)
    validation_generator = DataGenerator(inputs_val, labels_val, batch_size=val_batch_size, dim_in=input_shape, dim_out=output_shape, simulator=simulator, output_type = output_type)

    # Load weights
    history_save_filename=weights_filepath.replace(".hdf5", "_hist")
    if pretrain:
        model = load_model(weights_filepath)
    else:
        model = createModel(input_shape=input_shape, simulator=simulator, output_type = output_type)

    # Checkpoint
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    train_steps = int(np.floor(len(inputs_train)/train_batch_size))
    print 'Steps training: ', train_steps
    validation_steps = int(np.floor(len(inputs_val)/val_batch_size))
    print 'Steps validation: ', validation_steps

    print validation_steps
    # Run learning
    history = model.fit_generator(generator=training_generator, steps_per_epoch=train_steps, epochs=100, validation_data=validation_generator, validation_steps=validation_steps, callbacks=callbacks_list, workers=8, use_multiprocessing=True)

    # Save histoy file
    def save_file(filename, var):
        import cPickle as pickle
        import h5py
        import deepdish as dd
        dd.io.save(filename+'.h5', var)
    save_file(history_save_filename, history.history)

    # Save model
    model.to_json()

if __name__ == "__main__":
    train(pretrain=False)
