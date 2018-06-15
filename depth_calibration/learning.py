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
import os, sys
import cv2
import scipy
import math
import keras.losses
from depth_helper import custom_loss
keras.losses.custom_loss = custom_loss

def createModel(input_shape):
    print input_shape
    model = Sequential()
    #model.add(MaxPooling2D(pool_size=(3, 3), strides=3, input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu', input_shape=input_shape)) #,kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu')) #,kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu')) #,kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu')) #,kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu')) #,kernel_initializer='he_normal'))

    # model.add(Conv2D(64, (3, 3), padding='same',activation='relu',kernel_initializer='he_normal'))
    # model.add(Conv2D(64, (3, 3), padding='same',activation='relu',kernel_initializer='he_normal'))
    # model.add(Conv2D(64, (3, 3), padding='same',activation='relu',kernel_initializer='he_normal'))
    # model.add(Conv2D(64, (3, 3), padding='same',activation='relu',kernel_initializer='he_normal'))
    # model.add(Conv2D(64, (3, 3), padding='same',activation='relu',kernel_initializer='he_normal'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(64, (3, 3), activation='relu')) #,kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation='relu')) #,kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation='relu')) #,kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation='relu')) #,kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation='relu')) #,kernel_initializer='he_normal'))

    model.add(Conv2D(64, (1, 1), activation='relu')) #,kernel_initializer='he_normal'))
    # model.add(Conv2D(64, (1, 1), activation='relu',kernel_initializer='he_normal'))
    # model.add(Conv2D(64, (1, 1), activation='relu',kernel_initializer='he_normal'))

    model.add(Dropout(0.5))
    model.add(Conv2D(2, (1, 1), activation='tanh')) #,kernel_initializer='he_normal'))

    # Compile model
    ad = optimizers.Adam(lr=0.0001)
    model.compile(loss = custom_loss, optimizer=ad)
    model.summary()
    return model

def get_data_paths(paths, gradient, val_fraction=0.2, max_data_points=99999):
    inputs = []
    labels = []
    it = 0

    for path in paths:
        root_aux, dirs_aux, inputs_raw = os.walk(path).next()

        for inp in np.sort(inputs_raw):
            if '.png' in inp and 'img_' in inp:
                a = [int(s) for s in inp.replace('_', ' ').replace('.', ' ').split() if s.isdigit()]
                if a[0] > 2000:
                    continue
                    # pass
                inputs.append(path + inp)
                labels.append(path.replace('image/', "gradient/g") + gradient + '_' + inp.replace('.png', '.npy').replace('img_', ''))

                it += 1
                if it >= max_data_points:
                    break
        if it >= max_data_points:
            break

    combined = list(zip(inputs, labels))
    shuffle(combined)

    inputs[:], labels[:] = zip(*combined)
    m = int(len(inputs)*val_fraction)
    return inputs[m:], labels[m:], inputs[:m+1], labels[:m+1]


def train(pretrain = False):
    # Params:
    weights_filepath = "weights/weights.color_semicone1_and_sphere_v2.xy.hdf5"

    paths = ["data/test_semicone1/image/", "data/test_sphere/image/" ]
    # paths = ["data/color3_processed_h_mm/", "data/processed_color_D28.5_h_mm/image/", "data/semicone_1_processed_h_mm/"]
    # paths = ["data/processed_color_D28.5/image/"]
    # paths = ["data/semicone_1_processed_h_mm/image/"]
    input_shape = (480, 497, 5)

    # Datasets
    inputs_train, labels_train, inputs_val, labels_val = get_data_paths(paths=paths, gradient='x', val_fraction=0.2, max_data_points=9999999)
    print "Train size: " + str(len(inputs_train))
    print "Validation size: " + str(len(inputs_val))

    # Generators
    train_batch_size = 8
    val_batch_size = 8

    training_generator = DataGenerator(inputs_train, labels_train, batch_size = train_batch_size)
    validation_generator = DataGenerator(inputs_val, labels_val, batch_size = val_batch_size)

    # Load weights
    history_save_filename=weights_filepath.replace(".hdf5", "_hist")
    if pretrain:
        model = load_model(weights_filepath)
    else:
        model = createModel(input_shape=input_shape)

    # Checkpoint
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    train_steps = int(np.floor(len(inputs_train)/train_batch_size))
    print 'Steps training: ', train_steps
    validation_steps = int(np.floor(len(inputs_val)/val_batch_size))
    print 'Steps validation: ', validation_steps
    
    # Run learning
    history = model.fit_generator(generator=training_generator, steps_per_epoch=train_steps, epochs=100, validation_data=validation_generator, validation_steps= validation_steps,callbacks=callbacks_list, workers=8, use_multiprocessing=True)
    
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
    train(pretrain = False)
