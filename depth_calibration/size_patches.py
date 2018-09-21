import os, sys
from depth_helper import *
import time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
SHAPES_ROOT = os.getcwd().split("/silhouettes/")[0] + "/silhouettes/"
from keras.models import load_model
import keras.losses
import os.path
import glob


weights_path = '/home/mcube/weights_09-02-2018_last_true/'
weights_files = glob.glob(weights_path + '*.hdf5')
weights_files.sort(key=os.path.getmtime)
for weights_file in weights_files:
    print weights_file
    #weights_file = '/home/mcube/weights_server_3h/weights_type=all_08-23-2018_num=2000_gs_id=2_in={}_out={}_epoch=100_NN=basic_aug=5.hdf5'.format(input_type,output_type)
    gs_id = 1
    dim_out = [225,243]
    shapes = ['sphere', 'semicone_1', 'semicone_2','hollowcone_2', 'semipyramid_3']
    shapes = ['sphere', 'semicone_1', 'semicone_2','hollowcone_3','semipyramid_3']
    if 'grad' in weights_file: output_type = 'grad'
    elif 'angle' in weights_file: output_type = 'angle'
    else: output_type = 'height'
    if 'gray' in weights_file: input_type = 'gray'
    else: input_type = 'rgb'
    path_train = '/media/mcube/data/shapes_data/processed_09-02-2018/{}_09-02-2018_gs_id=1_rot=0/' #.format(gs_id)
    path_test = '/media/mcube/data/shapes_data/processed_09-02-2018/{}_09-02-2018_test_gs_id=1_rot=0/'#.format(gs_id)

    keras.losses.custom_loss = custom_loss
    model_path = weights_file
    model = load_model(model_path)
    ### Training
    train_data = 500
    test_data = 100
    thresholds = np.linspace(0, 1, num=101)
    def custom_loss(y_true, y_pred):
        return K.sum(K.square(y_true-y_pred))

    ## First evaluate test imatges:
    date = '09-02-2018'
    test_path = '/media/mcube/data/shapes_data/raw/test_{}_gs_id={}/'.format(date,gs_id)
    path_test_images = glob.glob(test_path + '*.png')
    path_test_images.sort(key=os.path.getmtime)

    directory_save_test = weights_file[:-5] + '/'
    #if len(glob.glob(directory_save_test+'*.npy')) > 3: continue  #Case where test, train and height have been saved
    if not os.path.exists(directory_save_test): os.makedirs(directory_save_test)
    max_heights = []
    size_patch_test = []
    size_patch_train = []
    for shape in shapes: 
        for i in range(2): #TODO
            print 'i = ', i
            losses = []
            if i:
                path = path_test.format(shape)
                pictures_range = np.arange(1,test_data*5,5)
            else:
                path = path_train.format(shape)
                pictures_range = np.arange(1,train_data*5,5)
            ## Get all sizes patches:
            
            for img_number in pictures_range:
                #import pdb; pdb.set_trace()
                print 'number: ', img_number
                print 'path: ', path
                img_path = path + 'image_raw/img_{}.png'.format(img_number)
                img_path_processed = path + 'image/img_{}.png'.format(img_number)
                size_patch = float(glob.glob(path + 'image_circled/img_{} *.png'.format(img_number))[0][:-4].split()[-1])
                if not os.path.exists(img_path):
                    print 'NOT HERE: ', img_path
                    continue
                
                if i:
                    
                    size_patch_test.append(size_patch)
                else:
                    
                    size_patch_train.append(size_patch)
               
    np.save(directory_save_test + 'contact_patches_train.npy', size_patch_train)
    np.save(directory_save_test + 'contact_patches_test.npy', size_patch_test)
    
