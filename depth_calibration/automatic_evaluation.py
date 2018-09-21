import os, sys
from depth_helper import *
import time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
SHAPES_ROOT = os.getcwd().split("/silhouettes/")[0] + "/silhouettes/"
from keras.models import load_model
import keras.losses
import os.path
import glob


weights_path = '/home/mcube/test_data/' # #weights_09-02-2018_last_2/'
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
    if len(glob.glob(directory_save_test+'*.npy')) > 3: continue  #Case where test, train and height have been saved
    if not os.path.exists(directory_save_test): os.makedirs(directory_save_test)
    max_heights = []
    if 1: #len(path_test_images) != len(glob.glob(directory_save_test+'*.png')):
        for img_number, path in enumerate(path_test_images):
            test_image = cv2.imread(path)
            depth_map = raw_gs_to_depth_map(
                test_image=test_image,
                model_path=model_path,
                plot=False,
                save=True,
                path=directory_save_test,
                img_number = img_number+1,
                output_type=output_type,
                model=model,
                input_type=input_type)
            max_heights.append(np.amax(depth_map))
            np.save(directory_save_test + 'gsimg_{}.npy'.format(img_number), test_image)
            np.save(directory_save_test + 'height_{}.npy'.format(img_number), depth_map)
            plt.close()
            print directory_save_test + '/height_{}.npy'.format(img_number)
            assert(False)
        np.save(directory_save_test + 'max_heights.npy', max_heights)
        print max_heights
        #raw_input('Go to: ' + directory_save_test + ' to see the images. OK?')
    else: 
        print 'Assuming test images already computed'

    losses_train = []
    losses_test = []
    mean_losses_train = []; std_losses_train = []
    mean_losses_test = []; std_losses_test = []
    loss_thresholds_train = []
    loss_thresholds_test = []
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
                test_image = cv2.imread(img_path)
                NN_image = cv2.imread(img_path_processed)
                grad_x = np.load(path  + 'gradient/gx_{}.npy'.format(img_number))
                grad_y = np.load(path  + 'gradient/gy_{}.npy'.format(img_number))
                
                grad_x = cv2.resize(grad_x, dsize=(dim_out[1], dim_out[0]), interpolation=cv2.INTER_LINEAR)
                grad_y = cv2.resize(grad_y, dsize=(dim_out[1], dim_out[0]), interpolation=cv2.INTER_LINEAR)

                grad_x = preprocess_label(grad_x)
                grad_y = preprocess_label(grad_y)

                test_depth = poisson_reconstruct(grad_y, grad_x)
                
                depth_map, loss = raw_gs_to_depth_map(
                    test_image=test_image, model_path=model_path,
                    plot=False, save=False, path=path+'converted/', img_number = img_number,
                    output_type=output_type, test_depth=test_depth, model=model, input_type=input_type)

                losses.append(loss)
                '''
                plt.imshow(NN_image)
                plt.show()
                plt.imshow(depth_map)
                plt.show()
                plt.imshow(test_depth)
                plt.show()
                #pdb.set_trace()
                '''
                ### Compute threshold values:
                test_depth[test_depth > 0] = 1
                loss_threshold = []
                for threshold in thresholds:
                    aux_depth_map = copy.deepcopy(depth_map)
                    aux_depth_map[aux_depth_map >= threshold] = 1
                    aux_depth_map[aux_depth_map < threshold] = 0
                    loss_threshold.append(np.sum(np.square(test_depth-aux_depth_map))) #TODO: assumed to be custom loss
                if i:
                    loss_thresholds_test.append(loss_threshold)
                    size_patch_test.append(size_patch)
                else:
                    loss_thresholds_train.append(loss_threshold)
                    size_patch_train.append(size_patch)
                '''
                if 'sphere' not in shape:
                    plt.show()
                    plt.plot(loss_threshold)
                    plt.show()
                '''
            '''
            plt.imshow(depth_map)
            plt.show()
            plt.imshow(test_depth)
            plt.show()
            import pdb; pdb.set_trace()
            '''
            if i:   ## TODO BIT ERROR
                losses_train.append(losses)
                mean_losses_train.append(np.mean(losses))
                std_losses_train.append(np.std(losses))
            else:
                losses_test.append(losses)
                mean_losses_test.append(np.mean(losses))
                std_losses_test.append(np.std(losses))
            #plt.plot(losses, '.', label=shape + '_{}'.format(i))
    losses_train = np.array(losses_train)
    losses_train = np.array(losses_train)
    np.save(directory_save_test + 'losses_train.npy', losses_train)
    np.save(directory_save_test + 'losses_test.npy', losses_test)
    np.save(directory_save_test + 'loss_thresholds_train.npy', loss_thresholds_train)
    np.save(directory_save_test + 'loss_thresholds_test.npy', loss_thresholds_test)
    np.save(directory_save_test + 'contact_patches_train.npy', size_patch_train)
    np.save(directory_save_test + 'contact_patches_test.npy', size_patch_test)
    print 'losses_train: ', mean_losses_train
    print 'losses_test: ', mean_losses_test
    '''
    plt.legend()
    plt.show()
    # loss =return K.sum(K.square(y_true-y_pred))
    import pdb; pdb.set_trace()
    '''
