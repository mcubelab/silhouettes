import os, sys
from depth_helper import *
import time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
SHAPES_ROOT = os.getcwd().split("/silhouettes/")[0] + "/silhouettes/"
from keras.models import load_model
import keras.losses
import os.path
import glob

output_type = 'grad'
weights_file = 'depth_calibration/weights/weights_semipyramid_2_08-17-2018_gs_id=1_rot=all_out_type={}.hdf5'.format(output_type)
gs_id = 2
dim_out = [225,243]
shapes = ['sphere', 'semicone_2','hollowcone_2', 'hollowcone_1', 'semipyramid_2']
path_train = '/media/mcube/data/shapes_data/processed/{}_08-17-2018_gs_id=1_rot={}/' #.format(gs_id)
path_test = '/media/mcube/data/shapes_data/prcoessed/{}_08-17-2018_test_gs_id=1_rot={}/'#.format(gs_id)

keras.losses.custom_loss = custom_loss
model_path = SHAPES_ROOT + weights_file
model = load_model(model_path)
### Training
train_data = 500
test_data = 100
thresholds = np.linspace(0, 1, num=100)
def custom_loss(y_true, y_pred):
    return K.sum(K.square(y_true-y_pred))

## First evaluate test imatges:
date = '08-17-2018'
test_path = '/media/mcube/data/shapes_data/raw/test_{}_gs_id={}/'.format(date,gs_id)
path_test_images = glob.glob(test_path + '*.png')
directory_save_test = '/media/mcube/data/shapes_data/weights_results/' + weights_file[:-5]
if not os.path.exists(directory_save_test): os.makedirs(directory_save_test)
for img_number, path in enumerate(path_test_images):
    test_image = cv2.imread(img_path)
    depth_map = raw_gs_to_depth_map(
        test_image=test_image,
        model_path=model_path,
        plot=False,
        save=True,
        path=directory_save_test,
        img_number = img_number+1,
        output_type=output_type,
        model=model)
    raw_input('Go to: ' + directory_save_test + ' to see the images. OK?')

for shape in shapes: 
    for i in range(2): #TODO
        losses = []
        if 'semipyramid' in shape or 'stamp' in shape:
            rotations = range(4)
        else:
            rotations = [0]
        for rotation in rotations:        
            if i == 0:
                pictures_range = range(train_data)
                path = path_train.format(shape, rotation)
            else:
                pictures_range = range(test_data)
                path = path_test.format(shape, rotation)
            for img_number in pictures_range:
                print 'number: ', img_number
                print 'path: ', path
                img_path = path + 'image_raw/img_{}.png'.format(img_number)
                img_path = path + 'image/img_{}.png'.format(img_number)
                if not os.path.exists(img_path):
                    continue
                test_image = cv2.imread(img_path)
                NN_image = cv2.imread(img_path)
                grad_x = np.load(path  + 'gradient/gx_{}.npy'.format(img_number))
                grad_y = np.load(path  + 'gradient/gy_{}.npy'.format(img_number))
                
                grad_x = cv2.resize(grad_x, dsize=(dim_out[1], dim_out[0]), interpolation=cv2.INTER_LINEAR)
                grad_y = cv2.resize(grad_y, dsize=(dim_out[1], dim_out[0]), interpolation=cv2.INTER_LINEAR)

                grad_x = preprocess_label(grad_x)
                grad_y = preprocess_label(grad_y)

                test_depth = poisson_reconstruct(grad_y, grad_x)
                
                depth_map, loss = raw_gs_to_depth_map(
                    test_image=test_image,
                    model_path=model_path,
                    plot=False,
                    save=False,
                    path=path+'converted/',
                    img_number = img_number,
                    output_type=output_type,
                    test_depth=test_depth,
                    model=model)

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
        plt.plot(losses, '.', label=shape + '_{}'.format(i))
plt.legend()
plt.show()
