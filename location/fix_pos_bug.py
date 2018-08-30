from location import Location
from depth_calibration.depth_helper import *

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
    from keras.applications.resnet50 import ResNet50
    from keras.preprocessing import image
    from keras.applications.resnet50 import preprocess_input, decode_predictions
except Exception as e:
    print "Not importing keras"
    
    
loc = Location()

name_id = 'flashlight_l=70_h=20_dx=10_dy=10_rot=0_debug'

gs_id = 2
model_path = '/home/mcube/weights_server_last/weights_type=all_08-23-2018_num=2000_gs_id=2_in=rgb_out=height_epoch=100_NN=basic_aug=5.hdf5'
directory = '/media/mcube/data/shapes_data/object_exploration/'
directory += name_id

keras.losses.custom_loss = custom_loss
model = load_model(model_path)

x_off = 837.2
y_off = 376
z_off = 295

touches = range(0, 17)
pc1 = loc.get_global_pointcloud(gs_id, directory, touches, [], model_path=model_path, model=model)

touches = range(17, 34)
pc2 = loc.get_global_pointcloud(gs_id, directory, touches, [], model_path=model_path, model=model)

touches = range(34, 48)
pc3 = loc.get_global_pointcloud(gs_id, directory, touches, [], model_path=model_path, model=model)

loc.old_visualize3_pointclouds([pc1, pc2, pc3])