from depth_helper import *

data_path = '/media/mcube/data/shapes_data/PROCESSED/semicone_1_processed_h_mm/gradient/'
#data_path = '/media/mcube/data/shapes_data/PROCESSED/processed_color_D28.5_h_mm/gradient/'
model_path = "weights/weights.test_sim_v2.hdf5"

num = 111

gx = np.load(data_path + 'gx_' + str(num) + '.npy')
gy = np.load(data_path + 'gy_' + str(num) + '.npy')

img = grad_to_gs(model_path, gx, gy)/255.
# print img.shape

cv2.imshow('sim', img)
cv2.waitKey(0)
