from depth_helper import *
import numpy as np
import cv2


for num in [7,8,25]:
    print num
    test_image = cv2.imread("/media/mcube/data/shapes_data/processed_08-24-2018_test/hollowcone_3_08-24-2018_gs_id=2_rot=0/image_raw/img_{}.png".format(num))
    test_image = cv2.imread("/media/mcube/data/shapes_data/raw/test_08-17-2018_gs_id=2/GS2_{}.png".format(num))

    model_path = '/home/mcube/weights_server_last/weights_type=all_08-23-2018_num=2000_gs_id=2_in=rgb_out=height_epoch=100_NN=basic_aug=5.hdf5'

    dm = raw_gs_to_depth_map(gs_id=2, test_image=test_image, ref=None, model_path=model_path, plot=False, save=False, path='', img_number='', output_type='height', test_depth=None, model=None, input_type='rgb', get_NN_input = False)
        
    plot_depth_map(dm, show=True, save=False, path='', img_number = '', top_view=True, palette=cm.Reds)
