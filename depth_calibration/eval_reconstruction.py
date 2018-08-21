import os, sys
from depth_helper import *

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
SHAPES_ROOT = os.getcwd().split("/silhouettes/")[0] + "/silhouettes/"


path = "/media/mcube/data/shapes_data/test_objects/"; pictures_range = range(1, 15)
#path = "/media/mcube/data/shapes_data/ball_D28.5/"; pictures_range = range(40,3011)
#path = "/media/mcube/data/shapes_data/semicone_1/"; pictures_range = range(1015,2011)
# pictures_range = [46,103,210,217,334]
for img_number in pictures_range:

    img_path = path + "GS2_" + str(img_number) + '.png'
    # img_path ='/media/mcube/data/shapes_data/pos_calib/bar_front/p_{}/GS2_0.png'.format(img_number)
    # img_path ='/media/mcube/data/shapes_data/height_test/GS2_{}.png'.format(img_number)
    test_image = cv2.imread(img_path)
    output_type = 'height'
    weights_file = 'depth_calibration/weights/weights_sphere_08-15-2018_out_type=angle.hdf5'
    #weights_file = 'depth_calibration/weights/weights.aug.v2.hdf5'


    '''
      #Investigate if we can change distribution
    test_image2 = cv2.imread(path + "GS2_" + str(1) + '.png')
    cv2.imshow('', test_image)
    for it in range(3):
        print np.mean(test_image[:,:,it])
        print np.mean(test_image2[:,:,it])
        test_image[:,:,it] = test_image[:,:,it]/np.mean(test_image[:,:,it])*np.mean(test_image2[:,:,it])
        print np.mean(test_image[:,:,it])

    #cv2.imshow('', test_image2)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
    '''

    depth_map = raw_gs_to_depth_map(
        gs_id = 2,
        test_image=test_image,
        model_path=SHAPES_ROOT + weights_file,
        plot=True,
        save=False,
        path=path+'converted/',
        img_number = img_number,
        output_type=output_type)
