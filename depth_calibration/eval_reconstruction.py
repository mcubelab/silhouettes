import os, sys
from depth_helper import *

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
SHAPES_ROOT = os.getcwd().split("/silhouettes/")[0] + "/silhouettes/"


path = "/media/mcube/data/shapes_data/test_objects/"; pictures_range = range(1, 15)
#path = "/media/mcube/data/shapes_data/ball_D28.5/"; pictures_range = range(40,3011)
#path = "/media/mcube/data/shapes_data/semicone_1/"; pictures_range = range(1015,2011)

for img_number in pictures_range:


    test_image = cv2.imread(path + "GS2_" + str(img_number) + '.png')
    weights_file = 'depth_calibration/weights/weights.v1.hdf5'
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
        img_number = img_number)
