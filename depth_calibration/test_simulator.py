from depth_helper import *

data_path = '/media/mcube/data/shapes_data/PROCESSED/semicone_1_augmented/gradient/'
model_path = "weights/weights.test_sim_v3.hdf5"

for num in range(100, 120):
# num = 111

    gx = np.load(data_path + 'gx_' + str(num) + '.npy')
    gy = np.load(data_path + 'gy_' + str(num) + '.npy')

    print gx.shape
    print float(gx.shape[0])/float(gx.shape[1])
    print float(gy.shape[0])/float(gy.shape[1])

    depth_map = poisson_reconstruct(gy, gx)
    print float(depth_map.shape[0])/float(depth_map.shape[1])
    # plot_depth_map(depth_map, top_view=True)

    img = grad_to_gs(model_path, gx, gy)/255.
    print float(img.shape[0])/float(img.shape[1])
    # print img.shape

    real = cv2.imread('/media/mcube/data/shapes_data/PROCESSED/semicone_1_processed_h_mm/image/img_' + str(num) + '.png')

    print float(real.shape[0])/float(real.shape[1])

    real = cv2.resize(real, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    # depth_map = cv2.resize(depth_map, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    cv2.imshow('sim', img)
    cv2.imshow('real', real)
    cv2.imshow('height', depth_map)
    cv2.imshow('gx', gx)
    cv2.waitKey(0)
