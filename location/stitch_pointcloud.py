# src/Python/Tutorial/Basic/working_with_numpy.py

import copy
import numpy as np
from open3d import *
import pdb



def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    #source_temp.paint_uniform_color([1, 0.706, 0])
    #target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    draw_geometries([source_temp, target_temp])

if __name__ == "__main__":

    # generate some neat n times 3 matrix using a variant of sync function
    x = np.linspace(-3, 3, 401)
    mesh_x, mesh_y = np.meshgrid(x,x)
    z = np.sinc((np.power(mesh_x,2)+np.power(mesh_y,2)))
    z_norm = (z-z.min())/(z.max()-z.min())
    xyz = np.zeros((np.size(mesh_x),3))
    xyz[:,0] = np.reshape(mesh_x,-1)
    xyz[:,1] = np.reshape(mesh_y,-1)
    xyz[:,2] = np.reshape(z_norm,-1)
    print('xyz')
    print(xyz)

    # Pass xyz to Open3D.PointCloud and visualize
    xyz = np.load('pointcloud_bar.npy')
    
    pcd = PointCloud()
    pcd.points = Vector3dVector(xyz)
    pcd_name = "sync.ply"
    write_point_cloud(pcd_name, pcd)

    xyz = np.load('pcd_6_front_bar2.npy')
    #xyz = xyz[100000:150000,:]
    pcd2 = PointCloud()
    pcd2.points = Vector3dVector(xyz)
    pcd_name2 = "sync2.ply"
    write_point_cloud(pcd_name2, pcd2)


    # Load saved point cloud and visualize it
    #draw_geometries([pcd])
    draw_geometries([pcd2])
    
    data = np.load('only_front.npy')
    #pdb.set_trace()
    pcd3 = PointCloud()
    pcd3.points = Vector3dVector(data)
    #pcd_name = "sync.ply"
    #write_point_cloud(pcd_name, pcd)
    #draw_geometries([pcd3])
    
    
    source = pcd
    target = pcd2
    threshold = 2
    trans_init = np.asarray(
                [[1,0,0,  120],
                [0,1,0,  0],
                [0,0,1, 0],
                [0.0, 0.0, 0.0, 1.0]])
    draw_registration_result(source, target, trans_init)
    print("Initial alignment")
    evaluation = evaluate_registration(source, target,
            threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = registration_icp(source, target, threshold, trans_init,
            TransformationEstimationPointToPoint(),
            ICPConvergenceCriteria(max_iteration = 2000))
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    draw_registration_result(source, target, reg_p2p.transformation)

    print("Apply point-to-plane ICP")
    reg_p2l = registration_icp(source, target, threshold, trans_init,
            TransformationEstimationPointToPlane())
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    print("")
    draw_registration_result(source, target, reg_p2l.transformation)
