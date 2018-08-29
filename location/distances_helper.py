import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from location import Location
import copy

def get_distance_to_axis(point, axis_origin, axis_vector):
    p = np.array(axis_origin)
    q = np.array(p + axis_vector)
    r = np.array(point)

    def t(p, q, r):
        x = p-q
        return np.dot(r-q, x)/np.dot(x, x)

    def d(p, q, r):
        return np.linalg.norm(t(p, q, r)*(p-q)+q-r)

    distance = d(p, q, r)
    return distance

def get_distance_to_plane(point, plane_point, normal_vector):
    normal_vector = normal_vector/np.linalg.norm(normal_vector)
    vector = np.array(point) - np.array(plane_point)
    return abs(np.dot(vector, normal_vector))

def get_cilinder_radius(pointcloud, axis_origin, axis_vector):
    distances = []
    for point in pointcloud:
        distances.append(get_distance_to_axis(point, axis_origin, axis_vector))
    return float(sum(distances))/float(len(distances))

def get_rectangle_side(pointcloud, plane_point, normal_vector):
    distances = []
    for point in pointcloud:
        distances.append(get_distance_to_plane(point, plane_point, normal_vector))
    return float(sum(distances))/float(len(distances))

def get_semicone_metrics(pointcloud, semicone_origin, semicone_axis_vector, visualize=False):
    """
    Returns the radius at the origin and the angle of the semicone
    semicone_origin: point at the center of the base of the semicone
    semicone_axis_vector: normal vector defining the semicone axis
    """
    # Freach point we get the distance to axis and distance to plane
    ax_dist = []
    plane_dist = []
    for point in pointcloud:
        ax_dist.append(get_distance_to_axis(point, semicone_origin, semicone_axis_vector))
        plane_dist.append(get_distance_to_plane(point, semicone_origin, semicone_axis_vector))

    # We fit a line between the points
    model = LinearRegression(fit_intercept=True)
    model.fit(np.array(plane_dist).reshape(-1, 1), np.array(ax_dist))

    if visualize:
        xfit = np.linspace(np.amin(plane_dist), np.amax(plane_dist), 1000)
        yfit = model.predict(xfit[:, np.newaxis])
        plt.scatter(plane_dist, ax_dist)
        plt.plot(xfit, yfit)
        plt.axis('equal')
        plt.show()

    # We return the radius at the beginning and the slope of the line
    r = model.predict(0)[0]
    slope = 180*np.arctan((model.predict(10)[0] - r)/10.)/np.pi
    return r, slope


if __name__=="__main__":
    # Standard testing
    # point = (0, 0, 0)
    # pointcloud = [(1, 1, 0), (1, 2, 0), (2, 1, 0)]
    # origin = (0, 0, 0)
    # vector = (1, 0, 0)
    #
    # print get_distance_to_axis(point, origin, vector)
    # print get_distance_to_plane(point, origin, vector)
    #
    # print "####"
    #
    # print get_cilinder_radius(pointcloud, origin, vector)
    # print get_rectangle_side(pointcloud, origin, vector)
    # print get_semicone_metrics(pointcloud, origin, vector, visualize=False)


    name = 'big_semicone_l=40_h=20_d=10_rot=0_only10.npy'
    name = "cilinder_l=50_h=20_dx=5_dy=5_rot=0_test.npy"
    # name = "cilinder_l=50_h=20_dx=5_dy=5_rot=0_testjust_one_source.npy"
    path = "/home/oleguer/shapes_data/" + name

    global_pointcloud = np.load(path)
    x_off = 860.42
    y_off = 376.8
    z_off = 291

    origin = (0, 0, 0)
    vector = (-1, 0, 0)

    final_global_pointcloud = global_pointcloud
    final_global_pointcloud[:, 0] -= x_off
    final_global_pointcloud[:, 1] -= y_off
    final_global_pointcloud[:, 2] -= z_off


    afinal_global_pointcloud = copy.deepcopy(final_global_pointcloud)
    # afinal_global_pointcloud[:,0] = final_global_pointcloud[:,2]
    # afinal_global_pointcloud[:,1] = final_global_pointcloud[:,1]
    # afinal_global_pointcloud[:,2] = -final_global_pointcloud[:,0]

    print get_cilinder_radius(final_global_pointcloud, origin, vector)
    print get_semicone_metrics(final_global_pointcloud, origin, vector, visualize=True)

    ax = []
    for i in range(100):
        ax.append((-i, 0, 0))

    afinal_global_pointcloud = np.concatenate([afinal_global_pointcloud, ax])
    loc = Location()
    loc.visualize_pointcloud(afinal_global_pointcloud)
