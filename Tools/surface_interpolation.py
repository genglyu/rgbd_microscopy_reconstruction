import open3d
import numpy
import scipy
import json
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def read_raw_robotic_pose_json_as_points(path):
    raw_robotic_pose_list = json.load(open(path,"r"))
    points = []
    for pose in raw_robotic_pose_list:
        points.append([pose[12], pose[13], pose[14]])
    return points


def find_bonding_box(points):
    x_max = points[0][0]
    x_min = points[0][0]
    y_max = points[0][1]
    y_min = points[0][1]
    z_max = points[0][2]
    z_min = points[0][2]
    for [x, y, z] in points:
        if x_max < x : x_max = x
        if x_min > x : x_min = x
        if y_max < y : y_max = y
        if y_min > y : y_min = y
        if z_max < z : z_max = z
        if z_min > z : z_min = z
    return [[x_min, y_min, z_min], [x_max, y_max, z_max]]


def recenter(points, reference):
    processed_points = []
    for point in points:
        processed_points.append((numpy.asarray(point) - numpy.asarray(reference)).tolist())
    return processed_points


def interpolate_points_z(points):
    points = numpy.asarray(list)
    [[x_min, y_min, z_min], [x_max, y_max, z_max]] = find_bonding_box(points)
    xy = points[:, 0:2]
    z = points[:, 2]
    grid_x, grid_y = numpy.mgrid[x_min:x_max:(200 * 1j), y_min:y_max:(200 * 1j)]
    grid_z = griddata(xy, z, (grid_x, grid_y), method='cubic')
    interpolate_points = numpy.c_[grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)]
    interpolate_points = interpolate_points[numpy.logical_not(numpy.isnan(interpolate_points[:, 2]))]
    return interpolate_points


list = read_raw_robotic_pose_json_as_points("/home/lvgeng/writing/microscope_project/fruit/testing_pose.testingjson")
list = recenter(list, find_bonding_box(list)[0])





# pcd = open3d.geometry.PointCloud()
# pcd.points = open3d.utility.Vector3dVector(interpolate_points)
# pcd.colors = open3d.utility.Vector3dVector(numpy.repeat([[1, 0.5, 0.5]], len(interpolate_points), axis=0))
# pcd.estimate_normals()
#
#
# viewer = open3d.visualization.VisualizerWithEditing()
# viewer.create_window(width=800, height=600)
# viewer.add_geometry(pcd)
# viewer.run()
#
# viewer.destroy_window()

