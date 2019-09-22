import numpy
import transforms3d
import json
import open3d

import sys

sys.path.append("../Utility")
import file_managing

import os.path
from scipy.spatial.transform import Rotation
import math

# rotation = numpy.array([[-1, 0, 0],
#                         [0, 1, 0],
#                         [0, 0, -1]])
# rotation = numpy.dot(rotation, Rotation.from_euler("xyz", [0, 0, -math.pi/2]).as_dcm())
# print(rotation)


# ================================================================================================================
trans_rob_to_img_matrix = transforms3d.affines.compose(T=[0, 0, 0],
                                                          R=[[0, -1, 0],
                                                             [-1, 0, 0],
                                                             [0, 0, -1]],
                                                          Z=[1, 1, 1])

plane_shifting_rotation_trans_matrix = numpy.array([[0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [1, 0, 0, 0],
                                                    [0, 0, 0, 1]])
trans_rob_to_img_matrix = numpy.dot(trans_rob_to_img_matrix, plane_shifting_rotation_trans_matrix)
trans_img_to_rob_matrix = numpy.linalg.inv(trans_rob_to_img_matrix)

# ================================================================================================================
trans_rob_to_camera_matrix = transforms3d.affines.compose(T=[0, 0, 0],
                                                          R=[[-1, 0, 0],
                                                             [0, -1, 0],
                                                             [0, 0, -1]],
                                                          Z=[1, 1, 1])
plane_shifting_rotation_trans_matrix = numpy.array([[0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [1, 0, 0, 0],
                                                    [0, 0, 0, 1]])
trans_rob_to_camera_matrix = numpy.dot(trans_rob_to_camera_matrix, plane_shifting_rotation_trans_matrix)
trans_camera_to_rob_matrix = numpy.linalg.inv(trans_rob_to_camera_matrix)


# get the RGBD camera pose
trans_to_rgbd_camera_matrix = numpy.array([[0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [1, 0, 0, 0],
                                           [0, 0, 0, 1]])


def rob_pose_to_img_trans(rob_pose):
    return numpy.dot(numpy.asarray(rob_pose), trans_rob_to_img_matrix)


def trans_to_rgbd_camera_pose(trans):
    return numpy.dot(numpy.asarray(trans), trans_camera_to_rob_matrix).T.reshape((-1)).tolist()


def read_points_list(path):
    points_list = json.load(open(path, "r"))
    return points_list


def read_trans_list(path, exclude_beginning_n=0):
    trans_data_list = json.load(open(path, "r"))
    trans_list = []
    for i, trans_data in enumerate(trans_data_list):
        if i >= exclude_beginning_n:
            trans_list.append(numpy.asarray(trans_data))
    return trans_list


def save_trans_list(path, trans_list):
    data_to_save = []
    for trans in trans_list:
        data_to_save.append(trans.tolist())
    json.dump(data_to_save, open(path, "w"), indent=4)


def save_robotic_full_pose_list(path, robotic_full_pose_list):
    json.dump(robotic_full_pose_list, open(path, "w"), indent=4)


# can also be used to make a sampled list.
def adjust_order(source_list, index_list):
    target_list = []
    for index in index_list:
        target_list.append(source_list[index])
    return target_list


def trans_list_to_points(trans_list):
    point_list = []
    for trans in trans_list:
        # print(trans)
        point = numpy.dot(trans, numpy.asarray([0, 0, 0, 1]).T).T[0:3].tolist()
        point_list.append(point)
    return point_list


def trans_list_to_points_normals(trans_list, original_normal=[1.0, 0, 0]):
    point_list = []
    normal_list = []
    for trans in trans_list:
        point = numpy.dot(trans, numpy.asarray([0, 0, 0, 1]).T).T[0:3].tolist()
        normal = numpy.dot(trans, numpy.asarray([original_normal[0],
                                                 original_normal[1],
                                                 original_normal[2],
                                                 0]).T).T[0:3].tolist()
        point_list.append(point)
        normal_list.append(normal)
    return point_list, normal_list


def trans_list_to_pos_ori_list(trans_list):
    robotic_full_pose_list = []
    for trans in trans_list:
        trans = numpy.dot(trans, trans_camera_to_rob_matrix)
        [T, R, Z, S] = transforms3d.affines.decompose(trans)
        quaternion = Rotation.from_dcm(R).as_quat()
        robotic_full_pose_list.append([T.tolist(), quaternion.tolist()])
    return robotic_full_pose_list


def non_outlier_index_list_static(point_list, nb_neighbours=20, std_ratio=2.0):
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(point_list)

    cl, ind = open3d.geometry.statistical_outlier_removal(pcd,
                                                          nb_neighbors=nb_neighbours,
                                                          std_ratio=std_ratio)
    display_inlier_outlier(pcd, ind)


def non_outlier_index_list_radius(point_list, nb_points=20, radius=0.01):
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(point_list)

    cl, ind = open3d.geometry.radius_outlier_removal(pcd,
                                                     nb_points=nb_points,
                                                     radius=radius)
    display_inlier_outlier(pcd, ind)


def display_inlier_outlier(cloud, ind):
    inlier_cloud = open3d.geometry.select_down_sample(cloud, ind)
    outlier_cloud = open3d.geometry.select_down_sample(cloud, ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    open3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
