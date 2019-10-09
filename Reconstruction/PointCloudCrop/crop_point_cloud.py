import numpy
import cv2
import sys
import os
import open3d
sys.path.append("../Data_processing")

import TileInfo


def load_tile_image_as_points_and_color(img_path, width_by_m, height_by_m):
    img_bgr = cv2.imread(img_path)
    img_array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    (height_by_pixel, width_by_pixel, value_length) = img_array.shape
    # Normal direction: x+ direction. uv -> yz. Up and right defined as positive.
    # Different from the coordinates of image pixels but suitable for visualization.
    pixel_amount = height_by_pixel * width_by_pixel
    points_position = numpy.mgrid[(height_by_pixel / 2):(-height_by_pixel + height_by_pixel / 2):-1,
                      (-width_by_pixel / 2):(width_by_pixel - width_by_pixel / 2):1].reshape(2, -1).T
    points_position = numpy.c_[
        numpy.zeros(pixel_amount),
        points_position[:, 1] * width_by_m / width_by_pixel,
        points_position[:, 0] * height_by_m / height_by_pixel
    ]

    points_color = img_array.reshape(-1, 3)
    points_color = points_color / 255.0

    # points_colors_array = numpy.c_[points_position, points_color]
    # points_colors_array_preserved = points_colors_array[numpy.logical_and(
    #     numpy.logical_and(points_colors_array[:, 1] <= width_by_m / 2 - 0.002,
    #                      points_colors_array[:, 1] >= -width_by_m / 2 + 0.002),
    #     numpy.logical_and(points_colors_array[:, 2] <= height_by_m / 2 - 0.001,
    #                      points_colors_array[:, 2] >= -height_by_m / 2 + 0.001)), :]
    # points_position = points_colors_array_preserved[:, 0:3]
    # points_color = points_colors_array_preserved[:, 3:6]

    return points_position, points_color


def crop_color_point_cloud(points, colors, source_trans, target_trans, target_tile_width_by_m, target_tile_height_by_m):
    # print("numpy.asarray(points).T.shape")
    # print(numpy.asarray(points).T.shape)
    s_center = numpy.dot(source_trans, numpy.array([0, 0, 0, 1]).T).T[0:3]
    t_center = numpy.dot(target_trans, numpy.array([0, 0, 0, 1]).T).T[0:3]

    s_end = numpy.dot(source_trans, numpy.array([0.00001, 0, 0, 1]).T).T[0:3]
    t_end = numpy.dot(target_trans, numpy.array([0.00001, 0, 0, 1]).T).T[0:3]

    center_distance = numpy.linalg.norm(s_center - t_center)
    end_distance = numpy.linalg.norm(s_end - t_end)

    points_array = numpy.append(numpy.asarray(points).T, numpy.ones((1, len(points))), axis=0)

    full_trans_matrix = numpy.dot(numpy.linalg.inv(target_trans), source_trans)
    trans_restore_matrix = numpy.linalg.inv(full_trans_matrix)

    # points_colors_array = numpy.append(numpy.dot(full_trans_matrix, points_array), numpy.asarray(colors).T, axis=0)
    points_colors_array = numpy.append(numpy.dot(full_trans_matrix, points_array), numpy.asarray(points).T, axis=0)
    points_colors_array = numpy.append(points_colors_array, numpy.asarray(colors).T, axis=0)



    if center_distance <= end_distance:
        # points_colors_preserved = \
        #     points_colors_array[:, points_colors_array[0, :] <= 0.0000003]
        points_colors_preserved = \
            points_colors_array[:, numpy.logical_or(points_colors_array[0, :] <= 0.0000003,
                                                    numpy.logical_or(numpy.logical_or(points_colors_array[1, :] >= target_tile_width_by_m / 2,
                                                                                      points_colors_array[1, :] <= -target_tile_width_by_m / 2),
                                                                     numpy.logical_or(points_colors_array[2, :] >= target_tile_height_by_m / 2,
                                                                                      points_colors_array[2, :] <= -target_tile_height_by_m / 2)))]
    else:
        # points_colors_preserved = \
        #     points_colors_array[:, points_colors_array[0, :] >= -0.0000003]

        points_colors_preserved = points_colors_array

        # points_colors_preserved = \
        #     points_colors_array[:, numpy.logical_or(points_colors_array[0, :] >= -0.0000003,
        #                                             numpy.logical_or(numpy.logical_or(points_colors_array[1, :] >= target_tile_width_by_m / 2 - 0.002,
        #                                                                               points_colors_array[1, :] <= -target_tile_width_by_m / 2 + 0.002),
        #                                                              numpy.logical_or(points_colors_array[2, :] >= target_tile_height_by_m / 2 - 0.001,
        #                                                                               points_colors_array[2, :] <= -target_tile_height_by_m / 2 + 0.001)))]
    # points_preserved = numpy.dot(trans_restore_matrix, points_colors_preserved[0:4, :])[0:3, :].T
    points_preserved = points_colors_preserved[4:7, :].T
    colors_preserved = points_colors_preserved[7:10, :].T

    return points_preserved, colors_preserved


def generate_cropped_tile(tile_index, tile_info_dict, img_directory_path):
    print("Generating cropped tile %6d" % tile_index)
    tile_info = tile_info_dict[tile_index]
    points, colors = \
        load_tile_image_as_points_and_color(
            img_path=os.path.join(img_directory_path, tile_info.file_name) + ".png",
            width_by_m=tile_info.width_by_m,
            height_by_m=tile_info.height_by_m)
    # pcd = open3d.PointCloud()
    # coor = open3d.geometry.create_mesh_coordinate_frame(size=0.01, origin=(0, 0, 0))
    # pcd.points = open3d.Vector3dVector(points)
    # pcd.colors = open3d.Vector3dVector(colors)
    # open3d.draw_geometries([pcd, coor])
    for adjacent_tile_index in tile_info.confirmed_neighbour_list:
        tile_info_target = tile_info_dict[adjacent_tile_index]
        points, colors = crop_color_point_cloud(points=points, colors=colors,
                                                source_trans=tile_info.pose_matrix,
                                                target_trans=tile_info_target.pose_matrix,
                                                target_tile_width_by_m=tile_info_target.width_by_m,
                                                target_tile_height_by_m=tile_info_target.height_by_m)
    # pcd = open3d.PointCloud()
    # coor = open3d.geometry.create_mesh_coordinate_frame(size=0.01, origin=(0, 0, 0))
    # pcd.points = open3d.Vector3dVector(points)
    # pcd.colors = open3d.Vector3dVector(colors)
    # open3d.draw_geometries([pcd, coor])
    return points, colors
