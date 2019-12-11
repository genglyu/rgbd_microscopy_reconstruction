import sys

sys.path.append("../Data_processing")
sys.path.append("../../Utility")
import image_processing
import TileInfoDict
import cv2
import crop_point_cloud
import open3d
import file_managing

# ======================================================================================================================
# def make_full_image_pcd_list_rgbd(tile_info_dict, color_directory_path,
#                                   depth_directory_path, depth_scale, depth_trunc,
#                                   width_by_pixel, height_by_pixel, width_by_m, height_by_m,
#                                   focal_distance_by_m,
#                                   downsample_factor=-1.0,
#                                   color_filter=[1.0, 1.0, 1.0]):
#     microcsope_intrinsic = \
#         generate_microscope_intrinsic_open3d(width_by_pixel=width_by_pixel,
#                                              height_by_pixel=height_by_pixel,
#                                              width_by_m=width_by_m,
#                                              height_by_m=height_by_m,
#                                              focal_distance_by_m=focal_distance_by_m)
#     pcd_list = []
#     for tile_info_key in tile_info_dict:
#         tile_info = tile_info_dict[tile_info_key]
#         color = io.read_image(join(color_directory_path, tile_info.file_name) + ".png")
#         depth = io.read_image(join(depth_directory_path, tile_info.file_name) + ".png")
#         rgbd = geometry.create_rgbd_image_from_color_and_depth(
#             color=color, depth=depth,
#             depth_scale=depth_scale,
#             depth_trunc=depth_trunc,
#             convert_rgb_to_intensity=False)
#         pcd = geometry.create_point_cloud_from_rgbd_image(rgbd, microcsope_intrinsic)
#         pcd.transform(tile_info.rgbd_camera_pose_matrix)
#         pcd_list.append(pcd)
#     return pcd_list


# openCV involved in these two functions.


def make_full_image_pcd_list_pose(tile_info_dict,
                                  path_data,
                                  dataset_group_template,
                                  color_directory_path,
                                  downsample_factor=-1.0,
                                  color_filter=[1.0, 1.0, 1.0]):
    pcd_list = []
    for tile_info_key in tile_info_dict:
        tile_info = tile_info_dict[tile_info_key]
        if len(tile_info.confirmed_neighbour_list) > 0:
            # img = cv2.imread(color_directory_path + tile_info.file_name + ".png")
            img = cv2.imread(
                file_managing.join(path_data,
                                   dataset_group_template % tile_info.tile_group,
                                   color_directory_path + tile_info.file_name) + ".png")
            pcd = image_processing.load_image_as_planar_point_cloud_open3d(
                image_bgr=img,
                width_by_mm=tile_info.width_by_mm,
                height_by_mm=tile_info.height_by_mm,
                cv_scale_factor=downsample_factor,
                color_filter=color_filter * tile_info.color_and_illumination_correction)
            pcd.transform(tile_info.pose_matrix)
            pcd_list.append(pcd)
        else:
            print("isolated tile %6d" % tile_info_key)
    return pcd_list


def make_cropped_image_pcd_list_pose(tile_info_dict, img_directory_path):
    pcd_list = []
    for tile_info_key in tile_info_dict:
        if len(tile_info_dict[tile_info_key].confirmed_neighbour_list) > 0:
            points, colors = crop_point_cloud.generate_cropped_tile(tile_index=tile_info_key, tile_info_dict=tile_info_dict,
                                                                    img_directory_path=img_directory_path)
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(points)
            pcd.colors = open3d.utility.Vector3dVector(colors * tile_info_dict[tile_info_key].color_and_illumination_correction)
            pcd.transform(tile_info_dict[tile_info_key].pose_matrix)

            pcd_list.append(pcd)
        else:
            print("isolated tile %6d" % tile_info_key)
    return pcd_list


def make_full_image_pcd_list_sensor_with_label(tile_info_dict,
                                               path_data,
                                               dataset_group_template,
                                               color_directory_path, downsample_factor=-1.0,
                                               size_factor=0.15, color_filter=[1.0, 1.0, 1.0]):
    pcd_list = []
    for tile_info_key in tile_info_dict:
        tile_info = tile_info_dict[tile_info_key]
        if len(tile_info.confirmed_neighbour_list) > 0:
            img = cv2.imread(
                file_managing.join(path_data,
                                   dataset_group_template % tile_info.tile_group,
                                   color_directory_path + tile_info.file_name) + ".png")

            cv2.putText(img=img, text=("%d" % tile_info_key), org=(200, 400),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=5,
                        color=(0, 255, 0),
                        thickness=5,
                        lineType=cv2.LINE_AA,
                        bottomLeftOrigin=False)
            pcd = image_processing.load_image_as_planar_point_cloud_open3d(image_bgr=img,
                                                                           width_by_mm=tile_info.width_by_mm * size_factor,
                                                                           height_by_mm=tile_info.height_by_mm * size_factor,
                                                                           cv_scale_factor=downsample_factor,
                                                                           color_filter=color_filter)
            pcd.transform(tile_info.init_transform_matrix)
            pcd_list.append(pcd)
        else:
            print("Too isolated tile %6d" % tile_info_key)
    return pcd_list


def make_full_image_pcd_list_sensor(tile_info_dict, color_directory_path, downsample_factor=-1.0,
                                    color_filter=[1.0, 1.0, 1.0]):
    pcd_list = []
    for tile_info_key in tile_info_dict:
        tile_info = tile_info_dict[tile_info_key]
        if len(tile_info.confirmed_neighbour_list) > 0:
            img = cv2.imread(color_directory_path + tile_info.file_name + ".png")

            pcd = image_processing.load_image_as_planar_point_cloud_open3d(image_bgr=img,
                                                                           width_by_mm=tile_info.width_by_mm,
                                                                           height_by_mm=tile_info.height_by_mm,
                                                                           cv_scale_factor=downsample_factor,
                                                                           color_filter=color_filter)
            pcd.transform(tile_info.init_transform_matrix)
            pcd_list.append(pcd)
        else:
            print("Too isolated tile %6d" % tile_info_key)
    return pcd_list


# ======================================================================================================================
# ======================================================================================================================
# For functions below, only open3d is used for visualization
# ======================================================================================================================


def make_connection_of_pcd_order(pcd, color=[0, 0, 0]):
    connection = open3d.geometry.LineSet()
    connection.points = pcd.points
    lines = []
    colors = []
    for i, point in enumerate(pcd.points):
        if i > 0:
            lines.append([i - 1, i])
            colors.append(color)
    connection.lines = open3d.utility.Vector2iVector(lines)
    connection.colors = open3d.utility.Vector3dVector(colors)
    return connection


def make_point_cloud(points, color=[0.0, 0.0, 0.0], normals=None):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(TileInfoDict.numpy.repeat([color], len(points), axis=0))
    if normals is not None:
        pcd.normals = open3d.utility.Vector3dVector(normals)
        pcd.normalize_normals()
    return pcd


def make_edge_set(points, edges, color=[0.0, 0.0, 0.0]):
    edge_set = open3d.geometry.LineSet()
    edge_set.points = open3d.utility.Vector3dVector(points)
    edge_set.lines = open3d.utility.Vector2iVector(edges)
    edge_set.colors = open3d.utility.Vector3dVector(TileInfoDict.numpy.repeat([color], len(edges), axis=0))
    return edge_set


def make_pose_sensor_edge_set(tile_info_dict, color=[0.0, 0.0, 0.0]):
    """:returns points as List[n, 3] cause it's only used for sensor edges"""
    points = []
    edges = []
    for i, tile_info_key in enumerate(tile_info_dict):
        points.append(TileInfoDict.numpy.dot(tile_info_dict[tile_info_key].pose_matrix,
                                             TileInfoDict.numpy.asarray([0, 0, 0, 1]).T).T[0:3])
        points.append(TileInfoDict.numpy.dot(tile_info_dict[tile_info_key].init_transform_matrix,
                                             TileInfoDict.numpy.asarray([0, 0, 0, 1]).T).T[0:3])
        edges.append([i * 2, i * 2 + 1])
    edge_set = make_edge_set(points, edges, color)
    return edge_set


def make_wireframes_pose(tile_info_dict, color=[0.0, 0.0, 0.0]):
    """:returns list of transformed rectangles made of open3d.LineSet"""
    tile_wireframe_list = []
    for tile_info_key in tile_info_dict:
        tile_info = tile_info_dict[tile_info_key]
        tile_wireframe_list.append(make_tile_frame(trans_matrix=tile_info.pose_matrix,
                                                   width_by_mm=tile_info.width_by_mm,
                                                   height_by_mm=tile_info.height_by_mm,
                                                   color=color))
    return tile_wireframe_list


def make_wireframes_sensor(tile_info_dict, color=[0.0, 0.0, 0.0]):
    """:returns list of transformed rectangles made of open3d.LineSet"""
    tile_wireframe_list = []
    for tile_info_key in tile_info_dict:
        tile_info = tile_info_dict[tile_info_key]
        tile_wireframe_list.append(make_tile_frame(trans_matrix=tile_info.init_transform_matrix,
                                                   width_by_mm=tile_info.width_by_mm,
                                                   height_by_mm=tile_info.height_by_mm,
                                                   color=color))
    return tile_wireframe_list


# =======================================================================================


def make_tile_frame(trans_matrix=TileInfoDict.numpy.identity(4),
                    width_by_mm=4.0, height_by_mm=3.0, color=[0.5, 0.5, 0.5]):
    tile_frame = open3d.geometry.LineSet()
    lb_rb_rt_lt = [[0, -width_by_mm / 2, -height_by_mm / 2],
                   [0, width_by_mm / 2, -height_by_mm / 2],
                   [0, width_by_mm / 2, height_by_mm / 2],
                   [0, -width_by_mm / 2, height_by_mm / 2]
                   ]
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    colors = [color, color, color, color]

    tile_frame.points = open3d.utility.Vector3dVector(lb_rb_rt_lt)
    tile_frame.lines = open3d.utility.Vector2iVector(lines)
    tile_frame.colors = open3d.utility.Vector3dVector(colors)
    tile_frame.transform(trans_matrix)

    return tile_frame


def generate_pose_points_and_normals(tile_info_dict):
    """:returns points_and_normals = {"points": [],"normals": []}"""
    points_and_normals = {"points": [],
                          "normals": []}
    for tile_info_key in tile_info_dict:
        points_and_normals["points"].append(TileInfoDict.numpy.dot(tile_info_dict[tile_info_key].pose_matrix,
                                                                   TileInfoDict.numpy.asarray([0, 0, 0, 1]).T).T[0:3])
        points_and_normals["normals"].append(TileInfoDict.numpy.dot(tile_info_dict[tile_info_key].pose_matrix,
                                                                    TileInfoDict.numpy.asarray([1, 0, 0, 0]).T).T[0:3])
    return points_and_normals


def generate_sensor_points_and_normals(tile_info_dict):
    """:returns points_and_normals = {"points": [],"normals": []}"""
    points_and_normals = {"points": [],
                          "normals": []}
    for tile_info_key in tile_info_dict:
        points_and_normals["points"].append(TileInfoDict.numpy.dot(tile_info_dict[tile_info_key].init_transform_matrix,
                                                                   TileInfoDict.numpy.asarray([0, 0, 0, 1]).T).T[0:3])
        points_and_normals["normals"].append(TileInfoDict.numpy.dot(tile_info_dict[tile_info_key].init_transform_matrix,
                                                                    TileInfoDict.numpy.asarray([1, 0, 0, 0]).T).T[0:3])
    return points_and_normals


def generate_confirmed_edges(tile_info_dict):
    key_list = list(tile_info_dict.keys())
    edges = []
    for tile_info_key in tile_info_dict:
        for confirmed_neighbour_key in tile_info_dict[tile_info_key].confirmed_neighbour_list:
            edges.append([key_list.index(tile_info_key), key_list.index(confirmed_neighbour_key)])
    return edges


def generate_false_edges(tile_info_dict):
    key_list = list(tile_info_dict.keys())
    edges = []
    for tile_info_key in tile_info_dict:
        for potential_neighbour_key in tile_info_dict[tile_info_key].potential_neighbour_list:
            if potential_neighbour_key not in tile_info_dict[tile_info_key].confirmed_neighbour_list:
                edges.append([key_list.index(tile_info_key), key_list.index(potential_neighbour_key)])
    return edges

# ========================================================================================
#
# def visualize_tile_info_dict_as_point_cloud(tile_info_dict, downsample_factor=-1):
#     draw_list = []
#     draw_list += make_full_image_pcd_list_pose(tile_info_dict, downsample_factor)
#     draw_geometries(draw_list)
