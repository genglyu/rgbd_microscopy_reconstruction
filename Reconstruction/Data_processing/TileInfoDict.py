from TileInfo import *
import cv2
import sys
from open3d import *

sys.path.append("../Utility")
from file_managing import *


def tile_info_dict_generate_kd_tree(tile_info_dict, use_refined_data=False):
    tile_positions = []
    tile_index_list = []
    for tile_info_index in tile_info_dict:
        tile_info = tile_info_dict[tile_info_index]
        if use_refined_data:
            tile_positions.append(numpy.dot(tile_info.pose_matrix, numpy.array([0, 0, 0, 1]).T).T[0:3])
        else:
            tile_positions.append(tile_info.position)
        tile_index_list.append(tile_info_index)
    tile_info_point_cloud = PointCloud()
    tile_info_point_cloud.points = Vector3dVector(tile_positions)
    tile_tree = KDTreeFlann(tile_info_point_cloud)  # Building KDtree to help searching.

    return tile_tree, tile_index_list, tile_positions


def make_info_dict(config):
    tile_info_directory_path = join(config["path_data"], config["path_tile_info"])
    if os.path.isdir(tile_info_directory_path):
        robotic_pose_file_list = get_file_list(tile_info_directory_path, extension=".json")
    else:
        print(tile_info_directory_path + " is not a directory")
        return None
    # Usually there is no need to add the sorting function
    tile_info_dict = {}
    # tile_laplacian_list = []
    # tile_blur = 0

    # make all the tile info into the list.
    for robotic_pose_file_path in robotic_pose_file_list:

        tile_info = load_robotic_pose_info(robotic_pose_file_path)
        print(join(config["path_data"], config["path_image_dir"], tile_info.file_name) + ".png")
        image = cv2.imread(join(config["path_data"], config["path_image_dir"], tile_info.file_name)+ ".png")
        (h, w, c) = image.shape

        [tile_info.width_by_pixel, tile_info.height_by_pixel] = [w, h]
        [tile_info.width_by_m, tile_info.height_by_m] = config["size_by_m"]
        # tile_info.laplacian = cv2.Laplacian(image, cv2.CV_64F).var()
        tile_info.pose_matrix = numpy.identity(4)
        tile_info_dict[tile_info.tile_index] = tile_info

    # Make kd_tree ====================================================================
    tile_tree, tile_index_list, tile_positions = tile_info_dict_generate_kd_tree(tile_info_dict)
    # Generate potential neighbours ===================================================
    for k, tile_info_key in enumerate(tile_info_dict):
        tile_info = tile_info_dict[tile_info_key]
        [_, idx, _] = \
            tile_tree.search_radius_vector_3d(tile_info.position,
                                              config["potential_neighbour_searching_range"])
        n_idx = numpy.array(idx)

        for index_in_list in n_idx:
            if tile_index_list[index_in_list] > tile_info.tile_index:
                potential_adjacent_tile_index = tile_index_list[index_in_list]

                trans_target = tile_info.init_transform_matrix
                trans_source = tile_info_dict[potential_adjacent_tile_index].init_transform_matrix

                s_normal = numpy.dot(trans_source, numpy.asarray([1, 0, 0, 0]).T).T
                t_normal = numpy.dot(trans_target, numpy.asarray([1, 0, 0, 0]).T).T
                product = (t_normal * s_normal).sum()

                if product > config["normal_direction_tolerance_by_cos"]:
                    tile_info.potential_neighbour_list.append(potential_adjacent_tile_index)
    # Done.
    return tile_info_dict


def save_tile_info_dict(tile_info_dict_file_path, tile_info_dict):
    data_to_save = {}
    for tile_info_key in tile_info_dict:
        data_to_save[tile_info_key] = tile_info_to_json_format(tile_info_dict[tile_info_key])
    json.dump(data_to_save, open(tile_info_dict_file_path, "w"), indent=4)


def read_tile_info_dict(tile_info_dict_file_path):
    tile_info_dict_json_format = json.load(open(tile_info_dict_file_path, "r"))
    tile_info_dict = {}
    for tile_info_key_json_format in tile_info_dict_json_format:
        tile_info_dict[int(tile_info_key_json_format)] = \
            tile_info_from_json_format(tile_info_dict_json_format[tile_info_key_json_format])
    return tile_info_dict
