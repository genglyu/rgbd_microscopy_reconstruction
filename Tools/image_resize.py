import cv2
import file_managing
import os


def make_info_dict_for_group(group_id, config):
    print(group_id)
    print(config["dataset_folder_template"])
    print(config["dataset_folder_template"] % group_id)
    tile_info_directory_path = join(config["path_data"], config["dataset_folder_template"] % group_id,
                                    config["path_tile_info"])
    tile_image_directory_path = join(config["path_data"], config["dataset_folder_template"] % group_id,
                                    config["path_image_dir"])

    if os.path.isdir(tile_info_directory_path):
        robotic_pose_file_list = get_file_list(tile_info_directory_path, extension=".json")
    else:
        print(tile_info_directory_path + " is not a directory")
        return None

    # Start loading files and check the details.
    tile_info_dict_single_group = {}
    for robotic_pose_file_path in robotic_pose_file_list:

        tile_info = load_robotic_pose_info(group_id=group_id, robotic_pose_info_path=robotic_pose_file_path)

        # print(join(config["path_data"], config["path_image_dir"], tile_info.file_name) + ".png")
        image = cv2.imread(join(tile_image_directory_path, tile_info.file_name) + ".png")
        (h, w, c) = image.shape

        [tile_info.width_by_pixel, tile_info.height_by_pixel] = [w, h]
        [tile_info.width_by_mm, tile_info.height_by_mm] = config["size_by_mm"]
        # tile_info.pose_matrix = numpy.identity(4)
        tile_info_dict_single_group[tile_info.tile_index] = tile_info
    return tile_info_dict_single_group