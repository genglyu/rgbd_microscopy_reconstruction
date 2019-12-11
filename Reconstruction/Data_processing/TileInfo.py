import numpy
import os
import json
from DataConvert import *


class TileInfo:
    def __init__(self,
                 tile_index=None,
                 tile_group_id=0,
                 tile_index_in_group=0,
                 file_name="",
                 init_transform_matrix=numpy.identity(4)
                 ):
        if tile_index is None:
            self.tile_group = tile_group_id
            self.tile_index = tile_group_id * 10000 + tile_index_in_group
        else:
            self.tile_index = tile_index
            self.tile_group = int(tile_index / 10000)

        self.file_name = file_name
        # The extension: these three different files share same names.
        # tile_info:    .json
        # rgb_image:    .png
        # depth_image:  .png
        # cropped_pcd:  .ply
        self.init_transform_matrix = init_transform_matrix
        self.pose_matrix_in_group = init_transform_matrix
        self.pose_matrix = init_transform_matrix

        self.rgbd_camera_pose_matrix = numpy.identity(4)

        self.color_and_illumination_correction = numpy.array([1.0, 1.0, 1.0])  # RGB space.

        self.position = numpy.array([init_transform_matrix[0][3],
                                     init_transform_matrix[1][3],
                                     init_transform_matrix[2][3]])

        self.width_by_pixel = 1280
        self.height_by_pixel = 720
        self.width_by_mm = 8.8
        self.height_by_mm = 4.95

        # self.laplacian = 0.0
        self.has_april_tag = False
        self.april_tags = []
        self.trans_from_april_tag = {}

        self.potential_neighbour_list = []
        self.confirmed_neighbour_list = []
        self.potential_neighbour_in_other_groups_list = []
        self.confirmed_neighbour_in_other_groups_list = []


def load_robotic_pose_info(group_id, robotic_pose_info_path):
    if os.path.isfile(robotic_pose_info_path):
        robotic_pose_data = json.load(open(robotic_pose_info_path, "r"))
        trans = rob_pose_to_img_trans(robotic_pose_data)
        # Convert from m to mm. Only translation needs to be adjusted.
        trans[0][3] = trans[0][3] * 1000
        trans[1][3] = trans[1][3] * 1000
        trans[2][3] = trans[2][3] * 1000

        file_name, extension = os.path.splitext(os.path.basename(robotic_pose_info_path))
        tile_index_in_group = int(file_name[len("tile_00"):])

        new_tile_info = TileInfo(
            tile_group_id=group_id,
            tile_index_in_group=tile_index_in_group,
            file_name=file_name,
            init_transform_matrix=trans
        )
        return new_tile_info
    else:
        print(robotic_pose_info_path + " is not a file")
        return None


def tile_info_to_json_format(tile_info: TileInfo):
    if tile_info.has_april_tag:
        april_tags_data = []
        trans_from_april_tag_data = {}
        for tag in tile_info.april_tags:
            tag_data = {'hamming': tag['hamming'],
                        'margin': tag["margin"],
                        'id': tag["id"],
                        'center': tag["center"].tolist(),
                        'lb-rb-rt-lt': tag["lb-rb-rt-lt"].tolist()}
            april_tags_data.append(tag_data)
        for trans_from_april_tag_id in tile_info.trans_from_april_tag:
            trans_from_april_tag_data[trans_from_april_tag_id] = \
                tile_info.trans_from_april_tag[trans_from_april_tag_id].tolist()
    else:
        april_tags_data = []
        trans_from_april_tag_data = {}

    tile_info_json_format = {"tile_group": tile_info.tile_group,
                             "tile_index": tile_info.tile_index,
                             "file_name": tile_info.file_name,

                             "init_transform_matrix": tile_info.init_transform_matrix.tolist(),
                             "pose_matrix_in_group": tile_info.pose_matrix_in_group.tolist(),
                             "pose_matrix": tile_info.pose_matrix.tolist(),

                             "rgbd_camera_pose_matrix": tile_info.rgbd_camera_pose_matrix.tolist(),

                             "color_and_illumination_correction": tile_info.color_and_illumination_correction.tolist(),

                             "position": tile_info.position.tolist(),

                             "width_by_pixel": tile_info.width_by_pixel,
                             "height_by_pixel": tile_info.height_by_pixel,
                             "width_by_mm": tile_info.width_by_mm,
                             "height_by_mm": tile_info.height_by_mm,

                             # "laplacian": tile_info.laplacian,
                             "has_april_tag": tile_info.has_april_tag,
                             "april_tags": april_tags_data,
                             "trans_from_april_tag": trans_from_april_tag_data,

                             "potential_neighbour_list": tile_info.potential_neighbour_list,
                             "confirmed_neighbour_list": tile_info.confirmed_neighbour_list,
                             "potential_neighbour_in_other_groups_list": tile_info.potential_neighbour_in_other_groups_list,
                             "confirmed_neighbour_in_other_groups_list": tile_info.confirmed_neighbour_in_other_groups_list}
    return tile_info_json_format


def tile_info_from_json_format(tile_info_json_format):
    new_tile_info = TileInfo()

    new_tile_info.tile_group = int(tile_info_json_format["tile_group"])
    new_tile_info.tile_index = int(tile_info_json_format["tile_index"])

    new_tile_info.file_name = tile_info_json_format["file_name"]
    new_tile_info.init_transform_matrix = numpy.asarray(tile_info_json_format["init_transform_matrix"])

    new_tile_info.pose_matrix_in_group = numpy.asarray(tile_info_json_format["pose_matrix_in_group"])
    new_tile_info.pose_matrix = numpy.asarray(tile_info_json_format["pose_matrix"])

    new_tile_info.rgbd_camera_pose_matrix = numpy.asarray(tile_info_json_format["rgbd_camera_pose_matrix"])
    new_tile_info.color_and_illumination_correction = numpy.asarray(
        tile_info_json_format["color_and_illumination_correction"])

    new_tile_info.position = numpy.asarray(tile_info_json_format["position"])

    new_tile_info.width_by_mm = tile_info_json_format["width_by_mm"]
    new_tile_info.height_by_mm = tile_info_json_format["height_by_mm"]
    new_tile_info.width_by_pixel = int(tile_info_json_format["width_by_pixel"])
    new_tile_info.height_by_pixel = int(tile_info_json_format["height_by_pixel"])

    new_tile_info.has_april_tag = bool(tile_info_json_format["has_april_tag"])
    for tag_data in tile_info_json_format["april_tags"]:
        tag = {"hamming": int(tag_data["hamming"]),
               "margin": float(tag_data["margin"]),
               "id": int(tag_data["id"]),
               "center": numpy.asarray(tag_data["center"]),
               "lb-rb-rt-lt": numpy.asarray(tag_data["lb-rb-rt-lt"])
               }
        new_tile_info.april_tags.append(tag)
    for trans_from_april_tag_id in tile_info_json_format["trans_from_april_tag"]:
        new_tile_info.trans_from_april_tag[int(trans_from_april_tag_id)] = \
            numpy.asarray(tile_info_json_format["trans_from_april_tag"][trans_from_april_tag_id])

    new_tile_info.potential_neighbour_list = tile_info_json_format["potential_neighbour_list"]
    new_tile_info.confirmed_neighbour_list = tile_info_json_format["confirmed_neighbour_list"]

    new_tile_info.potential_neighbour_in_other_groups_list = tile_info_json_format["potential_neighbour_in_other_groups_list"]
    new_tile_info.confirmed_neighbour_in_other_groups_list = tile_info_json_format["confirmed_neighbour_in_other_groups_list"]
    return new_tile_info
