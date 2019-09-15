import numpy
import os
import json
from DataConvert import *


class TileInfo:
    def __init__(self,
                 tile_index,
                 file_name,
                 init_transform_matrix=numpy.identity(4)
                 ):
        self.tile_index = tile_index
        self.file_name = file_name
        # The extension: these three different files share same names.
        # tile_info:    .json
        # rgb_image:    .png
        # depth_image:  .png
        self.init_transform_matrix = init_transform_matrix
        self.pose_matrix = numpy.identity(4)
        self.rgbd_camera_pose_matrix = numpy.identity(4)

        self.position = numpy.array([init_transform_matrix[0][3],
                                     init_transform_matrix[1][3],
                                     init_transform_matrix[2][3]])

        self.width_by_pixel = 640
        self.height_by_pixel = 480
        self.width_by_m = 0.0064
        self.height_by_m = 0.0048

        # self.laplacian = 0.0
        self.has_april_tag = False
        self.april_tags = []
        self.trans_from_april_tag = {}

        self.potential_neighbour_list = []
        self.confirmed_neighbour_list = []


def load_robotic_pose_info(robotic_pose_info_path):
    if os.path.isfile(robotic_pose_info_path):
        robotic_pose_data = json.load(open(robotic_pose_info_path, "r"))
        trans = rob_pose_to_trans(robotic_pose_data)

        file_name, extension = os.path.splitext(os.path.basename(robotic_pose_info_path))
        tile_index = int(file_name[len("tile_"):])

        new_tile_info = TileInfo(
            tile_index=tile_index,
            file_name=file_name,
            init_transform_matrix=trans
        )
        return new_tile_info
    else:
        print(robotic_pose_info_path + " is not a file")
        return None


def tile_info_from_json_format(tile_info_json_format):
    new_tile_info = TileInfo(
        tile_index=int(tile_info_json_format["tile_index"]),
        file_name=tile_info_json_format["file_name"],
        init_transform_matrix=numpy.asarray(tile_info_json_format["init_transform_matrix"]))

    new_tile_info.pose_matrix = numpy.asarray(tile_info_json_format["pose_matrix"])
    new_tile_info.rgbd_camera_pose_matrix = numpy.asarray(tile_info_json_format["rgbd_camera_pose_matrix"])

    new_tile_info.position = numpy.asarray(tile_info_json_format["position"])

    new_tile_info.width_by_m = tile_info_json_format["width_by_m"]
    new_tile_info.height_by_m = tile_info_json_format["height_by_m"]
    new_tile_info.width_by_pixel = int(tile_info_json_format["width_by_pixel"])
    new_tile_info.height_by_pixel = int(tile_info_json_format["height_by_pixel"])
    # new_tile_info.laplacian = float(tile_info_json_format["laplacian"])
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

    tile_info_json_format = {"tile_index": tile_info.tile_index,
                             "file_name": tile_info.file_name,

                             "init_transform_matrix": tile_info.init_transform_matrix.tolist(),
                             "pose_matrix": tile_info.pose_matrix.tolist(),
                             "rgbd_camera_pose_matrix": tile_info.rgbd_camera_pose_matrix.tolist(),

                             "position": tile_info.position.tolist(),

                             "width_by_pixel": tile_info.width_by_pixel,
                             "height_by_pixel": tile_info.height_by_pixel,
                             "width_by_m": tile_info.width_by_m,
                             "height_by_m": tile_info.height_by_m,

                             # "laplacian": tile_info.laplacian,
                             "has_april_tag": tile_info.has_april_tag,
                             "april_tags": april_tags_data,
                             "trans_from_april_tag": trans_from_april_tag_data,

                             "potential_neighbour_list": tile_info.potential_neighbour_list,
                             "confirmed_neighbour_list": tile_info.confirmed_neighbour_list}
    return tile_info_json_format
