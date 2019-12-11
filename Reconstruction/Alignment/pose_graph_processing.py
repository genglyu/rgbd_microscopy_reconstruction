import open3d
import g2o
import numpy
import cv2
from scipy.spatial.transform import Rotation
import transforms3d
import sys

sys.path.append("../Utility")
sys.path.append("../Reconstruction/Data_processing")
# from pose_estimation_cv import *
from TileInfo import *
from TileInfoDict import *
from TransformationData import *
from typing import Dict
from PoseGraphG2o import *
from local_transformation_estimation import *
import make_depth_map


def make_pose_graph_single_group(pose_graph,
                                 tile_info_dict,
                                 trans_data_manager: TransformationDataPool,
                                 config):
    print("Start making pose graph")
    # Add all the nodes (real nodes and virtual sensor nodes)
    for tile_info_key in tile_info_dict:
        tile_info = tile_info_dict[tile_info_key]
        pose_graph.add_node(id_outside=tile_info.tile_index,
                            trans=tile_info.init_transform_matrix,
                            sensor_info=trans_info_sensor(config["sensor_info_weight"],
                                                          numpy.asarray(config["sensor_info"])),
                            fixed=False)
    # Add read all the edges. Reading from a built TransformationDataPool
    for trans_estimation_result_key in trans_data_manager.trans_dict:
        trans_estimation_result = trans_data_manager.trans_dict[trans_estimation_result_key]
        if trans_estimation_result.success:
            pose_graph.add_matching_edge(s_id=trans_estimation_result.s, t_id=trans_estimation_result.t,
                                         trans=trans_estimation_result.trans,
                                         info=trans_info_matching(trans_estimation_result.conf,
                                                                  config["matching_info_weight"],
                                                                  numpy.asarray(config["match_info"])))
    # for tile_info_key in tile_info_dict:
    #     tile_info = tile_info_dict[tile_info_key]
    #     for neighbour_index in tile_info.confirmed_neighbour_list:
    #         success, conf, trans = \
    #             trans_data_manager.get_trans_extend(s_id=tile_info.tile_index, t_id=neighbour_index)
    #         if success:
    #             self.add_matching_edge(s_id=tile_info.tile_index, t_id=neighbour_index,
    #                                    trans=trans,
    #                                    info=trans_info_matching(conf, config["matching_info_weight"],
    #                                                             numpy.asarray(config["match_info_g2o"])))

#
# def make_pose_graph_fragment(pose_graph,
#                              trans_data_manager: TransformationDataPool,
#                              config):
#     print("Start making pose graph")
#     # Add all the nodes (real nodes and virtual sensor nodes)
#     for tile_info_key in tile_info_dict:
#         tile_info = tile_info_dict[tile_info_key]
#         pose_graph.add_node(id_outside=tile_info.tile_index,
#                             trans=tile_info.init_transform_matrix,
#                             sensor_info=trans_info_sensor(config["sensor_info_weight"],
#                                                           numpy.asarray(config["sensor_info"])),
#                             fixed=False)
#     # Add read all the edges. Reading from a built TransformationDataPool
#     for trans_estimation_result_key in trans_data_manager.trans_dict:
#         trans_estimation_result = trans_data_manager.trans_dict[trans_estimation_result_key]
#         if trans_estimation_result.success:
#             pose_graph.add_matching_edge(s_id=trans_estimation_result.s, t_id=trans_estimation_result.t,
#                                          trans=trans_estimation_result.trans,
#                                          info=trans_info_matching(trans_estimation_result.conf,
#                                                                   config["matching_info_weight"],
#                                                                   numpy.asarray(config["match_info"])))


def update_tile_info_dict_pose_in_group(pose_graph_single_group, tile_info_dict_single_group):
    for tile_info_key in tile_info_dict_single_group:
        tile_info = tile_info_dict_single_group[tile_info_key]
        tile_info.pose_matrix_in_group = pose_graph_single_group.get_pose(tile_info.tile_index)
        tile_info.pose_matrix = pose_graph_single_group.get_pose(tile_info.tile_index)
    return tile_info_dict_single_group


def update_tile_info_dict_pose(pose_graph_all, tile_info_dict_all):
    for tile_info_key in tile_info_dict_all:
        tile_info = tile_info_dict_all[tile_info_key]
        tile_info.pose_matrix = pose_graph_all.get_pose(tile_info.tile_index)
    return tile_info_dict_all
