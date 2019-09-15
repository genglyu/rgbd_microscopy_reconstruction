import open3d
import g2o
import numpy
import cv2
from scipy.spatial.transform import Rotation
import transforms3d
import sys
sys.path.append("../Utility")
sys.path.append("../Reconstruction/Data_processing")
from TileInfo import *
from TileInfoDict import *
from TransformationData import *
from file_managing import *
from local_transformation_estimation import *
import multiprocessing
from joblib import Parallel, delayed


def update_local_trans_data_multiprocessing(transformation_data:TransformationDataPool, tile_info_dict, config):
    estimation_list = []
    for tile_info_s_key in tile_info_dict:
        tile_info_s = tile_info_dict[tile_info_s_key]
        for tile_info_t_key in tile_info_s.potential_neighbour_list:
            estimation_list.append([tile_info_s_key, tile_info_t_key])
    max_thread = min(multiprocessing.cpu_count(), max(len(estimation_list), 1))

    # There might be an efficiency issue. the function self.get_trans(s, t)
    # It might be better if there is a independent function.
    # Currently, the multiprocessing does have multiple threads, but there is only one core working.
    # it is possible that some part of the class instance cannot be shared between cores.

    estimation_results = Parallel(n_jobs=max_thread)(
        delayed(trans_estimation_pure)(s_id=s_id, t_id=t_id,
                                       s_img_path=join(config["path_data"],
                                                       config["path_image_dir"],
                                                       tile_info_dict[s_id].file_name, ".png"),
                                       t_img_path=join(config["path_data"],
                                                       config["path_image_dir"],
                                                       tile_info_dict[t_id].file_name, ".png"),
                                       width_by_pixel_s=tile_info_dict[s_id].width_by_pixel,
                                       height_by_pixel_s=tile_info_dict[s_id].height_by_pixel,
                                       width_by_pixel_t=tile_info_dict[t_id].width_by_pixel,
                                       height_by_pixel_t=tile_info_dict[t_id].height_by_pixel,
                                       width_by_m_s=tile_info_dict[s_id].width_by_m,
                                       height_by_m_s=tile_info_dict[s_id].height_by_m,
                                       width_by_m_t=tile_info_dict[t_id].width_by_m,
                                       height_by_m_t=tile_info_dict[t_id].height_by_m,
                                       crop_w=config["crop_width_by_pixel"],
                                       crop_h=config["crop_height_by_pixel"],
                                       s_init_trans=tile_info_dict[s_id].init_transform_matrix,
                                       t_init_trans=tile_info_dict[t_id].init_transform_matrix,
                                       n_features=config["n_features"],
                                       num_matches_thresh1=config["num_matches_thresh1"],
                                       match_conf_threshold=config["conf_threshold"],
                                       scaling_tolerance=config["scaling_tolerance"],
                                       rotation_tolerance=config["rotation_tolerance"])
        for [s_id, t_id] in estimation_list)
    # Update the transformation data pool
    for trans_estimation in estimation_results:
        transformation_data.update_trans(trans_estimation)

    return transformation_data


def update_trans_info_dict_confirmed_neighbours():
    return True
