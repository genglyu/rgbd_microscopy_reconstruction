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


def update_local_trans_data_multiprocessing(transformation_data:TransformationDataPool, tile_info_dict, config,
                                            in_group=True):
    estimation_list = []
    if in_group:
        for tile_info_s_key in tile_info_dict:
            tile_info_s = tile_info_dict[tile_info_s_key]
            for tile_info_t_key in tile_info_s.potential_neighbour_list:
                if tile_info_s_key < tile_info_t_key:
                    estimation_list.append((tile_info_s_key, tile_info_t_key))
    else:
        for tile_info_s_key in tile_info_dict:
            tile_info_s = tile_info_dict[tile_info_s_key]
            for tile_info_t_key in tile_info_s.potential_neighbour_in_other_groups_list:
                if tile_info_s_key < tile_info_t_key:
                    estimation_list.append((tile_info_s_key, tile_info_t_key))

    max_thread = min(multiprocessing.cpu_count(), max(len(estimation_list), 1))

    # There might be an efficiency issue. the function self.get_trans(s, t)
    # It might be better if there is a independent function.
    # Currently, the multiprocessing does have multiple threads, but there is only one core working.
    # it is possible that some part of the class instance cannot be shared between cores.

    estimation_results = Parallel(n_jobs=max_thread)(
        delayed(trans_estimation_pure)(s_id=s_id, t_id=t_id,
                                       s_img_path=join(config["path_data"],
                                                       config["dataset_folder_template"] % (int(s_id / 10000)),
                                                       config["path_image_dir"],
                                                       tile_info_dict[s_id].file_name) + ".png",
                                       t_img_path=join(config["path_data"],
                                                       config["dataset_folder_template"] % (int(t_id / 10000)),
                                                       config["path_image_dir"],
                                                       tile_info_dict[t_id].file_name) + ".png",
                                       width_by_pixel_s=tile_info_dict[s_id].width_by_pixel,
                                       height_by_pixel_s=tile_info_dict[s_id].height_by_pixel,
                                       width_by_pixel_t=tile_info_dict[t_id].width_by_pixel,
                                       height_by_pixel_t=tile_info_dict[t_id].height_by_pixel,
                                       width_by_mm_s=tile_info_dict[s_id].width_by_mm,
                                       height_by_mm_s=tile_info_dict[s_id].height_by_mm,
                                       width_by_mm_t=tile_info_dict[t_id].width_by_mm,
                                       height_by_mm_t=tile_info_dict[t_id].height_by_mm,
                                       crop_w=config["crop_width_by_pixel"],
                                       crop_h=config["crop_height_by_pixel"],
                                       s_init_trans=tile_info_dict[s_id].init_transform_matrix,
                                       t_init_trans=tile_info_dict[t_id].init_transform_matrix,
                                       n_features=config["n_features"],
                                       num_matches_thresh1=config["num_matches_thresh1"],
                                       match_conf_threshold=config["conf_threshold"],
                                       scaling_tolerance=config["scaling_tolerance"],
                                       rotation_tolerance=config["rotation_tolerance"])
        for (s_id, t_id) in estimation_list)
    # Update the transformation data pool
    for trans_estimation in estimation_results:
        transformation_data.update_trans(trans_estimation)

    return transformation_data


def update_trans_info_dict_confirmed_neighbours(transformation_data:TransformationDataPool, tile_info_dict):
    for (s, t) in transformation_data.trans_dict:
        # print("s: %d, t: %d" % (s, t))
        # print(transformation_data.trans_dict[(s, t)].success)
        # print(t not in tile_info_dict[s].confirmed_neighbour_list)
        # since the later "make_pose_graph" read edges from LocalTransformationDataPool, so the confirmed neighbour
        # list can be all the tiles actually related rather than just the ones s < t.
        same_group = (int(s / 10000) == int(t / 10000))
        if transformation_data.trans_dict[(s, t)].success:
            if t not in tile_info_dict[s].confirmed_neighbour_list:
                if same_group:
                    tile_info_dict[s].confirmed_neighbour_list.append(t)
                else:
                    tile_info_dict[s].confirmed_neighbour_in_other_groups_list.append(t)

            if s not in tile_info_dict[t].confirmed_neighbour_list:
                if same_group:
                    tile_info_dict[t].confirmed_neighbour_list.append(s)
                else:
                    tile_info_dict[t].confirmed_neighbour_in_other_groups_list.append(s)
        else:
            while t in tile_info_dict[s].confirmed_neighbour_list:
                tile_info_dict[s].confirmed_neighbour_list.remove(t)
            while t in tile_info_dict[s].confirmed_neighbour_in_other_groups_list:
                tile_info_dict[s].confirmed_neighbour_in_other_groups_list.remove(t)
            while s in tile_info_dict[t].confirmed_neighbour_list:
                tile_info_dict[t].confirmed_neighbour_list.remove(s)
            while s in tile_info_dict[t].confirmed_neighbour_in_other_groups_list:
                tile_info_dict[t].confirmed_neighbour_in_other_groups_list.remove(s)
    return tile_info_dict
