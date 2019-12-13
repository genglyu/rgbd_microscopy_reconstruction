import os
import json
import argparse
import time, datetime
from os.path import *

import sys

from file_managing import *
import TileInfoDict
import capture_virtual_tile
import color_coded_triangle_mesh
import file_managing
import similarity_evalutation

if __name__ == "__main__":
    # set_verbosity_level(verbosity_level=VerbosityLevel.Debug)
    parser = argparse.ArgumentParser(description="Curved surface microscopy level reconstruction")
    parser.add_argument("config", help="path to the config file")
    # Update tile_info_dict_all ===============================================================
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config) as json_file:
            config = json.load(json_file)
        config["path_data"] = os.path.dirname(args.config)
    assert config is not None

    # Prepare tile info dict. Individual groups and the entire dict. ============================================
    tile_info_dict_all = TileInfoDict.read_tile_info_dict(join(config["path_data"], config["tile_info_dict_name"]))
    triangle_infos = color_coded_triangle_mesh.read_triangle_info_list(
        file_managing.join(config["path_data"], config["textured_triangle_mesh_data"]))

    print("file_managing.join(config===============================================")
    print(file_managing.join(config["path_data"], config["recaptured_tile_dir"]))

    capture_virtual_tile.recapture_all_tiles_in_dict(
        tile_info_dict=tile_info_dict_all,
        triangle_info_list=triangle_infos,
        texture_dir=file_managing.join(config["path_data"], config["merged_texture_dir"]),
        width_pixel=config["microscope_intrinsic"]["width_by_pixel"],
        height_pixel=config["microscope_intrinsic"]["height_by_pixel"],
        tile_width=config["size_by_mm"][0],
        camera_focal_distance_by_mm=config["microscope_intrinsic"]["focal_distance_by_mm"],
        recaptured_tile_dir=file_managing.join(config["path_data"], config["recaptured_tile_dir"]),
        recaptured_tile_template=config["recaptured_tile_template"]
    )

    similarity_evalutation.evaluate_similarity(tile_info_dict=tile_info_dict_all,
                                               path_data=config["path_data"],
                                               dataset_folder_template=config["dataset_folder_template"],
                                               path_image_dir=config["path_image_dir"],
                                               recaptured_tile_dir=file_managing.join(config["path_data"],
                                                                                      config["recaptured_tile_dir"]),
                                               recaptured_tile_template=config["recaptured_tile_template"])


