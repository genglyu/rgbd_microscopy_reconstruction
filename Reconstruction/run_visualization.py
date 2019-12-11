import os
import json
import argparse
import time, datetime
from os.path import *

import sys

sys.path.append("./Utility")
sys.path.append("./Alignment")
sys.path.append("./Color_correction")
sys.path.append("./Data_processing")
sys.path.append("./Visualization")
sys.path.append("./Integration")

from file_managing import *
from copy import deepcopy
import TileInfoDict
import TransformationData
import PoseGraphG2o
import VisualizerOpen3d
import pose_graph_processing

# Visualize ==================================================================================
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

    viewer = VisualizerOpen3d.MicroscopyReconstructionVisualizerOpen3d(tile_info_dict_all, config)
    viewer.visualize_config()
