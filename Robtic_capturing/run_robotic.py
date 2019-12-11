import json
import argparse
import time, datetime
import file_managing
from os.path import *


import sys
sys.path.append("../../Utility")
sys.path.append("../Alignment")
sys.path.append("../Data_processing")
from open3d import *
from DataConvert import *
import numpy
import math
from scipy.spatial.transform import Rotation
import transforms3d



if __name__ == "__main__":
    # set_verbosity_level(verbosity_level=VerbosityLevel.Debug)
    parser = argparse.ArgumentParser(description="Curved surface microscopy level reconstruction")
    parser.add_argument("robotic_config", help="path to the robotic related config file")

    # parser.add_argument('--pose_list_make_all', '-d_make_all',
    #                     action="store_true",
    #                     help="Make pose_list in a single file, calculate the laplacian of images and save.")
    #
    # parser.add_argument('--align', '-a',
    #                     action="store_true",
    #                     help='Align the 3d rough reconstruction poses.')
    #
    # parser.add_argument('--interpolation', '-i',
    #                     action="store_true",
    #                     help='Interpolation through the aligned poses to generate the dense sampling points.')

    parser.add_argument('--crop_interpolated', '-c',
                        action="store_true",
                        help='Crop interpolated points to make sure no one is far away from the original points.')

    parser.add_argument('--navigation', '-n',
                        action="store_true",
                        help='Generate a relatively reasonable route through all the interpolated sampling points')

    parser.add_argument('--visualization', '-v',
                        action="store_true",
                        help='')

    args = parser.parse_args()

    # if not args.pose_list_make_all\
    #         and not args.align\
    #         and not args.interpolation \
    #         and not args.crop_interpolated \
    #         and not args.navigation \
    #         and not args.visualization:
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)
    if not args.crop_interpolated \
            and not args.navigation \
            and not args.visualization:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.robotic_config is not None:
        with open(args.robotic_config) as json_file:
            robotic_config = json.load(json_file)
        robotic_config["path_data"] = os.path.dirname(args.robotic_config)
    assert robotic_config is not None

    if args.crop_interpolated:
        import crop_interpolated

        trans_list_original = read_trans_list(join(robotic_config["path_data"],
                                                   robotic_config["robotic_reconstruction_workspace"],
                                                   robotic_config["robotic_reconstruction_trans_list_all"]))
        trans_list_interpolated = read_trans_list(join(robotic_config["path_data"],
                                                       robotic_config["robotic_reconstruction_workspace"],
                                                       robotic_config["robotic_reconstruction_trans_interpolated"]))

        trans_list_cropped = crop_interpolated.remove_out_range_trans(source_trans_list=trans_list_interpolated,
                                                                      reference_trans_list=trans_list_original,
                                                                      search_radius=0.01,
                                                                      amount_threholds=8,
                                                                      off_center_rate=0.5)
        save_trans_list(join(robotic_config["path_data"],
                             robotic_config["robotic_reconstruction_workspace"],
                             robotic_config["robotic_reconstruction_trans_interpolated_cropped"]),
                        trans_list_cropped)

    if args.navigation:
        import navigation
        try:
            trans_list = read_trans_list(join(robotic_config["path_data"],
                                              robotic_config["robotic_reconstruction_workspace"],
                                              robotic_config["robotic_reconstruction_trans_interpolated_cropped"]))
        except:
            trans_list = read_trans_list(join(robotic_config["path_data"],
                                              robotic_config["robotic_reconstruction_workspace"],
                                              robotic_config["robotic_reconstruction_trans_interpolated"]))

        navigator = navigation.NavigationGraph()

        # points = trans_list_to_points(trans_list_interpolated)
        navigator.load_trans_list(trans_list, 0.004)
        # order = navigator.dfs()
        # order = navigator.bfs(2)

        # points = adjust_order(points, order)
        # trans_list_ordered = adjust_order(trans_list_interpolated, order)
        trans_list_ordered = navigator.dfs()

        save_trans_list(join(robotic_config["path_data"],
                             robotic_config["robotic_reconstruction_workspace"],
                             robotic_config["robotic_reconstruction_trans_ordered"]),
                        trans_list_ordered)

        save_robotic_full_pose_list(join(robotic_config["path_data"],
                                         robotic_config["robotic_reconstruction_workspace"],
                                         robotic_config["robotic_reconstruction_robotic_trans_ordered"]),
                                    trans_list_to_pos_ori_list(trans_list_ordered))

    if args.visualization:
        import robotic_visualizer
        viewer = robotic_visualizer.RoboticVisualizerOpen3d(robotic_config)
        viewer.view_via_robotic_config()

    # Visualize ==================================================================================


