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

if __name__ == "__main__":
    # set_verbosity_level(verbosity_level=VerbosityLevel.Debug)

    parser = argparse.ArgumentParser(description="Curved surface microscopy level reconstruction")

    parser.add_argument("config", help="path to the config file")

    # tile_info_dict related
    parser.add_argument('--dict_make', '-d_make',
                        action="store_true",
                        help="Pick the clear image from tile_info_dict_all to make tile_info dict. "
                             "Find potential loop closures")
    parser.add_argument('--dict_load', '-d_load',
                        action="store_true",
                        help="Pick the clear image from tile_info_dict_all to make tile_info dict. "
                             "Find potential loop closures")

    # estimate all the local transformation involved. Information matrix have different format.
    parser.add_argument('--register', '-r',
                        action="store_true",
                        help="Register pose graph and generate local transformations without calculating information."
                             "Add confirmed loop closure to tile_info_dict members")
    parser.add_argument('--color_correction', '-c',
                        action="store_true",
                        help="Generate all color filter for global color correction")
    # make pose graph. Both pose graph and information matrix have different format.
    parser.add_argument('--make_pose_graph', '-m',
                        action="store_true",
                        help="make g2o pose graph and save.")
    parser.add_argument('--optimize', '-op',
                        action="store_true",
                        help="Optimize g2o pose graph, update the pose of each tile in tile_info_dict and save the "
                             "results.")

    parser.add_argument('--generate_depth_map', '-gdm',
                        action="store_true",
                        help="Make curved depth map for all tiles.")
    parser.add_argument('--integrate', '-i',
                        action="store_true",
                        help="Integrate images into a mesh.")

    # Visualize pose graph. Actually, raw and rough option should have same results.
    parser.add_argument('-visualization', '-v',
                        action="store_true",
                        help='Visualize via config file')

    args = parser.parse_args()

    if not args.dict_make \
            and not args.dict_load \
            and not args.register \
            and not args.color_correction \
            and not args.make_pose_graph \
            and not args.optimize \
            and not args.generate_depth_map \
            and not args.integrate \
            and not args.visualization:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.config is not None:
        with open(args.config) as json_file:
            config = json.load(json_file)
        config["path_data"] = os.path.dirname(args.config)
    assert config is not None

    # Prepare tile info dict ======================================================================
    if args.dict_make:
        tile_info_dict = TileInfoDict.make_info_dict(config)
        TileInfoDict.save_tile_info_dict(join(config["path_data"], config["tile_info_dict_name"]), tile_info_dict)

    if args.dict_load:
        dict_path = join(config["path_data"], config["tile_info_dict_name"])
        if os.path.isfile(dict_path):
            tile_info_dict = TileInfoDict.read_tile_info_dict(dict_path)
        else:
            print("No tile_info_dict available at default path")
            sys.exit()

    # Registering ================================================================================
    if args.register:
        import image_registration

        if tile_info_dict is not None:
            trans_data_manager = TransformationData.TransformationDataPool(tile_info_dict, config)
            try:
                trans_data_manager.read(join(config["path_data"], config["local_transformation_data_pool_name"]))
            except:
                print("No trans_data available in default path. Start updating")
                image_registration.update_local_trans_data_multiprocessing(transformation_data=trans_data_manager,
                                                                           tile_info_dict=tile_info_dict,
                                                                           config=config)
                trans_data_manager.save(join(config["path_data"], config["local_transformation_data_pool_name"]))
            tile_info_dict = image_registration.update_trans_info_dict_confirmed_neighbours(
                transformation_data=trans_data_manager, tile_info_dict=tile_info_dict)
            TileInfoDict.save_tile_info_dict(join(config["path_data"], config["tile_info_dict_name"]),
                                             tile_info_dict)

        else:
            print("No tile_info_dict available in RAM")
            sys.exit()

    if args.color_correction:
        # import global_color_correction_by_group
        # tile_info_dict = \
        #     global_color_correction_by_group.generate_color_filters(tile_info_dict=tile_info_dict,
        #                                                             trans_data_manager=trans_data_manager,
        #                                                             volum_size_by_m=0.005)

        # import global_color_correction_by_group_luminance
        # tile_info_dict = \
        #     global_color_correction_by_group_luminance.generate_color_filters(tile_info_dict=tile_info_dict,
        #                                                                       trans_data_manager=trans_data_manager,
        #                                                                       volum_size_by_m=0.001)

        import global_color_correction_luminance
        tile_info_dict = global_color_correction_luminance.generate_color_filters(tile_info_dict=tile_info_dict,
                                                                                  trans_data_manager=trans_data_manager)
        TileInfoDict.save_tile_info_dict(join(config["path_data"], config["tile_info_dict_name"]),
                                         tile_info_dict)

    # Make pose graph ============================================================================
    if args.make_pose_graph:
        pose_graph = PoseGraphG2o.PoseGraphOptimizerRoboticG2o()
        # pose_graph = pose_graph_processing.make_pose_graph(pose_graph, tile_info_dict, trans_data_manager, config)
        try:
            print("Trying to make pose graph")
            # print(tile_info_dict)
            # print(trans_data_manager)
            pose_graph_processing.make_pose_graph(pose_graph, tile_info_dict, trans_data_manager, config)
            print("Pose graph made")
            pose_graph.save(join(config["path_data"], config["rough_pose_graph_name"]))
        except:
            print("No tile_info_dict / trans_data_manager available in RAM or at default path")
            sys.exit()
            # tile_info_dict = pose_graph_g2o.update_tile_info_dict(tile_info_dict)

    if args.optimize:
        print("Start optimizing the pose graph =============================================")
        try:
            pose_graph.optimize(config["max_iterations"])
        except:
            print("No rough_pose_graph available in RAM. Try reading from default path")
            rough_posegraph_path = join(config["path_data"], config["rough_pose_graph_name"])
            if os.path.isfile(rough_posegraph_path):
                pose_graph = PoseGraphG2o.PoseGraphOptimizerRoboticG2o()
                pose_graph.load(rough_posegraph_path)
                pose_graph.optimize(config["max_iterations"])
            else:
                print("No rough_pose_graph available at default path")
                sys.exit()

        pose_graph.save(join(config["path_data"], config["optimized_pose_graph_name"]))

        tile_info_dict = pose_graph_processing.update_tile_info_dict(pose_graph, tile_info_dict)
        TileInfoDict.save_tile_info_dict(join(config["path_data"], config["tile_info_dict_name"]), tile_info_dict)

    # generate_depth_map ==================================================================================
    if args.generate_depth_map:
        import integrate_rgbd

        tile_info_dict = integrate_rgbd.generate_depth_map_multiprocessing(tile_info_dict=tile_info_dict, config=config)
        TileInfoDict.save_tile_info_dict(join(config["path_data"], config["tile_info_dict_name"]), tile_info_dict)

    # Integrate ==================================================================================
    if args.integrate:
        import integrate_rgbd

        integrate_rgbd.integrate_object(tile_info_dict=tile_info_dict, config=config, save_mesh=True)

    # Visualize ==================================================================================
    if args.visualization:
        viewer = VisualizerOpen3d.MicroscopyReconstructionVisualizerOpen3d(tile_info_dict, config)
        viewer.visualize_config()
