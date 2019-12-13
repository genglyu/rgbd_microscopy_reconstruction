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

    # Processing in each individual groups ===============================================================
    # estimate all the local transformation involved. Information matrix have different format.
    parser.add_argument('--register_in_group', '-r',
                        action="store_true",
                        help="Register pose graph and generate local transformations without calculating information."
                             "Add confirmed loop closure to tile_info_dict members")
    # make pose graph. Both pose graph and information matrix have different format.
    parser.add_argument('--make_pose_graph_in_group', '-m',
                        action="store_true",
                        help="make g2o pose graph and save.")
    parser.add_argument('--optimize_in_group', '-op',
                        action="store_true",
                        help="Optimize g2o pose graph, update the pose of each tile in tile_info_dict and save the "
                             "results.")
    parser.add_argument('--make_fragments', '-mf',
                        action="store_true",
                        help="Make fragments from all groups for registering between groups.")
    # Matching between fragments ===============================================================
    # parser.add_argument('--register_fragment', '-rg',
    #                     action="store_true",
    #                     help="Estimate the transformation between all fragments.")
    # parser.add_argument('--make_pose_graph_fragment', '-rg',
    #                     action="store_true",
    #                     help="Make pose_graph used for fragment transformation refine.")
    # parser.add_argument('--optimize_fragment', '-opf',
    #                     action="store_true",
    #                     help="Refine the fragments transformations.")
    # Update tile_info_dict_all ===============================================================
    parser.add_argument('--update_all', '-u_all',
                        action="store_true",
                        help="Update tile_info_dict_all to generate the neighbours from different groups .")
    parser.add_argument('--register_all', '-r_all',
                        action="store_true",
                        help="Make new TransformationData.")
    parser.add_argument('--make_pose_graph_all', '-m_all',
                        action="store_true",
                        help="Make new TransformationData.")
    parser.add_argument('--optimize_all', '-op_all',
                        action="store_true",
                        help="Optimize g2o pose graph, update the pose of each tile in tile_info_dict and save the "
                             "results.")

    parser.add_argument('--color_correction', '-c',
                        action="store_true",
                        help="Generate all color filter for global color correction")

    # Visualize pose graph. Actually, raw and rough option should have same results.
    parser.add_argument('-visualization', '-v',
                        action="store_true",
                        help='Visualize via config file')

    args = parser.parse_args()

    # if not args.dict_make \
    #         and not args.dict_load \
    #         and not args.register_groups \
    #         and not args.color_correction \
    #         and not args.make_pose_graph_group \
    #         and not args.optimize_group \
    #         and not args.integrate_group \
    #         and not args.visualization:
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)

    if args.config is not None:
        with open(args.config) as json_file:
            config = json.load(json_file)
        config["path_data"] = os.path.dirname(args.config)
    assert config is not None

    dataset_group_ids = config["group_folder_ids"]
    print(dataset_group_ids)

    tile_info_dict_all = {}
    tile_info_dict_groups = {}

    trans_data_manager_all = None
    trans_data_manager_groups = {}

    pose_graph_all = None
    pose_graph_groups = {}

    fragments_pcds = {}
    fragments_meshes = {}

    # Prepare tile info dict. Individual groups and the entire dict. ============================================
    if args.dict_make:
        for group_id in dataset_group_ids:
            group_id = int(group_id)
            print("Generate group_tile_dict %02d" % group_id)
            tile_info_dict_single_group = TileInfoDict.make_info_dict_for_group(group_id=group_id, config=config)
            # Generate potential neighbours in each group.
            TileInfoDict.update_potential_neighbour_list_in_group(
                tile_info_dict_single_group=tile_info_dict_single_group,
                potential_neighbour_searching_range=config["potential_neighbour_searching_range"],
                normal_direction_tolerance_by_cos=config["normal_direction_tolerance_by_cos"])
            # Save a copy
            TileInfoDict.save_tile_info_dict(join(config["path_data"],
                                                  config["dataset_folder_template"] % group_id,
                                                  config["tile_info_dict_name"]),
                                             tile_info_dict_single_group)

            # Update the result to the tile_info_dict_groups
            tile_info_dict_groups[group_id] = tile_info_dict_single_group

        # ===============================================================================
        # Saving the tile_info_dict_all
        for group_id in dataset_group_ids:
            tile_info_dict_all.update(tile_info_dict_groups[group_id])
        TileInfoDict.save_tile_info_dict(join(config["path_data"],
                                              config["tile_info_dict_name"]),
                                         tile_info_dict_all)
        # ===============================================================================
    if args.dict_load:
        # The individual dicts for groups
        for group_id in dataset_group_ids:
            dict_single_group_path = join(config["path_data"],
                                          config["dataset_folder_template"] % group_id,
                                          config["tile_info_dict_name"])
            if os.path.isfile(dict_single_group_path):
                tile_info_dict_single_group = TileInfoDict.read_tile_info_dict(dict_single_group_path)
                tile_info_dict_groups[group_id] = tile_info_dict_single_group
            else:
                print("No tile_info_dict available at default path")
                sys.exit()
        # The dict contains all tile info
        dict_all_path = join(config["path_data"], config["tile_info_dict_name"])
        if os.path.isfile(dict_all_path):
            tile_info_dict_all = TileInfoDict.read_tile_info_dict(dict_all_path)
        else:
            print("No tile_info_dict_all available at default path")
            sys.exit()

    # Register_in_group ================================================================================
    if args.register_in_group:
        import image_registration
        for group_id in tile_info_dict_groups:
            print("Processing group %02d" % group_id)
            tile_info_dict_single_group = tile_info_dict_groups[group_id]
            trans_data_manager_single_group = \
                TransformationData.TransformationDataPool(tile_info_dict=tile_info_dict_single_group,
                                                          config=config)
            try:
                trans_data_manager_single_group.read(join(config["path_data"],
                                                          config["dataset_folder_template"] % group_id,
                                                          config["local_transformation_data_pool_name"]))
            except:
                print("No trans_data available in default path. Start updating")
                image_registration.update_local_trans_data_multiprocessing(
                    transformation_data=trans_data_manager_single_group,
                    tile_info_dict=tile_info_dict_single_group,
                    config=config)
                trans_data_manager_single_group.save(join(config["path_data"],
                                                          config["dataset_folder_template"] % group_id,
                                                          config["local_transformation_data_pool_name"]))
            # Update the tile_info_dict in each group.
            print("Updating the confirmed_neighbour_list of " + config["dataset_folder_template"] % group_id)
            tile_info_dict_single_group = image_registration.update_trans_info_dict_confirmed_neighbours(
                transformation_data=trans_data_manager_single_group, tile_info_dict=tile_info_dict_single_group)

            TileInfoDict.save_tile_info_dict(join(config["path_data"],
                                                  config["dataset_folder_template"] % group_id,
                                                  config["tile_info_dict_name"]),
                                             tile_info_dict_single_group)
            # Add to the trans inside groups dict
            trans_data_manager_groups[group_id] = trans_data_manager_single_group
        # Update to the large data pool
        trans_data_manager_all = TransformationData.TransformationDataPool(tile_info_dict=tile_info_dict_all,
                                                                           config=config)
        try:
            trans_data_manager_all.read(join(config["path_data"],
                                             config["local_transformation_data_pool_name"]))
        except:
            for group_id in tile_info_dict_groups:
                trans_data_manager_all.update_from_other_trans_data_pool(trans_data_manager_groups[group_id])
            trans_data_manager_all.save(join(config["path_data"],
                                             config["local_transformation_data_pool_name"]))

        # ===============================================================================
        # Saving the tile_info_dict_all. The neighbour list might change in this progress
        print(tile_info_dict_groups)
        for group_id in dataset_group_ids:
            tile_info_dict_all.update(tile_info_dict_groups[group_id])
        TileInfoDict.save_tile_info_dict(join(config["path_data"],
                                              config["tile_info_dict_name"]),
                                         tile_info_dict_all)
        # ===============================================================================

    # Make_pose_graph_in_group ============================================================================
    if args.make_pose_graph_in_group:
        for group_id in tile_info_dict_groups:
            pose_graph_single_group = PoseGraphG2o.PoseGraphOptimizerTileG2o()
            # pose_graph = pose_graph_processing.make_pose_graph(pose_graph, tile_info_dict, trans_data_manager, config)
            try:
                print("Trying to make pose graph")
                # print(tile_info_dict)
                # print(trans_data_manager)
                pose_graph_processing.make_pose_graph_single_group(pose_graph=pose_graph_single_group,
                                                                   tile_info_dict=tile_info_dict_groups[group_id],
                                                                   trans_data_manager=trans_data_manager_groups[group_id],
                                                                   config=config)
                print("Pose graph made")
                pose_graph_single_group.save(join(config["path_data"],
                                                  config["dataset_folder_template"] % group_id,
                                                  config["rough_pose_graph_name"]))
            except:
                print("No tile_info_dict / trans_data_manager available in RAM or at default path")
                sys.exit()
                # tile_info_dict = pose_graph_g2o.update_tile_info_dict(tile_info_dict)

    if args.optimize_in_group:
        print("Start optimizing the pose graph =============================================")
        for group_id in tile_info_dict_groups:
            try:
                pose_graph_single_group = pose_graph_groups[group_id]
                pose_graph_single_group.optimize(config["max_iterations"])
            except:
                print("No rough_pose_graph available in RAM. Try reading from default path")
                rough_pose_graph_path = join(config["path_data"],
                                             config["dataset_folder_template"] % group_id,
                                             config["rough_pose_graph_name"])
                if os.path.isfile(rough_pose_graph_path):
                    pose_graph_single_group = PoseGraphG2o.PoseGraphOptimizerTileG2o()
                    pose_graph_single_group.load(rough_pose_graph_path)
                    pose_graph_single_group.optimize(config["max_iterations"])
                else:
                    print("No rough_pose_graph available at default path")
                    sys.exit()
            pose_graph_single_group.save(join(config["path_data"],
                                              config["dataset_folder_template"] % group_id,
                                              config["optimized_pose_graph_name"]))
            # Update the optimized result to tile_info_dict of each group.
            tile_info_dict_single_group = tile_info_dict_groups[group_id]
            tile_info_dict_single_group = \
                pose_graph_processing.update_tile_info_dict_pose_in_group(pose_graph_single_group,
                                                                          tile_info_dict_single_group)
            TileInfoDict.save_tile_info_dict(join(config["path_data"],
                                                  config["dataset_folder_template"] % group_id,
                                                  config["tile_info_dict_name"]),
                                             tile_info_dict_single_group)
        # ===============================================================================
        # Saving the tile_info_dict_all. The pose_matrix might change in this progress.
        for group_id in dataset_group_ids:
            tile_info_dict_all.update(tile_info_dict_groups[group_id])
        TileInfoDict.save_tile_info_dict(join(config["path_data"],
                                              config["tile_info_dict_name"]),
                                         tile_info_dict_all)
        # ===============================================================================

    # Generate fragments from all the groups for transformation estimation
    if args.make_fragments:
        touch_folder(join(config["path_data"], config["path_fragments_dir"]))
        import make_fragments
        import color_correction_seperate_channels
        import open3d
        for group_id in tile_info_dict_groups:
            print("Making fragment from " + config["dataset_folder_template"] % group_id)

            tile_info_dict_groups[group_id] = \
                color_correction_seperate_channels.generate_color_filters(
                    tile_info_dict=tile_info_dict_groups[group_id],
                    trans_data_manager=trans_data_manager_groups[group_id])

            TileInfoDict.save_tile_info_dict(join(config["path_data"],
                                                  config["dataset_folder_template"] % group_id,
                                                  config["tile_info_dict_name"]),
                                             tile_info_dict_groups[group_id])

            fragment_pcd = make_fragments.make_fragment_pcd_for_single_group(
                tile_info_dict_single_group=tile_info_dict_groups[group_id],
                img_directory_path=join(config["path_data"],
                                        config["dataset_folder_template"] % group_id,
                                        config["path_image_dir"]),
                voxel_size=config["fragments_voxel_size"])

            fragments_pcds[group_id] = fragment_pcd
            make_fragments.save_fragment_pcd(fragment=fragment_pcd,
                                             save_path=join(config["path_data"],
                                                        config["path_fragments_dir"],
                                                        config["fragment_template"] % group_id))

            # fragments_mesh, _ = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(fragment_pcd,
            #                                                                                  depth=15,
            #                                                                                  linear_fit=True)
            # fragments_meshes[group_id] = fragments_mesh
        import open3d
        for group_id in tile_info_dict_groups:
            open3d.visualization.draw_geometries([fragments_pcds[group_id]])
            # open3d.visualization.draw_geometries([fragments_meshes[group_id]])


    # Estimate the transformation between group fragments
    # if args.register_fragment:
    #     for group_id in dataset_group_ids:
    #         fragments[group_id] = make_fragments.read_fragment(save_path=join(config["path_data"],
    #                                                                           config["path_fragments_dir"],
    #                                                                           config["fragment_template"] % group_id))
    #     for group_id in




    # # generate_depth_map ==================================================================================
    # if args.generate_depth_map:
    #     import integrate_rgbd
    #
    #     tile_info_dict = integrate_rgbd.generate_depth_map_multiprocessing(tile_info_dict=tile_info_dict, config=config)
    #     TileInfoDict.save_tile_info_dict(join(config["path_data"], config["tile_info_dict_name"]), tile_info_dict)
    #
    # # Integrate ==================================================================================
    # if args.integrate:
    #     import integrate_rgbd
    #
    #     integrate_rgbd.integrate_object(tile_info_dict=tile_info_dict, config=config, save_mesh=True)
    # Color correction ==================================================================================
    if args.color_correction:
        import global_color_correction_luminance
        dict_all_path = join(config["path_data"], config["tile_info_dict_name"])
        if os.path.isfile(dict_all_path):
            tile_info_dict_all = TileInfoDict.read_tile_info_dict(dict_all_path)
        else:
            print("No tile_info_dict_all available at default path")
            sys.exit()

        trans_data_manager_all = TransformationData.TransformationDataPool(tile_info_dict=tile_info_dict_all,
                                                                           config=config)
        try:
            trans_data_manager_all.read(join(config["path_data"],
                                             config["local_transformation_data_pool_name"]))
        except:
            for group_id in tile_info_dict_groups:
                trans_data_manager_all.update_from_other_trans_data_pool(trans_data_manager_groups[group_id])
            trans_data_manager_all.save(join(config["path_data"],
                                             config["local_transformation_data_pool_name"]))
        # generate_color_filter ========================================================================
        tile_info_dict_all = global_color_correction_luminance.generate_color_filters(
            tile_info_dict=tile_info_dict_all,
            trans_data_manager=trans_data_manager_all)
        TileInfoDict.save_tile_info_dict(join(config["path_data"], config["tile_info_dict_name"]),
                                         tile_info_dict_all)

