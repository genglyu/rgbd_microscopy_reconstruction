import color_coded_triangle_mesh
import open3d

import os
import json
import argparse

from file_managing import *
import TileInfoDict
import VisualizerOpen3d


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
    # viewer = VisualizerOpen3d.MicroscopyReconstructionVisualizerOpen3d(tile_info_dict_all, config)
    # viewer.visualize_config()
    pcd, mesh = color_coded_triangle_mesh.generate_vertex_triangle_mesh_from_tile_info_dict(
        tile_info_dict=tile_info_dict_all,
        radii_list=[config["size_by_mm"][1] / 2,
                    config["potential_neighbour_searching_range"]/2,
                    config["potential_neighbour_searching_range"]])

    open3d.io.write_triangle_mesh(filename=join(config["path_data"], config["color_coded_triangle_mesh"]),
                                  mesh=mesh, write_ascii=True, compressed=False,
                                  write_vertex_normals=True,
                                  write_vertex_colors=True,
                                  write_triangle_uvs=True,
                                  print_progress=True)
    # open3d.visualization.draw_geometries([mesh])
    triangle_infos = color_coded_triangle_mesh.process_color_coded_ply(
        ply_path=join(config["path_data"], config["color_coded_triangle_mesh"]),
        tile_info_dict=tile_info_dict_all,
        path_data=config["path_data"],
        dataset_folder_template=config["dataset_folder_template"],
        path_image_dir=config["path_image_dir"],
        merged_texture_dir=config["merged_texture_dir"],
        merged_texture_file_name_template=config["merged_texture_file_name_template"]
    )

    color_coded_triangle_mesh.save_triangle_info_list(triangle_info_list=triangle_infos,
                                                      save_path=join(config["path_data"],
                                                                     config["textured_triangle_mesh_data"]))






