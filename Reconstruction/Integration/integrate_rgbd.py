import sys
from make_depth_map import *
import multiprocessing
from joblib import Parallel, delayed
import sys

sys.path.append("../../Utility")
from file_managing import *
from open3d import *


def generate_depth_map_multiprocessing(tile_info_dict, config):
    touch_folder(join(config["path_data"], config["path_depth_dir"]))

    tile_tree, tile_index_list, tile_positions = tile_info_dict_generate_kd_tree(tile_info_dict, use_refined_data=True)
    # print(config["microscope_intrinsic"])
    microcsope_intrinsic = \
        generate_microscope_intrinsic_open3d(width_by_pixel=config["microscope_intrinsic"]["width_by_pixel"],
                                             height_by_pixel=config["microscope_intrinsic"]["height_by_pixel"],
                                             width_by_mm=config["size_by_mm"][0],
                                             height_by_mm=config["size_by_mm"][1],
                                             focal_distance_by_mm=config["microscope_intrinsic"]["focal_distance"])

    # max_thread = min(multiprocessing.cpu_count(), max(len(tile_info_dict), 1))
    # Parallel(n_jobs=max_thread)(
    #     delayed(make_single_depth_image)(tile_trans=tile_info_dict[tile_info_key].pose_matrix,
    #                                      all_tiles_center_position_list=tile_positions,
    #                                      all_tiles_center_position_kd_tree=tile_tree,
    #                                      searching_range_radius=tile_info_dict[tile_info_key].width_by_m * 2,
    #                                      depth_camera_intrinsic=microcsope_intrinsic,
    #                                      focal_distance_by_m=config["microscope_intrinsic"]["focal_distance"],
    #                                      depth_scaling_factor=config["microscope_intrinsic"]["depth_scaling"],
    #                                      saving_path=join(config["path_data"], config["path_depth_dir"],
    #                                                       tile_info_dict[tile_info_key].file_name) + ".png")
    #     for tile_info_key in tile_info_dict)

    for tile_info_key in tile_info_dict:
        print("generating depth map for tile %6d" % tile_info_key)
        make_single_depth_image(tile_trans=tile_info_dict[tile_info_key].pose_matrix,
                                all_tiles_center_position_list=tile_positions,
                                all_tiles_center_position_kd_tree=tile_tree,
                                searching_range_radius=tile_info_dict[tile_info_key].width_by_m * 2,
                                depth_camera_intrinsic=microcsope_intrinsic,
                                focal_distance_by_mm=config["microscope_intrinsic"]["focal_distance"],
                                depth_scaling_factor=config["microscope_intrinsic"]["depth_scaling"],
                                saving_path=join(config["path_data"], config["path_depth_dir"],
                                                 tile_info_dict[tile_info_key].file_name) + ".png")


    for tile_info_key in tile_info_dict:
        tile_info = tile_info_dict[tile_info_key]
        tile_info.rgbd_camera_pose_matrix = get_rgbd_camera_trans(tile_trans=tile_info.pose_matrix)

    return tile_info_dict


def integrate_object(tile_info_dict, config, save_mesh=True):
    microcsope_intrinsic = \
        generate_microscope_intrinsic_open3d(width_by_pixel=config["microscope_intrinsic"]["width_by_pixel"],
                                             height_by_pixel=config["microscope_intrinsic"]["height_by_pixel"],
                                             width_by_mm=config["size_by_mm"][0],
                                             height_by_mm=config["size_by_mm"][1],
                                             focal_distance_by_mm=config["microscope_intrinsic"]["focal_distance"])
    print(config["size_by_mm"][0] / config["microscope_intrinsic"]["width_by_pixel"] / 2)
    volume = integration.ScalableTSDFVolume(
        voxel_length=config["size_by_mm"][0] / config["microscope_intrinsic"]["width_by_pixel"] / 2,
        sdf_trunc= config["size_by_mm"][0] / config["microscope_intrinsic"]["width_by_pixel"] * 2,
        color_type=integration.TSDFVolumeColorType.RGB8)
    for tile_info_key in tile_info_dict:
        print("Integrating tile %6d" % tile_info_key)
        tile_info = tile_info_dict[tile_info_key]
        color = io.read_image(join(config["path_data"], config["path_image_dir"],
                                   tile_info.file_name) + ".png")
        depth = io.read_image(join(config["path_data"], config["path_depth_dir"],
                                   tile_info.file_name) + ".png")
        rgbd = geometry.create_rgbd_image_from_color_and_depth(
            color=color, depth=depth,
            depth_scale=config["microscope_intrinsic"]["depth_scaling"],
            depth_trunc=config["microscope_intrinsic"]["depth_threshold"],
            convert_rgb_to_intensity=False)

        pcd_rgbd = geometry.create_point_cloud_from_rgbd_image(rgbd, microcsope_intrinsic)
        draw_geometries([pcd_rgbd])
        # volume.integrate(rgbd, microcsope_intrinsic, np.linalg.inv(tile_info.rgbd_camera_pose_matrix))

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    if save_mesh:
        touch_folder(join(config["path_data"], config["path_scene_dir"]))
        mesh_name = join(config["path_data"], config["path_scene_dir"], "scene.ply")
        io.write_triangle_mesh(mesh_name, mesh, False, True)
    draw_geometries([mesh])
    return mesh
