import open3d
import file_managing
import numpy
import visualize_textured_mesh_panda3d
import cv2

def recapture_all_tiles_in_dict(tile_info_dict, triangle_info_list, texture_dir="",
                                width_pixel=800, height_pixel=600,
                                tile_width=4.4,
                                camera_focal_distance_by_mm=50,
                                recaptured_tile_dir="",
                                recaptured_tile_template="tile_%06d.png"):
    visualizer_panda3d = visualize_textured_mesh_panda3d.MicroscopyApp()
    visualizer_panda3d.init_virtual_microscope_camera(focal_distance_by_mm=camera_focal_distance_by_mm,
                                                      tile_width_by_pixel=width_pixel,
                                                      tile_height_by_pixel=height_pixel,
                                                      tile_width_by_mm=tile_width)
    visualizer_panda3d.load_textured_triangles(triangle_info_list=triangle_info_list, merged_texture_dir=texture_dir)

    file_managing.touch_folder(recaptured_tile_dir)

    for tile_info_key in tile_info_dict:
        tile_id = tile_info_dict[tile_info_key].tile_index
        tile_pose = tile_info_dict[tile_info_key].pose_matrix
        visualizer_panda3d.set_virtual_camera_to_tile_pose(tile_pose=tile_pose,
                                                           focal_distance_by_mm=camera_focal_distance_by_mm)

        visualizer_panda3d.capture_rendered_image(
            save_path=file_managing.join(recaptured_tile_dir, recaptured_tile_template) % tile_id)


