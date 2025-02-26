import sys
import numpy

sys.path.append("../Data_processing")
sys.path.append("../../Utility")
import image_processing
import TileInfo
import TileInfoDict
import visualization_make
import open3d


class MicroscopyReconstructionVisualizerOpen3d:
    def __init__(self, tile_info_dict, config):

        self.tile_info_dict = tile_info_dict
        self.config = config

        self.pose_pan = visualization_make.generate_pose_points_and_normals(tile_info_dict)
        self.sensor_pan = visualization_make.generate_sensor_points_and_normals(tile_info_dict)

        self.c_edges = visualization_make.generate_confirmed_edges(tile_info_dict)
        self.f_edges = visualization_make.generate_false_edges(tile_info_dict)

        # ==================================================
        self.pcd_pose = None
        self.wireframes_pose = None

        self.c_edgeset_pose = None
        self.f_edgeset_pose = None

        self.full_image_pcd_list_pose = None
        self.cropped_image_pcd_list_pose = None

        self.highlight_tile = None
        # ==================================================
        self.pcd_sensor = None
        self.wireframes_sensor = None

        self.c_edgeset_sensor = None
        self.f_edgeset_sensor = None

        self.full_image_pcd_list_sensor = None
        self.full_image_pcd_list_sensor_with_label = None

        self.sensor_edgeset = None

    def visualize_config(self):
        for view_setting in self.config["visualization"]:
            self.view(
                pcd_pose=view_setting["pcd_pose"],
                wireframes_pose=view_setting["wireframes_pose"],
                c_edgeset_pose=view_setting["c_edgeset_pose"],
                f_edgeset_pose=view_setting["f_edgeset_pose"],
                full_image_pcd_list_pose=view_setting["full_image_pcd_list_pose"],
                cropped_image_pcd_list_pose=view_setting["cropped_image_pcd_list_pose"],
                highlight_tile_list=view_setting["highlight_tile_list"],

                pcd_sensor=view_setting["pcd_sensor"],
                wireframes_sensor=view_setting["wireframes_sensor"],
                c_edgeset_sensor=view_setting["c_edgeset_sensor"],
                f_edgeset_sensor=view_setting["f_edgeset_sensor"],
                full_image_pcd_list_sensor=view_setting["full_image_pcd_list_sensor"],
                full_image_pcd_list_sensor_with_label=view_setting["full_image_pcd_list_sensor_with_label"],
                image_pcd_downsample_factor=view_setting["image_pcd_downsample_factor"]
            )

    def view(self,
             pcd_pose=False,
             wireframes_pose=False,
             c_edgeset_pose=False,
             f_edgeset_pose=False,
             full_image_pcd_list_pose=False,
             cropped_image_pcd_list_pose=False,
             highlight_tile_list=[],

             pcd_sensor=False,
             wireframes_sensor=False,
             c_edgeset_sensor=False,
             f_edgeset_sensor=False,
             full_image_pcd_list_sensor=False,
             full_image_pcd_list_sensor_with_label=False,

             sensor_edgeset=False,
             image_pcd_downsample_factor=0.1):

        draw_list = []

        if len(highlight_tile_list) > 0:
            highlighted_wire_frames = []
            for tile_key in highlight_tile_list:
                tile_info = self.tile_info_dict[tile_key]
                wire_frame = visualization_make.make_tile_frame(
                    trans_matrix=TileInfoDict.numpy.dot(tile_info.pose_matrix, TileInfoDict.numpy.array([[1, 0, 0, 0.00002], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]])),
                    width_by_mm=tile_info.width_by_m, height_by_mm=tile_info.height_by_m, color=[0.0, 1.0, 0.])
                highlighted_wire_frames.append(wire_frame)
            self.highlight_tile = highlighted_wire_frames
            draw_list += self.highlight_tile

        if pcd_pose:
            if self.pcd_pose is None:
                self.pcd_pose = visualization_make.make_point_cloud(points=self.pose_pan["points"],
                                                                    color=self.config["pcd_pose_color"],
                                                                    normals=self.pose_pan["normals"])
            draw_list += [self.pcd_pose]

        if wireframes_pose:
            if self.wireframes_pose is None:
                self.wireframes_pose = \
                    visualization_make.make_wireframes_pose(tile_info_dict=self.tile_info_dict,
                                                            color=self.config["wireframes_pose_color"])
            draw_list += self.wireframes_pose

        if c_edgeset_pose:
            if self.c_edgeset_pose is None:
                self.c_edgeset_pose = visualization_make.make_edge_set(points=self.pose_pan["points"],
                                                                       edges=self.c_edges,
                                                                       color=self.config["c_edgeset_pose_color"])
            draw_list += [self.c_edgeset_pose]
        if f_edgeset_pose:
            if self.f_edgeset_pose is None:
                self.f_edgeset_pose = visualization_make.make_edge_set(points=self.pose_pan["points"],
                                                                       edges=self.f_edges,
                                                                       color=self.config["f_edgeset_pose_color"])
            draw_list += [self.f_edgeset_pose]

        if full_image_pcd_list_pose:
            if self.full_image_pcd_list_pose is None:
                self.full_image_pcd_list_pose = \
                    visualization_make.make_full_image_pcd_list_pose(tile_info_dict=self.tile_info_dict,

                                                                     path_data=self.config["path_data"],
                                                                     dataset_group_template=self.config[
                                                                         "dataset_folder_template"],
                                                                     color_directory_path=self.config["path_image_dir"],

                                                                     downsample_factor=image_pcd_downsample_factor,
                                                                     color_filter=self.config["full_image_pcd_pose_color_filter"]
                                                                     )
            draw_list += self.full_image_pcd_list_pose

        if cropped_image_pcd_list_pose:
            if self.cropped_image_pcd_list_pose is None:
                self.cropped_image_pcd_list_pose = \
                    visualization_make.make_cropped_image_pcd_list_pose(tile_info_dict=self.tile_info_dict,
                                                                        img_directory_path=TileInfoDict.join(self.config["path_data"],
                                                                                                             self.config["path_image_dir"]))
            draw_list += self.cropped_image_pcd_list_pose

        if pcd_sensor:
            if self.pcd_sensor is None:
                self.pcd_sensor = visualization_make.make_point_cloud(points=self.sensor_pan["points"],
                                                                      color=self.config["pcd_sensor_color"],
                                                                      normals=self.sensor_pan["normals"])
            draw_list += [self.pcd_sensor]

        if wireframes_sensor:
            if self.wireframes_sensor is None:
                self.wireframes_sensor = \
                    visualization_make.make_wireframes_sensor(tile_info_dict=self.tile_info_dict,
                                                              color=self.config["wireframes_sensor_color"])
            draw_list += self.wireframes_sensor

        if c_edgeset_sensor:
            if self.c_edgeset_sensor is None:
                self.c_edgeset_sensor = visualization_make.make_edge_set(points=self.sensor_pan["points"],
                                                                         edges=self.c_edges,
                                                                         color=self.config["c_edgeset_sensor_color"])
            draw_list += [self.c_edgeset_sensor]
        if f_edgeset_sensor:
            if self.f_edgeset_sensor is None:
                self.f_edgeset_sensor = visualization_make.make_edge_set(points=self.sensor_pan["points"],
                                                                         edges=self.f_edges,
                                                                         color=self.config["f_edgeset_sensor_color"])
            draw_list += [self.f_edgeset_sensor]


        if full_image_pcd_list_sensor:
            if self.full_image_pcd_list_sensor is None:
                self.full_image_pcd_list_sensor = \
                    visualization_make.make_full_image_pcd_list_sensor(tile_info_dict=self.tile_info_dict,
                                                                       downsample_factor=image_pcd_downsample_factor,
                                                                       color_directory_path=TileInfoDict.join(self.config["path_data"],
                                                                                                              self.config["path_image_dir"]),
                                                                       color_filter=self.config["full_image_pcd_sensor_color_filter"]
                                                                       )
            draw_list += self.full_image_pcd_list_sensor

        if full_image_pcd_list_sensor_with_label:
            if self.full_image_pcd_list_sensor_with_label is None:
                self.full_image_pcd_list_sensor_with_label = \
                    visualization_make.make_full_image_pcd_list_sensor_with_label(tile_info_dict=self.tile_info_dict,
                                                                                  downsample_factor=image_pcd_downsample_factor,
                                                                                  path_data=self.config["path_data"],
                                                                                  dataset_group_template=self.config["dataset_folder_template"],
                                                                                  color_directory_path=self.config["path_image_dir"],
                                                                                  color_filter=self.config["full_image_pcd_sensor_color_filter"]
                                                                                  )
            draw_list += self.full_image_pcd_list_sensor_with_label

        if sensor_edgeset:
            if self.sensor_edgeset is None:
                self.sensor_edgeset = visualization_make.make_pose_sensor_edge_set(tile_info_dict=self.tile_info_dict,
                                                                                   color=self.config["sensor_edgeset_color"])
            draw_list += [self.sensor_edgeset]

        open3d.visualization.draw_geometries(draw_list)
