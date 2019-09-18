import sys
sys.path.append("../Data_processing")
sys.path.append("../../Utility")
from image_processing import *
from TileInfo import *
from TileInfoDict import *
from visualization_make import *


class MicroscopyReconstructionVisualizerOpen3d:
    def __init__(self, tile_info_dict, config):

        self.tile_info_dict = tile_info_dict
        self.config = config

        self.pose_pan = generate_pose_points_and_normals(tile_info_dict)
        self.sensor_pan = generate_sensor_points_and_normals(tile_info_dict)

        self.c_edges = generate_confirmed_edges(tile_info_dict)
        self.f_edges = generate_false_edges(tile_info_dict)

        # ==================================================
        self.pcd_pose = None
        self.wireframes_pose = None

        self.c_edgeset_pose = None
        self.f_edgeset_pose = None

        self.full_image_pcd_list_pose = None
        # ==================================================
        self.pcd_sensor = None
        self.wireframes_sensor = None

        self.c_edgeset_sensor = None
        self.f_edgeset_sensor = None

        self.full_image_pcd_list_sensor = None

        self.sensor_edgeset = None

    def visualize_config(self):
        for view_setting in self.config["visualization"]:
            self.view(
                pcd_pose=view_setting["pcd_pose"],
                wireframes_pose=view_setting["wireframes_pose"],
                c_edgeset_pose=view_setting["c_edgeset_pose"],
                f_edgeset_pose=view_setting["f_edgeset_pose"],
                full_image_pcd_list_pose=view_setting["full_image_pcd_list_pose"],
                pcd_sensor=view_setting["pcd_sensor"],
                wireframes_sensor=view_setting["wireframes_sensor"],
                c_edgeset_sensor=view_setting["c_edgeset_sensor"],
                f_edgeset_sensor=view_setting["f_edgeset_sensor"],
                full_image_pcd_list_sensor=view_setting["full_image_pcd_list_sensor"],
                image_pcd_downsample_factor=view_setting["image_pcd_downsample_factor"]
            )

    def view(self,
             pcd_pose=False,
             wireframes_pose=False,
             c_edgeset_pose=False,
             f_edgeset_pose=False,
             full_image_pcd_list_pose=False,

             pcd_sensor=False,
             wireframes_sensor=False,
             c_edgeset_sensor=False,
             f_edgeset_sensor=False,
             full_image_pcd_list_sensor=False,

             sensor_edgeset=False,
             image_pcd_downsample_factor=0.1):

        draw_list = []
        if pcd_pose:
            if self.pcd_pose is None:
                self.pcd_pose = make_point_cloud(points=self.pose_pan["points"],
                                                 color=self.config["key_frame_pcd_pose_color"],
                                                           normals=self.pose__pan["normals"])
            draw_list += [self.pcd_pose]

        if wireframes_pose:
            if self.wireframes_pose is None:
                self.wireframes_pose = \
                    make_wireframes_pose(tile_info_dict=self.tile_info_dict,
                                         color=self.config["key_frame_wireframes_pose_color"])
            draw_list += self.wireframes_pose

        if c_edgeset_pose:
            if self.c_edgeset_pose is None:
                self.c_edgeset_pose = make_edge_set(points=self.pose_pan["points"],
                                                    edges=self.c_edges,
                                                    color=self.config["c_edgeset_pose_color"])
            draw_list += [self.c_edgeset_pose]
        if f_edgeset_pose:
            if self.f_edgeset_pose is None:
                self.f_edgeset_pose = make_edge_set(points=self.pose_pan["points"],
                                                    edges=self.f_edges,
                                                    color=self.config["f_edgeset_pose_color"])
            draw_list += [self.c_edgeset_pose]

        if full_image_pcd_list_pose:
            if self.full_image_pcd_list_pose is None:
                self.full_image_pcd_list_pose = \
                    make_full_image_pcd_list_pose(tile_info_dict=self.tile_info_dict,
                                                  downsample_factor=image_pcd_downsample_factor,
                                                  color_filter=self.config["full_image_pcd_pose_color_filter"]
                                                  )
            draw_list += self.full_image_pcd_list_pose

        if pcd_sensor:
            if self.pcd_sensor is None:
                self.pcd_sensor = make_point_cloud(points=self.pose_pan["points"],
                                                 color=self.config["key_frame_pcd_pose_color"],
                                                           normals=self.pose_pan["normals"])
            draw_list += [self.pcd_sensor]

        if wireframes_sensor:
            if self.wireframes_sensor is None:
                self.wireframes_sensor = \
                    make_wireframes_sensor(tile_info_dict=self.tile_info_dict,
                                         color=self.config["key_frame_wireframes_sensor_color"])
            draw_list += self.wireframes_sensor

        if c_edgeset_sensor:
            if self.c_edgeset_sensor is None:
                self.c_edgeset_sensor = make_edge_set(points=self.sensor_pan["points"],
                                                    edges=self.c_edges,
                                                    color=self.config["c_edgeset_sensor_color"])
            draw_list += [self.c_edgeset_sensor]
        if f_edgeset_sensor:
            if self.f_edgeset_sensor is None:
                self.f_edgeset_sensor = make_edge_set(points=self.sensor_pan["points"],
                                                    edges=self.f_edges,
                                                    color=self.config["f_edgeset_sensor_color"])
            draw_list += [self.c_edgeset_sensor]

        if full_image_pcd_list_sensor:
            if self.full_image_pcd_list_sensor is None:
                self.full_image_pcd_list_sensor = \
                    make_full_image_pcd_list_sensor(tile_info_dict=self.tile_info_dict,
                                                  downsample_factor=image_pcd_downsample_factor,
                                                  color_filter=self.config["full_image_pcd_sensor_color_filter"]
                                                  )
            draw_list += self.full_image_pcd_list_sensor

        if sensor_edgeset:
            if self.sensor_edgeset is None:
                self.sensor_edgeset = make_pose_sensor_edge_set(tile_info_dict=self.tile_info_dict,
                                                                color=self.config["sensor_edgeset_color"])
            draw_list += [self.sensor_edgeset]

        draw_geometries(draw_list)
