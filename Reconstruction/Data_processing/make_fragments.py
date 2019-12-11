import sys
sys.path.append("./PointCloudCrop")
from crop_point_cloud import generate_cropped_tile
import open3d
import numpy


def make_fragment_pcd_for_single_group(tile_info_dict_single_group,
                                       img_directory_path,
                                       voxel_size=-1):
    # points_single_group = []
    # colors_single_group = []
    point_cloud = open3d.geometry.PointCloud()

    for tile_info_key in tile_info_dict_single_group:
        tile_info = tile_info_dict_single_group[tile_info_key]
        if len(tile_info.confirmed_neighbour_list) > 0:
            points, colors = generate_cropped_tile(tile_index=tile_info_key,
                                                   tile_info_dict=tile_info_dict_single_group,
                                                   img_directory_path=img_directory_path,
                                                   in_group=True)
            # points_single_group += points
            # colors_single_group += colors
            pcd_single_group = open3d.geometry.PointCloud()
            pcd_single_group.points = open3d.utility.Vector3dVector(points)
            pcd_single_group.colors = open3d.utility.Vector3dVector(colors)
            pcd_single_group.normals = open3d.utility.Vector3dVector(numpy.repeat([[1, 0, 0]], len(points), axis=0))

            pcd_single_group.transform(tile_info.pose_matrix_in_group)

            point_cloud += pcd_single_group
    # point_cloud.points = Vector3dVector(points)
    # point_cloud.colors = Vector3dVector(colors)
    print("Fragment points original: %d" % len(point_cloud.points))
    if voxel_size != -1:
        point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
        print("Fragment points downsampled: %d" % len(point_cloud.points))
    return point_cloud


def save_fragment_pcd(fragment:open3d.geometry.PointCloud, save_path):
    open3d.io.write_point_cloud(filename=save_path, pointcloud=fragment, compressed=True)


def read_fragment_pcd(save_path):
    return open3d.io.read_point_cloud(filename=save_path)





