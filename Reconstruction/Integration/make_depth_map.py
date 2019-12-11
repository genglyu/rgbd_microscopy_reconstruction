import numpy
import cv2
import sys
from open3d import *

sys.path.append("../Data_processing/")
from DataConvert import *
from TileInfoDict import *
from scipy.interpolate import griddata


def generate_microscope_intrinsic_open3d(width_by_pixel=640, height_by_pixel=480, width_by_mm=10, height_by_mm=7.5,
                                         focal_distance_by_mm=0.1):
    intrinsic = camera.PinholeCameraIntrinsic(camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    intrinsic.width = width_by_pixel
    intrinsic.height = height_by_pixel

    cx = (width_by_pixel - 1.0) / 2
    cy = (height_by_pixel - 1.0) / 2
    fx = focal_distance_by_mm * width_by_pixel / width_by_mm
    fy = focal_distance_by_mm * height_by_pixel / height_by_mm

    intrinsic.intrinsic_matrix = [[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]]
    # print("intrinsic.width: %d " % intrinsic.width)
    # print("intrinsic.height: %d" % intrinsic.height)
    return intrinsic


def get_rgbd_camera_trans(tile_trans=numpy.identity(4), focal_distance_by_m=0.1):
    tile_to_rgbd_camera_matrix = numpy.array([[0, 0, -1, 0],
                                              [1, 0, 0, 0],
                                              [0, -1, 0, 0],
                                              [0, 0, 0, 1]])
    rgbd_camera_trans = numpy.dot(tile_trans, tile_to_rgbd_camera_matrix)
    rgbd_camera_trans = numpy.dot(rgbd_camera_trans, numpy.array([[1, 0, 0, 0],
                                                                  [0, 1, 0, 0],
                                                                  [0, 0, 1, -focal_distance_by_m],
                                                                  [0, 0, 0, 1]]))
    return rgbd_camera_trans


# Make single depth map.
def make_single_depth_image(tile_trans,
                            all_tiles_center_position_list, all_tiles_center_position_kd_tree, searching_range_radius,
                            depth_camera_intrinsic:camera.PinholeCameraIntrinsic,
                            focal_distance_by_mm=100,
                            depth_scaling_factor=100000,
                            saving_path=None):
    # find the closest ones to the tile needs depth map.
    tile_position = numpy.dot(tile_trans, numpy.array([0, 0, 0, 1]).T).T[0:3]
    camera_trans = get_rgbd_camera_trans(tile_trans=tile_trans, focal_distance_by_m=focal_distance_by_mm)

    [_, idx, _] = all_tiles_center_position_kd_tree.search_radius_vector_3d(tile_position, searching_range_radius)
    reference_points = numpy.ones((len(idx), 4))
    reference_points[:, :-1] = numpy.asarray(adjust_order(source_list=all_tiles_center_position_list, index_list=idx))

    # move to the view matrix ==================================================================================
    reference_points = numpy.dot(numpy.linalg.inv(camera_trans), reference_points.T)[0:3].T


   # apply projection
    fx = depth_camera_intrinsic.intrinsic_matrix[0][0]
    fy = depth_camera_intrinsic.intrinsic_matrix[0][0]
    cx = depth_camera_intrinsic.intrinsic_matrix[0][2]
    cy = depth_camera_intrinsic.intrinsic_matrix[1][2]
    depth_reference_points = []
    for point in reference_points:
        # print(point)
        z = point[2]
        x = (fx / z) * point[0] + cx
        y = (fy / z) * point[1] + cy
        depth_reference_points.append([x, y, z])

    depth_reference_points = numpy.asarray(depth_reference_points)
    # print(depth_reference_points)
    # start interpolation
    xy = depth_reference_points[:, 0:2]
    z = depth_reference_points[:, 2]

    grid_x, grid_y = numpy.mgrid[
                     0:depth_camera_intrinsic.height:(depth_camera_intrinsic.height * 1j),
                     0:depth_camera_intrinsic.width:(depth_camera_intrinsic.width * 1j)]

    # grid_z = griddata(xy, z, (grid_x, grid_y), method='cubic')
    grid_z = griddata(xy, z, (grid_x, grid_y), method='linear')
    # formatting and saving=======================================================================
    depth_map = grid_z.reshape((depth_camera_intrinsic.height, depth_camera_intrinsic.width)) * depth_scaling_factor
    depth_map = depth_map.astype("uint16")

    if saving_path is not None:
        cv2.imwrite(saving_path, depth_map)

    return depth_map
