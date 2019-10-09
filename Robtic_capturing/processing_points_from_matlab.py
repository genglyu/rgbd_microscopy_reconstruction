import sys
sys.path.append("../../Utility")
sys.path.append("../Alignment")
sys.path.append("../Data_processing")
from open3d import *
import DataConvert
import numpy
import math
from scipy.spatial.transform import Rotation
import transforms3d


# def rotation_matrix(normal):
#     a = numpy.asarray(normal)
#     [x, y, z] = (a / numpy.linalg.norm(a)).tolist()
#     projected_yz = y*y + z*z
#     if projected_yz != 0:
#         rot_y = math.acos(x)
#         rot_x = math.asin(y / math.sqrt(projected_yz))
#         if z < 0:
#             rot_x = math.pi - rot_x
#         [n_x, n_y, n_z] = Rotation.from_euler("yxz", [-rot_y, -rot_x, 0]).as_euler("xyz")
#         rotation = Rotation.from_euler("xyz", [0, n_y, n_z]).as_dcm()
#     else:
#         if x >= 0:
#             rotation = numpy.identity(3)
#         else:
#             rotation = Rotation.from_euler("xyz", [0, math.pi, 0]).as_dcm()
#     return rotation

# trans_rob_to_camera = transforms3d.affines.compose(T=[0, 0, 0],
#                                                    R=[[-1, 0, 0],
#                                                       [0, -1, 0],
#                                                       [0, 0, -1]],
#                                                    Z=[1, 1, 1])

plane_shifting_rotation_trans = numpy.array([[0, 1, 0],
                                             [0, 0, 1],
                                             [1, 0, 0]])

def rotation_matrix(normal):
    a = numpy.asarray(normal)
    [x, y, z] = (a / numpy.linalg.norm(a)).tolist()
    projected_xy = x*x + y*y
    if projected_xy != 0:
        rot_x = math.acos(z)
        rot_z = math.asin(x / math.sqrt(projected_xy))
        if y < 0:
            rot_z = math.pi - rot_z
        [n_z, n_x, n_y] = Rotation.from_euler("xzy", [-rot_x, -rot_z, 0]).as_euler("zxy")
        rotation = Rotation.from_euler("zxy", [0, n_x, n_y]).as_dcm()
        rotation = numpy.dot(rotation, plane_shifting_rotation_trans)
    else:
        if z >= 0:
            rotation = numpy.identity(3)
        else:
            rotation = Rotation.from_euler("zxy", [0, math.pi, 0]).as_dcm()
    return rotation


point_list_read = DataConvert.read_points_list("/home/lvgeng/Code/TestingData/robotic/matlab/surface_interpolation_dir/matlab_1002_2.json")

point_list = []
for point in point_list_read:
    print(point)
    # if point[2] >= 0.137:
    point_list.append(point)

pcd = PointCloud()
pcd.points = Vector3dVector(point_list)
estimate_normals(pcd)

normals = []
for normal in pcd.normals:
    if normal[2] < 0:
        normals.append(normal * -1)
    else:
        normals.append(normal)
pcd.normals = Vector3dVector(normals)

pcd_down = geometry.voxel_down_sample(pcd, voxel_size=0.0013)

coor = geometry.create_mesh_coordinate_frame(size=0.1)

pcd_down = io.read_point_cloud("cropped_1.ply")

viewer = VisualizerWithEditing()
viewer.create_window()

viewer.add_geometry(pcd_down)

viewer.add_geometry(coor)

viewer.run()
viewer.destroy_window()


pcd_cropped = io.read_point_cloud("cropped_1.ply")

draw_geometries([coor, pcd_cropped])


trans_list = []
for i, position in enumerate(pcd_cropped.points):
    print("Rotation for point %d" % i)
    rotation_m = rotation_matrix(pcd_cropped.normals[i])
    trans = transforms3d.affines.compose(T=position, R=rotation_m, Z=[1, 1, 1])
    trans_list.append(trans)

DataConvert.save_trans_list(
    path="/home/lvgeng/Code/TestingData/robotic/matlab/surface_interpolation_dir/robotic_reconstruction_trans_interpolated.json",
    trans_list=trans_list)

