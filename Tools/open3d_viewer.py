import VisualizerOpen3d
import surface_interpolation
import DataConvert
import open3d
import numpy
import visualization_make

raw_human_guide_sampling_points = surface_interpolation.read_raw_robotic_pose_json_as_points(
    "/home/lvgeng/writing/microscope_project/fruit/testing_pose.testingjson")

reference_center = surface_interpolation.find_bonding_box(raw_human_guide_sampling_points)[0]

print(surface_interpolation.find_bonding_box(raw_human_guide_sampling_points))
print(len(raw_human_guide_sampling_points))

raw_human_guide_sampling_points = surface_interpolation.recenter(
    raw_human_guide_sampling_points,
    reference_center)

interpolated_points = surface_interpolation.interpolate_points_z(raw_human_guide_sampling_points)



points_fitting = DataConvert.read_points_list(
    "/home/lvgeng/writing/microscope_project/fruit/matlab1001_1.json")

points_fitting = surface_interpolation.recenter(
    points_fitting,
    reference_center)

print(surface_interpolation.find_bonding_box(raw_human_guide_sampling_points))
print(surface_interpolation.find_bonding_box(points_fitting))


points_fitting = numpy.asarray(points_fitting)
points_fitting = points_fitting[numpy.logical_not(
    points_fitting[:, 2] < 0.005
)]


pcd_raw = open3d.geometry.PointCloud()
pcd_raw.points = open3d.utility.Vector3dVector(raw_human_guide_sampling_points)
pcd_raw_line = visualization_make.make_connection_of_pcd_order(pcd_raw)

pcd_interpolated = open3d.geometry.PointCloud()
pcd_interpolated.points = open3d.utility.Vector3dVector(interpolated_points)
pcd_interpolated.estimate_normals()

pcd_fitting = open3d.geometry.PointCloud()
pcd_fitting.points = open3d.utility.Vector3dVector(points_fitting)
pcd_fitting.estimate_normals()
# pcd_fitting = pcd_fitting.voxel_down_sample(voxel_size=0.001)


open3d.visualization.draw_geometries([pcd_raw, pcd_raw_line])
open3d.visualization.draw_geometries([pcd_interpolated])
open3d.visualization.draw_geometries([pcd_fitting])




