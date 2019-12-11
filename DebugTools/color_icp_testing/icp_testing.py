import local_transformation_estimation_fragment
import open3d
import numpy

pcd_s = open3d.io.read_point_cloud("/home/lvgeng/Code/TestingData/multiple_group_testing/fragments/tile_group_00.ply")
pcd_t = open3d.io.read_point_cloud("/home/lvgeng/Code/TestingData/multiple_group_testing/fragments/tile_group_02.ply")

# pcd_s.transform(numpy.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))

pcd_s, pcd_s_fpfh = local_transformation_estimation_fragment.point_cloud_processing(pcd=pcd_s, voxel_size=0.001)
pcd_t, pcd_t_fpfh = local_transformation_estimation_fragment.point_cloud_processing(pcd=pcd_t, voxel_size=0.001)

success, init_trans = local_transformation_estimation_fragment.init_trans_fragments_fpfh(
    pcd_s, pcd_t,
    pcd_s_fpfh, pcd_t_fpfh,
    voxel_size=0.001)
# init_trans = numpy.identity(4)

open3d.draw_geometries([pcd_s, pcd_t])

# trans = local_transformation_estimation_fragment.multiscale_color_icp(
#     pcd_s, pcd_t,
#     corresponding_distance_list=[0.01],
#     max_iter_list=[50],
#     init_transformation=init_trans)

# pcd_s.transform(trans)
pcd_s.transform(init_trans)
print("success")
print(success)

pcd_s.colors = open3d.Vector3dVector(pcd_s.colors * numpy.array([1, 0.5, 0.5]))
open3d.draw_geometries([pcd_s, pcd_t])