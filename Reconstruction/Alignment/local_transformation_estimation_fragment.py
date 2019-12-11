import open3d
import numpy


def point_cloud_processing(pcd, voxel_size=0.001):
    pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
    pcd_fpfh_feature = open3d.registration.compute_fpfh_feature(
        input=pcd,
        search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                                             max_nn=30))
    return pcd, pcd_fpfh_feature


def init_trans_fragments_fpfh(pcd_s, pcd_t,
                              pcd_s_fpfh_feature, pcd_t_fpfh_feature,
                              voxel_size=0.001):
    distance_threshold = voxel_size * 1.4
    result = open3d.registration.registration_fast_based_on_feature_matching(
        pcd_s, pcd_t, pcd_s_fpfh_feature, pcd_t_fpfh_feature,
        open3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))

    # result = registration.registration_ransac_based_on_feature_matching(
    #     pcd_s, pcd_t,
    #     pcd_s_fpfh_feature, pcd_t_fpfh_feature,
    #     distance_threshold,
    #     registration.TransformationEstimationPointToPoint(False),
    #     4,
    #     [registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
    #      registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
    #     registration.RANSACConvergenceCriteria(4000000, 500))

    if result.transformation.trace() == 4.0:
        return False, numpy.identity(4)
    else:
        information = open3d.registration.get_information_matrix_from_point_clouds(
            pcd_s, pcd_t, distance_threshold, result.transformation)
        if information[5, 5] / min(len(pcd_s.points), len(pcd_t.points)) < 0.3:
            return False, numpy.identity(4)
        else:
            return True, result.transformation


# def trans_estimation_fragments_color_icp()
def multiscale_color_icp(pcd_s,
                         pcd_t,
                         corresponding_distance_list=[0.005, 0.002, 0.0005],
                         max_iter_list=[50, 40, 30],
                         init_transformation=open3d.np.identity(4)):
    current_trans = init_transformation
    for scale_level, voxel_size in enumerate(corresponding_distance_list):
        # voxel_size = corresponding_distance_list[scale_level]
        iter = max_iter_list[scale_level]

        pcd_s_down = pcd_s.voxel_down_sample(voxel_size=voxel_size)
        pcd_t_down = pcd_t.voxel_down_sample(voxel_size=voxel_size)

        pcd_s_down.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                                                                         max_nn=30))
        pcd_t_down.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                                                                         max_nn=30))

        result_icp = open3d.registration.registration_colored_icp(source=pcd_s_down,
                                                                  target=pcd_t_down,
                                                                  max_correspondence_distance=voxel_size,
                                                                  init=current_trans,
                                                                  criteria=open3d.registration.ICPConvergenceCriteria(
                                                                      relative_fitness=1e-6,
                                                                      relative_rmse=1e-6,
                                                                      max_iteration=iter))
        current_trans = result_icp.transformation
    return current_trans
