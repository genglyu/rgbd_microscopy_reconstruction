import sys
sys.path.append("../../Utility")
sys.path.append("../Alignment")
sys.path.append("../Data_processing")
from open3d import *
import numpy
import math
from scipy.spatial.transform import Rotation
import transforms3d


from open3d import *
from DataConvert import *
import numpy


def remove_out_range_trans(source_trans_list, reference_trans_list,
                           search_radius=0.01, amount_threholds=1, off_center_rate=0.2):
    source_points = trans_list_to_points(source_trans_list)
    reference_points = trans_list_to_points(reference_trans_list)

    pcd_reference = PointCloud()
    pcd_reference.points = Vector3dVector(reference_points)
    kd_tree_reference = KDTreeFlann(pcd_reference)

    cropped_source_points = []
    cropped_source_trans_list = []
    for index, point in enumerate(source_points):
        [_, idx, _] = kd_tree_reference.search_radius_vector_3d(point, radius=search_radius)

        points_in_range_center = numpy.array([0.0, 0.0, 0.0])
        if len(idx) >= amount_threholds:
            for i in idx:
                points_in_range_center += numpy.asarray(reference_points[i])
            # print(points_in_range_center)
            points_in_range_center = points_in_range_center / len(idx)
            distance = numpy.linalg.norm(points_in_range_center - point)
            if distance < search_radius * off_center_rate:
                cropped_source_points.append(point)
                cropped_source_trans_list.append(source_trans_list[index])

    print(cropped_source_trans_list)
    return cropped_source_trans_list

