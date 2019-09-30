import cv2
import numpy
import math
import transforms3d
from scipy.spatial.transform import Rotation
import sys

from TransformationData import LocalTransformationEstimationResult

sys.path.append("../../Utility")
sys.path.append("../Alignment")
sys.path.append("../Data_processing")
from TileInfo import *
from file_managing import *


# All the 4x4 transformation matrices is defined as the tile's initial pose is using x+ as normal direction,
# y+ as right, z+ as top direction for the photo tile.

# This is a planar transformation estimation cropping the border area. (To deal with the border noise)
# All transformation is made in the image space, represented as 3x3 matrices.
# x direction = right, y direction = down, same with the RGBD coordinates.


def planar_transformation_cv(s_img, t_img, crop_w=0, crop_h=0,
                             nfeatures=400, num_matches_thresh1=6, match_conf=0.3):
    (s_h, s_w, s_c) = s_img.shape
    (t_h, t_w, t_c) = t_img.shape
    s_img_crop = s_img[crop_h:s_h - crop_h, crop_w:s_w - crop_w]
    t_img_crop = t_img[crop_h:t_h - crop_h, crop_w:t_w - crop_w]
    # orb_finder = cv2.ORB_create(scaleFactor=1.2, nlevels=4, edgeThreshold=31,
    #                             firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE,
    #                             nfeatures=nfeatures, patchSize=31)

    sift_finder = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures,
                                              nOctaveLayers=3,
                                              contrastThreshold=0.001,
                                              edgeThreshold=100,
                                              sigma=1.6)

    matcher = cv2.detail_AffineBestOf2NearestMatcher(full_affine=False, try_use_gpu=True,
                                                     match_conf=match_conf,
                                                     num_matches_thresh1=num_matches_thresh1)

    # source_feature = cv2.detail.computeImageFeatures2(featuresFinder=orb_finder, image=s_img_crop)
    # target_feature = cv2.detail.computeImageFeatures2(featuresFinder=orb_finder, image=t_img_crop)
    source_feature = cv2.detail.computeImageFeatures2(featuresFinder=sift_finder, image=s_img_crop)
    target_feature = cv2.detail.computeImageFeatures2(featuresFinder=sift_finder, image=t_img_crop)

    # print(source_feature)
    # print(target_feature)
    matching_result = matcher.apply(source_feature, target_feature)

    matcher.collectGarbage()
    # print(type(matching_result))

    match_conf = matching_result.confidence

    if match_conf != 0.0:
        trans_cv = numpy.asarray(matching_result.H)

        trans_deoffset = numpy.asarray([[1.0, 0, crop_w],
                                        [0, 1.0, crop_h],
                                        [0, 0, 1]])
        trans_offset = numpy.asarray([[1.0, 0, -crop_w],
                                      [0, 1.0, -crop_h],
                                      [0, 0, 1]])
        trans_cv = numpy.dot(trans_deoffset, numpy.dot(trans_cv, trans_offset))
        # get the mean of the overlapping area   ============================================================
        img_s_warped = cv2.warpAffine(src=s_img, M=trans_cv[0:2, :], dsize=(t_w, t_h))

        img_s_gray = cv2.cvtColor(img_s_warped, cv2.COLOR_BGR2GRAY)
        ret, img_s_mask_one_channel = cv2.threshold(img_s_gray, 0, 1, cv2.THRESH_BINARY)
        mask_extended = img_s_mask_one_channel.reshape((-1)).T

        pixel_count_overlapping = len(mask_extended[mask_extended[:] > 0])

        mask_three_channel = numpy.c_[mask_extended,
                                      mask_extended,
                                      mask_extended].reshape((t_h, t_w, 3))

        masked_img_t = mask_three_channel * t_img

        mean_s = numpy.sum(img_s_warped.reshape((-1, 3)), axis=0) / pixel_count_overlapping
        mean_t = numpy.sum(masked_img_t.reshape((-1, 3)), axis=0) / pixel_count_overlapping

    else:
        trans_cv = numpy.identity(4)
        mean_s = numpy.asarray([0.0, 0.0, 0.0])
        mean_t = numpy.asarray([0.0, 0.0, 0.0])
        pixel_count_overlapping = 0
    return match_conf, trans_cv, mean_s, mean_t, pixel_count_overlapping # These means are working in the BGR space.


# ===========================================================================================================
# ===========================================================================================================
# ===========================================================================================================
# There are different versions of function transform_convert_from_2d_to_3d(), transform_planar_add_normal_direction()
# and trans_local_check()
# ===========================================================================================================
# Version 1: initial pose use x+ as normal direction
# ===========================================================================================================


def transform_convert_from_2d_to_3d(trans_cv_2d,
                                    width_by_pixel_s=320, height_by_pixel_s=240,
                                    width_by_m_s=0.0032, height_by_m_s=0.0016,
                                    width_by_pixel_t=640, height_by_pixel_t=480,
                                    width_by_m_t=0.0064, height_by_m_t=0.0048):
    scaling_compensate_x = (width_by_m_t / width_by_pixel_t) / (width_by_m_s / width_by_pixel_s)
    scaling_compensate_y = (height_by_m_t / height_by_pixel_t) / (height_by_m_s / height_by_pixel_s)
    trans_compensate_x = width_by_m_t / width_by_pixel_t
    trans_compensate_y = height_by_m_t / height_by_pixel_t

    compensate_factors = numpy.asarray([[1, 0, 0, 0],
                                        [0, scaling_compensate_x, scaling_compensate_y, trans_compensate_x],
                                        [0, scaling_compensate_x, scaling_compensate_y, trans_compensate_y],
                                        [0, 0, 0, 1]])
    trans_cv_four = numpy.asarray([[1, 0, 0, 0],
                                   [0, trans_cv_2d[0][0], -trans_cv_2d[0][1], trans_cv_2d[0][2]],
                                   [0, -trans_cv_2d[1][0], trans_cv_2d[1][1], -trans_cv_2d[1][2]],
                                   [0, 0, 0, 1]])
    trans_cv_3d = trans_cv_four * compensate_factors
    # (ct, cr, cz, cs) = transforms3d.affines.decompose44(trans_cv_four)
    # The trans_deoffset and trans_offset are because for image space the range is 0 < x < width, height > y > 0
    # but in world space it is -width/2 < x < width/2, -height/2 < y < height/2
    trans_deoffset = [[1, 0, 0, 0],
                      [0, 1, 0, -width_by_m_t / 2],
                      [0, 0, 1, height_by_m_t / 2],
                      [0, 0, 0, 1]]
    trans_offset = [[1, 0, 0, 0],
                    [0, 1, 0, width_by_m_s / 2],
                    [0, 0, 1, -height_by_m_s / 2],
                    [0, 0, 0, 1]]

    trans_planar_3d = numpy.dot(numpy.dot(trans_deoffset, trans_cv_3d), trans_offset)
    return trans_planar_3d


def transform_planar_add_normal_direction(trans_planar, trans_s, trans_t):
    trans_rotation = numpy.dot(numpy.linalg.inv(trans_t), trans_s)

    (pt, pr, pz, ps) = transforms3d.affines.decompose44(trans_planar)
    (rt, rr, rz, rs) = transforms3d.affines.decompose44(trans_rotation)

    rr_euler = Rotation.from_dcm(rr).as_euler("xyz")  # should have x and y value.
    # trans_one = transforms3d.affines.compose(pt / 2, pr, pz, ps)
    trans_one = transforms3d.affines.compose(pt / 2, pr, [1, 1, 1])

    print("rr_euler")
    print(rr_euler)

    trans_two = transforms3d.affines.compose([0, 0, 0],
                                             Rotation.from_euler("xyz", [0, rr_euler[1], rr_euler[2]]).as_dcm(),
                                             [1, 1, 1])

    trans_three = transforms3d.affines.compose(pt / 2, numpy.identity(3), [1, 1, 1])
    trans_with_normal_direction = numpy.dot(numpy.dot(trans_three, trans_two), trans_one)

    return trans_with_normal_direction


def trans_local_check(trans_local, s_init_trans, t_init_trans, scaling_tolerance=0.05, rotation_tolerance=0.2):
    trans_diff = numpy.dot(
        numpy.linalg.inv(numpy.dot(s_init_trans, numpy.linalg.inv(t_init_trans))), trans_local)

    (pt, pr, pz, ps) = transforms3d.affines.decompose44(trans_diff)
    rotation_euler_x = Rotation.from_dcm(pr).as_euler("xyz")[0]

    if abs(pz[0] - 1) > scaling_tolerance \
            or abs(pz[1] - 1) > scaling_tolerance \
            or abs(pz[2] - 1) > scaling_tolerance:
        print("pz: %3f, %3f, %3f" % (pz[0], pz[1], pz[2]))
        print(trans_diff)
        return False
    if abs(rotation_euler_x) > rotation_tolerance:
        print("rotation_euler_x: %3f" % rotation_euler_x)
        print(trans_diff)
        return False
    return True


def trans_info_matching(match_conf, weight=1, match_info=numpy.identity(4)):
    # The order of the 6x6 matrix should be: x, y, z, rotation_x, rotation_y, rotation_z for g2o.
    info_matrix = weight * match_conf * match_info
    # info_matrix = weight * match_info
    return info_matrix


def trans_info_sensor(weight=1, sensor_info=numpy.identity(4)):
    # The order should be: x, y, z, rotation_x, rotation_y, rotation_z
    info_matrix = weight * sensor_info
    return info_matrix


def trans_estimation_pure(s_id,
                          t_id,
                          s_img_path,
                          t_img_path,
                          width_by_pixel_s,
                          height_by_pixel_s,
                          width_by_pixel_t,
                          height_by_pixel_t,

                          width_by_m_s,
                          height_by_m_s,
                          width_by_m_t,
                          height_by_m_t,

                          crop_w,
                          crop_h,

                          s_init_trans,
                          t_init_trans,

                          n_features,
                          num_matches_thresh1,
                          match_conf_threshold,
                          scaling_tolerance,
                          rotation_tolerance):
    print(s_img_path)
    print(t_img_path)

    s_img = cv2.imread(s_img_path)
    t_img = cv2.imread(t_img_path)

    matching_conf, trans_cv_2d, mean_s, mean_t, overlapping_pixels = \
        planar_transformation_cv(s_img, t_img,
                                 crop_w=crop_w,
                                 crop_h=crop_h,
                                 nfeatures=n_features,
                                 num_matches_thresh1=num_matches_thresh1,
                                 match_conf=match_conf_threshold)
    if matching_conf == 0:
        return LocalTransformationEstimationResult(s=s_id, t=t_id,
                                                   success=False,
                                                   conf=0,
                                                   trans=numpy.identity(4),
                                                   planar_trans=numpy.identity(3),
                                                   mean_s=numpy.asarray([0.0, 0.0, 0.0]),
                                                   mean_t=numpy.asarray([0.0, 0.0, 0.0]),
                                                   overlapping_pixels=0)

    trans_planar_3d = transform_convert_from_2d_to_3d(trans_cv_2d=trans_cv_2d,
                                                      width_by_pixel_s=width_by_pixel_s,
                                                      height_by_pixel_s=height_by_pixel_s,
                                                      width_by_m_s=width_by_m_s,
                                                      height_by_m_s=height_by_m_s,
                                                      width_by_pixel_t=width_by_pixel_t,
                                                      height_by_pixel_t=height_by_pixel_t,
                                                      width_by_m_t=width_by_m_t,
                                                      height_by_m_t=height_by_m_t)

    if not trans_local_check(trans_planar_3d, s_init_trans, t_init_trans,
                             scaling_tolerance=scaling_tolerance,
                             rotation_tolerance=rotation_tolerance):
        print("Tile %05d and %05d : Local trans check fails " % (s_id, t_id))
        # return s_id, t_id, False, matching_conf, trans_planar_3d
        return LocalTransformationEstimationResult(s=s_id, t=t_id,
                                                   success=False,
                                                   conf=matching_conf,
                                                   trans=trans_planar_3d,
                                                   planar_trans=trans_cv_2d,
                                                   mean_s=numpy.asarray([0.0, 0.0, 0.0]),
                                                   mean_t=numpy.asarray([0.0, 0.0, 0.0]),
                                                   overlapping_pixels=0)
    else:
        print("Tile %05d and %05d : Trans check passed" % (s_id, t_id))
        trans_3d = transform_planar_add_normal_direction(trans_planar_3d, s_init_trans, t_init_trans)
        # return s_id, t_id, True, matching_conf, trans_3d
        return LocalTransformationEstimationResult(s=s_id, t=t_id,
                                                   success=True,
                                                   conf=matching_conf,
                                                   trans=trans_3d,
                                                   planar_trans=trans_cv_2d,
                                                   mean_s=mean_s,
                                                   mean_t=mean_t,
                                                   overlapping_pixels=overlapping_pixels)


def trans_estimation_tile(tile_info_s: TileInfo, tile_info_t: TileInfo, config):
    return trans_estimation_pure(s_id=tile_info_s.tile_index, t_id=tile_info_t.tile_index,
                                 s_img_path=join(config["path_data"],
                                                 config["path_image_dir"],
                                                 tile_info_s.file_name) + ".png",
                                 t_img_path=join(config["path_data"],
                                                 config["path_image_dir"],
                                                 tile_info_t.file_name) + ".png",
                                 width_by_pixel_s=tile_info_s.width_by_pixel,
                                 height_by_pixel_s=tile_info_s.height_by_pixel,
                                 width_by_pixel_t=tile_info_t.width_by_pixel,
                                 height_by_pixel_t=tile_info_t.height_by_pixel,
                                 width_by_m_s=tile_info_s.width_by_m,
                                 height_by_m_s=tile_info_s.height_by_m,
                                 width_by_m_t=tile_info_t.width_by_m,
                                 height_by_m_t=tile_info_t.height_by_m,
                                 crop_w=config["crop_width_by_pixel"],
                                 crop_h=config["crop_height_by_pixel"],
                                 s_init_trans=tile_info_s.init_transform_matrix,
                                 t_init_trans=tile_info_t.init_transform_matrix,
                                 n_features=config["n_features"],
                                 num_matches_thresh1=config["num_matches_thresh1"],
                                 match_conf_threshold=config["conf_threshold"],
                                 scaling_tolerance=config["scaling_tolerance"],
                                 rotation_tolerance=config["rotation_tolerance"])


# ================================================================================================
# If take the curved surface and projection into consideration, there would be some changes in the translation distance
# ================================================================================================


def real_translation_distance(planar_translation_distance, normal_direction_difference_angle, camera_ins):
    curved_translation_distance = 0
    return curved_translation_distance

# Need some more work in this function.


# def extract_translation_distance(trans_matrix):
#     # takes a 4x4 matrix but it should only happen in the planar surface YoZ.#
#     translation_x = trans_matrix[0][3]
#     translation_y = trans_matrix[1][3]
#     translation_z = trans_matrix[2][3]
#     distance = math.sqrt(math.pow(translation_x, 2) + math.pow(translation_y, 2) + math.pow(translation_z, 2))
#     return distance


# def update_translation_distance(trans_matrix, new_distance):
#     distance = extract_translation_distance(trans_matrix)
#     distance_factor = new_distance / distance
#     trans_matrix[0][3] = trans_matrix[0][3] * distance_factor
#     trans_matrix[1][3] = trans_matrix[1][3] * distance_factor
#     trans_matrix[2][3] = trans_matrix[2][3] * distance_factor
#     return trans_matrix
