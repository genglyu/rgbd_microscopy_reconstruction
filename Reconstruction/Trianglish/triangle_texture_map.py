import cv2
import numpy
import math

# def merge_triangle_texture(images, blend_strength=5):
#     return images[0], [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]


def merge_triangle_texture(images, blend_strength=5):
    seam_work_aspect = 1
    image_sizes = []
    features = []

    # Feature detection
    # finder = cv2.xfeatures2d_SIFT.create()
    finder = cv2.xfeatures2d.SIFT_create(nfeatures=500,
                                         nOctaveLayers=3,
                                         contrastThreshold=0.001,
                                         edgeThreshold=100,
                                         sigma=1.6)
    for image in images:
        image_sizes.append((image.shape[1], image.shape[0]))
        image_feature = cv2.detail.computeImageFeatures2(finder, image)
        features.append(image_feature)

    # Feature matching
    matcher = cv2.detail_AffineBestOf2NearestMatcher(full_affine=False, try_use_gpu=True,
                                                     match_conf=0.3, num_matches_thresh1=6)
    matching_result = matcher.apply2(features)
    matcher.collectGarbage()

    # Transformation estimation
    estimator = cv2.detail_HomographyBasedEstimator()
    _, cameras = estimator.apply(features, matching_result, None)
    for camera in cameras:
        camera.R = camera.R.astype(numpy.float32)

    # Bundle adjustment
    adjuster = cv2.detail_BundleAdjusterRay()
    adjuster.setConfThresh(1)
    refine_mask = numpy.array([[1, 1, 1], [0, 1, 1], [0, 0, 0]], numpy.uint8)
    adjuster.setRefinementMask(refine_mask)
    _, cameras = adjuster.apply(features, matching_result, cameras)

    # camera focal estimation
    focals = []
    for cam in cameras:
        focals.append(cam.focal)
    sorted(focals)
    if len(focals) % 2 == 1:
        warped_image_scale = focals[len(focals) // 2]
    else:
        warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2

    # Merging ==================================================
    corners = []
    sizes = []
    masks = []
    masks_warped = []
    images_warped = []

    # make masks
    for image in images:
        h, w, c = image.shape
        mask = cv2.UMat(255 * numpy.ones((h, w), numpy.uint8))
        masks.append(mask)

    warper = cv2.PyRotationWarper("plane", warped_image_scale * seam_work_aspect)  # warper peut etre nullptr?
    for idx, image in enumerate(images):
        K = cameras[idx].K().astype(numpy.float32)
        swa = seam_work_aspect
        K[0, 0] *= swa
        K[0, 2] *= swa
        K[1, 1] *= swa
        K[1, 2] *= swa
        corner, image_wp = warper.warp(images[idx], K, cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        corners.append(corner)
        sizes.append((image_wp.shape[1], image_wp.shape[0]))
        images_warped.append(image_wp)

        _, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        masks_warped.append(mask_wp.get())

    compensator = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_NO)
    compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

    # Seam generation
    images_warped_float32 = []
    for image in images_warped:
        images_warped_float32.append(image.astype(numpy.float32))
    seam_finder = cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_NO)
    seam_finder.find(images_warped_float32, corners, masks_warped)

    # warping to find the final size
    warper = cv2.PyRotationWarper(type="plane", scale=warped_image_scale)
    for i, image in enumerate(images):
        K = cameras[i].K().astype(numpy.float32)
        roi = warper.warpRoi((image_sizes[i][0], image_sizes[i][1]), K, cameras[i].R)
        corners.append(roi[0:2])
        sizes.append(roi[2:4])
    dst_sz = cv2.detail.resultRoi(corners=corners, sizes=sizes)

    # Initialize MultiBandBlender
    blender = cv2.detail_MultiBandBlender()
    blend_width = numpy.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
    blender.setNumBands((numpy.log(blend_width) / numpy.log(2.) - 1.).astype(numpy.int))
    blender.prepare(dst_sz)

    warped_image_centers = []

    for idx, image in enumerate(images):
        K = cameras[idx].K().astype(numpy.float32)
        R = cameras[idx].R
        w = image.shape[1]
        h = image.shape[0]

        corner, image_warped = warper.warp(image, K, R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        mask = 255 * numpy.ones((h, w), numpy.uint8)
        matching_result, mask_warped = warper.warp(mask, K, R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        compensator.apply(idx, corners[idx], image_warped, mask_warped)
        image_warped_s = image_warped.astype(numpy.int16)
        dilated_mask = cv2.dilate(masks_warped[idx], None)
        seam_mask = cv2.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv2.INTER_LINEAR_EXACT)
        mask_warped = cv2.bitwise_and(seam_mask, mask_warped)

        blender.feed(cv2.UMat(image_warped_s), mask_warped, corners[idx])
        # generate texture coords
        img_center = ((w-1.0)/2, (h-1.0)/2)

        (dst_x, dst_y) = warper.warpPoint(pt=img_center, K=K, R=R)
        (warped_image_center_x, warped_image_center_y) = (dst_x - dst_sz[0], dst_y - dst_sz[1])

        warped_image_centers.append([warped_image_center_x, warped_image_center_y])

    # Merge images
    merged_texture = None
    merged_texture_mask = None
    merged_texture, merged_texture_mask = blender.blend(merged_texture, merged_texture_mask)

    left_top = numpy.amin(numpy.asarray(warped_image_centers), axis=0)
    right_bottom = numpy.amax(numpy.asarray(warped_image_centers), axis=0)

    cropped_texture = merged_texture[
                      math.floor(left_top[1]):math.ceil(right_bottom[1]),
                      math.floor(left_top[0]): math.ceil(right_bottom[0])
                      ]

    texture_coords = []
    for [x, y] in warped_image_centers:
        texture_coords.append(
            [
                (x - left_top[0]) / (right_bottom[0]-left_top[0]),
                1 - (y - left_top[1]) / (right_bottom[1]-left_top[1])
            ])
    # print(texture_coords)
    return cropped_texture, texture_coords



