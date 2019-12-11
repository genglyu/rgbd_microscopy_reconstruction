import open3d
import file_managing
import numpy
import visualize_textured_mesh_panda3d
import cv2
import pandas
import matplotlib.pyplot as plt


def draw_histogram(data_list):
    commutes = pandas.Series(data_list)
    commutes.plot.hist(grid=True,
                       # bins=20, rwidth=0.9,
                       color='#607c8e')
    plt.title('Commute Times for 1,000 Commuters')
    plt.xlabel('Counts')
    plt.ylabel('Commute Time')
    plt.grid(axis='y', alpha=0.75)



def evaluate_similarity(tile_info_dict,
                        path_data="",
                        dataset_folder_template="Dataset_%02d/",
                        path_image_dir="color/",
                        recaptured_tile_dir="captured/",
                        recaptured_tile_template="tile_%06d.png"):
    sift_finder = cv2.xfeatures2d.SIFT_create(nfeatures=500,
                                              nOctaveLayers=3,
                                              contrastThreshold=0.001,
                                              edgeThreshold=100,
                                              sigma=1.6)
    matcher = cv2.detail_AffineBestOf2NearestMatcher(full_affine=False, try_use_gpu=True,
                                                     match_conf=0.3,
                                                     num_matches_thresh1=6)
    matching_confidence_list = []
    for tile_info_key in tile_info_dict:
        original_image = cv2.imread(
            file_managing.join(path_data,
                               dataset_folder_template % tile_info_dict[tile_info_key].tile_group,
                               path_image_dir,
                               tile_info_dict[tile_info_key].file_name) + ".png")
        recaptured_image = cv2.imread(
            file_managing.join(path_data,
                               recaptured_tile_dir,
                               recaptured_tile_template % tile_info_key))

        source_feature = cv2.detail.computeImageFeatures2(featuresFinder=sift_finder, image=original_image)
        target_feature = cv2.detail.computeImageFeatures2(featuresFinder=sift_finder, image=recaptured_image)
        matching_result = matcher.apply(source_feature, target_feature)
        matcher.collectGarbage()
        match_conf = matching_result.confidence
        print("Tile %d confidence: %f" % (tile_info_key, match_conf))
        matching_confidence_list.append(match_conf)
    print(matching_confidence_list)
    draw_histogram(matching_confidence_list)