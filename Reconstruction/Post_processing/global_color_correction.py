import numpy
import scipy.optimize

import TransformationData
import TileInfo
import TileInfoDict
import numba


def generate_color_filters(tile_info_dict, trans_data_manager: TransformationData.TransformationDataPool):
    edges_count = 0
    for (s, t) in trans_data_manager.trans_dict:
        trans_estimation = trans_data_manager.trans_dict[(s, t)]
        if trans_estimation.success and trans_estimation.overlapping_pixels > (tile_info_dict[s].width_by_pixel * tile_info_dict[s].height_by_pixel * 0.3):
            edges_count += 1

    print("Preparing to process for %d equations" % (3 * edges_count))

    def color_equations(rgb_color_filters):
        equations = []
        for (s, t) in trans_data_manager.trans_dict:
            trans_estimation_st = trans_data_manager.trans_dict[(s, t)]
            if trans_estimation_st.success and trans_estimation_st.overlapping_pixels > (tile_info_dict[s].width_by_pixel * tile_info_dict[s].height_by_pixel * 0.3):
                mean_s = trans_estimation_st.mean_s
                mean_t = trans_estimation_st.mean_t

                eq_left_b = (pow(mean_s[0] * rgb_color_filters[s * 3 + 2] -
                                 mean_t[0] * rgb_color_filters[t * 3 + 2], 2) / 100 + pow(1 - rgb_color_filters[s * 3 + 2], 2) * 100) * 0.0722
                eq_left_g = (pow(mean_s[1] * rgb_color_filters[s * 3 + 1] -
                                 mean_t[1] * rgb_color_filters[t * 3 + 1], 2) / 100 + pow(1 - rgb_color_filters[s * 3 + 1], 2) * 100) * 0.7152
                eq_left_r = (pow(mean_s[2] * rgb_color_filters[s * 3 + 0] -
                                 mean_t[2] * rgb_color_filters[t * 3 + 0], 2) / 100 + pow(1 - rgb_color_filters[s * 3 + 0], 2) * 100) * 0.2126
                equations += [eq_left_b, eq_left_g, eq_left_r]
        return equations

    def color_equations_dfunction(rgb_color_filters):

        d_funs = numpy.zeros((edges_count * 3, len(tile_info_dict) * 3))

        local_edges_count = 0
        for (s, t) in trans_data_manager.trans_dict:
            trans_estimation_st = trans_data_manager.trans_dict[(s, t)]
            if trans_estimation_st.success and trans_estimation_st.overlapping_pixels > (640 * 480 * 0.2):
                mean_s = trans_estimation_st.mean_s
                mean_t = trans_estimation_st.mean_t
                eq_left_b_s = ((pow(mean_s[0], 2) * rgb_color_filters[s * 3 + 2] - mean_s[0] * mean_t[0] * rgb_color_filters[t * 3 + 2]) / 50 + (rgb_color_filters[s * 3 + 2] - 1) * 50) * 0.0722
                eq_left_b_t = (pow(mean_t[0], 2) * rgb_color_filters[t * 3 + 2] - mean_s[0] * mean_t[0] * rgb_color_filters[s * 3 + 2]) / 50 * 0.0722

                eq_left_g_s = ((pow(mean_s[1], 2) * rgb_color_filters[s * 3 + 1] - mean_s[1] * mean_t[1] * rgb_color_filters[t * 3 + 1]) / 50 + (rgb_color_filters[s * 3 + 1] - 1) * 50) * 0.7152
                eq_left_g_t = (pow(mean_t[1], 2) * rgb_color_filters[t * 3 + 1] - mean_s[1] * mean_t[1] * rgb_color_filters[s * 3 + 1]) / 50 * 0.7152

                eq_left_r_s = ((pow(mean_s[2], 2) * rgb_color_filters[s * 3 + 0] - mean_s[2] * mean_t[2] * rgb_color_filters[t * 3 + 0]) / 50 + (rgb_color_filters[s * 3 + 0] - 1) * 50) * 0.2126
                eq_left_r_t = (pow(mean_t[2], 2) * rgb_color_filters[t * 3 + 0] - mean_s[2] * mean_t[2] * rgb_color_filters[s * 3 + 0]) / 50 * 0.2126

                d_funs[local_edges_count][s * 3 + 2] = eq_left_b_s
                d_funs[local_edges_count][t * 3 + 2] = eq_left_b_t

                d_funs[local_edges_count][s * 3 + 1] = eq_left_g_s
                d_funs[local_edges_count][t * 3 + 1] = eq_left_g_t

                d_funs[local_edges_count][s * 3 + 0] = eq_left_r_s
                d_funs[local_edges_count][t * 3 + 0] = eq_left_r_t

                local_edges_count += 1
        return d_funs

    # solve_result = scipy.optimize.fsolve(color_equations, numpy.ones(egdes_count * 3))
    solve_result, _ = scipy.optimize.leastsq(func=color_equations, x0=numpy.ones(len(tile_info_dict) * 3), Dfun=color_equations_dfunction)
    color_filters = numpy.asarray(solve_result).reshape((-1, 3))

    print("Updating the color_filters to tile_info_dict")
    for tile_info_key in tile_info_dict:
        tile_info_dict[tile_info_key].color_and_illumination_correction = color_filters[tile_info_key]
        # print("Color filter for tile %6d" % tile_info_key)
        # print(color_filters[tile_info_key])
    print("Updated")

    return tile_info_dict
