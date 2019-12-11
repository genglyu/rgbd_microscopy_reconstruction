import numpy
import scipy.optimize

import TransformationData
import TileInfo
import TileInfoDict
import numba


def bgr_to_luminance(bgr_color):
    return 0.2126 * bgr_color[2] + 0.7152 * bgr_color[1] + 0.2126 * bgr_color[0]


def generate_color_filters(tile_info_dict, trans_data_manager: TransformationData.TransformationDataPool):
    edges_count = 0
    for (s, t) in trans_data_manager.trans_dict:
        print("(s, t)")
        print((s, t))
        trans_estimation = trans_data_manager.trans_dict[(s, t)]
        if trans_estimation.success \
                and trans_estimation.overlapping_pixels > (tile_info_dict[s].width_by_pixel * tile_info_dict[s].height_by_pixel * 0.3):
            edges_count += 1

    tile_info_key_list = []
    tile_info_key_to_list = {}
    for tile_info_key in tile_info_dict:
        tile_info_key_list.append(tile_info_key)
        tile_info_key_to_list[tile_info_key] = tile_info_key_list.index(tile_info_key)

    print("Preparing to process for %d equations" % (edges_count))

    def color_equations(luminance_filters):
        equations = []
        for (s, t) in trans_data_manager.trans_dict:
            trans_estimation_st = trans_data_manager.trans_dict[(s, t)]
            if trans_estimation_st.success and trans_estimation_st.overlapping_pixels > (tile_info_dict[s].width_by_pixel * tile_info_dict[s].height_by_pixel * 0.3):
                mean_s = bgr_to_luminance(trans_estimation_st.mean_s)
                mean_t = bgr_to_luminance(trans_estimation_st.mean_t)

                eq_left = (pow(mean_s * luminance_filters[tile_info_key_to_list[s]] -
                                 mean_t * luminance_filters[tile_info_key_to_list[t]], 2) / 100 + pow(1 - luminance_filters[tile_info_key_to_list[s]], 2) * 100)
                equations += [eq_left]
        return equations

    def color_equations_dfunction(luminance_filters):
        d_funs = numpy.zeros((edges_count, len(tile_info_dict)))
        local_edges_count = 0
        for (s, t) in trans_data_manager.trans_dict:
            trans_estimation_st = trans_data_manager.trans_dict[(s, t)]
            if trans_estimation_st.success and trans_estimation_st.overlapping_pixels > (tile_info_dict[s].width_by_pixel * tile_info_dict[s].height_by_pixel * 0.3):
                mean_s = bgr_to_luminance(trans_estimation_st.mean_s)
                mean_t = bgr_to_luminance(trans_estimation_st.mean_t)

                eq_left_s = ((pow(mean_s, 2) * luminance_filters[tile_info_key_to_list[s]] - mean_s * mean_t * luminance_filters[tile_info_key_to_list[t]]) / 50 + (luminance_filters[tile_info_key_to_list[s]] - 1) * 50)
                eq_left_t = (pow(mean_t, 2) * luminance_filters[tile_info_key_to_list[t]] - mean_s * mean_t * luminance_filters[tile_info_key_to_list[s]]) / 50

                d_funs[local_edges_count][tile_info_key_to_list[s]] = eq_left_s
                d_funs[local_edges_count][tile_info_key_to_list[t]] = eq_left_t

                local_edges_count += 1
        return d_funs

    # solve_result = scipy.optimize.fsolve(color_equations, numpy.ones(egdes_count * 3))
    solve_result, _ = scipy.optimize.leastsq(func=color_equations, x0=numpy.ones(len(tile_info_dict)), Dfun=color_equations_dfunction)
    color_filters = numpy.asarray(solve_result).reshape((-1))

    print("Updating the color_filters to tile_info_dict")
    for tile_info_key in tile_info_dict:
        tile_info_dict[tile_info_key].color_and_illumination_correction = \
            numpy.array([color_filters[tile_info_key_to_list[tile_info_key]], color_filters[tile_info_key_to_list[tile_info_key]], color_filters[tile_info_key_to_list[tile_info_key]]])
        # print("Color filter for tile %6d" % tile_info_key)
        # print(color_filters[tile_info_key])
    print("Updated")

    return tile_info_dict
