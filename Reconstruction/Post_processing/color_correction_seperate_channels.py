import numpy
import scipy.optimize

import TransformationData
import TileInfo
import TileInfoDict
import numba


def bgr_to_luminance(bgr_color):
    return 0.2126 * bgr_color[2] + 0.7152 * bgr_color[1] + 0.2126 * bgr_color[0]


def color_processing(tile_info_dict, trans_data_manager: TransformationData.TransformationDataPool):
    edges_count = 0
    edges_b = {}
    edges_g = {}
    edges_r = {}
    edges_l = {}

    tile_info_key_list = []
    tile_info_key_to_list = {}

    for tile_info_key in tile_info_dict:
        tile_info_key_list.append(tile_info_key)
        tile_info_key_to_list[tile_info_key] = tile_info_key_list.index(tile_info_key)

    for (s, t) in trans_data_manager.trans_dict:
        trans_estimation_st = trans_data_manager.trans_dict[(s, t)]
        if trans_estimation_st.success\
                and trans_estimation_st.overlapping_pixels > (tile_info_dict[s].width_by_pixel
                                                              * tile_info_dict[s].height_by_pixel
                                                              * 0.3):
            edges_count += 1
            mean_s_b = trans_estimation_st.mean_s[0]
            mean_t_b = trans_estimation_st.mean_t[0]
            mean_s_g = trans_estimation_st.mean_s[1]
            mean_t_g = trans_estimation_st.mean_t[1]
            mean_s_r = trans_estimation_st.mean_s[2]
            mean_t_r = trans_estimation_st.mean_t[2]

            mean_s_l = bgr_to_luminance(trans_estimation_st.mean_s)
            mean_t_l = bgr_to_luminance(trans_estimation_st.mean_t)

            edges_b[(s, t)] = [mean_s_b, mean_t_b, trans_estimation_st.overlapping_pixels]
            edges_g[(s, t)] = [mean_s_g, mean_t_g, trans_estimation_st.overlapping_pixels]
            edges_r[(s, t)] = [mean_s_r, mean_t_r, trans_estimation_st.overlapping_pixels]
            edges_l[(s, t)] = [mean_s_r, mean_t_r, trans_estimation_st.overlapping_pixels]

    return tile_info_key_list, tile_info_key_to_list, edges_count, edges_b, edges_g, edges_r, edges_l


def solve_single_channel_filter(tile_info_key_to_list, edges):
    def single_channel_equations(filters):
        equations = []
        for (s, t) in edges:
            [mean_s, mean_t, overlapping_pixels] = edges[(s, t)]

            eq_left = overlapping_pixels * (
                    pow(mean_s * filters[tile_info_key_to_list[s]] -
                        mean_t * filters[tile_info_key_to_list[t]], 2) / 100
                    + pow(1 - filters[tile_info_key_to_list[s]], 2) * 100
                )
            equations += [eq_left]
        return equations

    def single_channel_equations_dfunction(filters):
        d_funs = numpy.zeros((len(edges), len(tile_info_key_to_list)))
        local_edges_count = 0

        for (s, t) in edges:
            [mean_s, mean_t, overlapping_pixels] = edges[(s, t)]

            eq_left_s = overlapping_pixels * ((pow(mean_s, 2) * filters[tile_info_key_to_list[s]]
                                               - mean_s * mean_t * filters[tile_info_key_to_list[t]]) / 50
                                              + (1 - 2 * filters[tile_info_key_to_list[s]]) * 100)
            eq_left_t = overlapping_pixels * ((pow(mean_t, 2) * filters[tile_info_key_to_list[t]]
                                               - mean_s * mean_t * filters[tile_info_key_to_list[s]]) / 50)

            d_funs[local_edges_count][tile_info_key_to_list[s]] = eq_left_s
            d_funs[local_edges_count][tile_info_key_to_list[t]] = eq_left_t

            local_edges_count += 1
        return d_funs

    solve_result, _ = scipy.optimize.leastsq(func=single_channel_equations,
                                             x0=numpy.ones(len(tile_info_key_to_list)),
                                             Dfun=single_channel_equations_dfunction)
    single_channel_filters = numpy.asarray(solve_result).reshape((-1))
    return single_channel_filters


def generate_color_filters(tile_info_dict, trans_data_manager: TransformationData.TransformationDataPool):
    tile_info_key_list, tile_info_key_to_list, edges_count, edges_b, edges_g, edges_r, edges_l = \
        color_processing(tile_info_dict, trans_data_manager)

    r_filter = solve_single_channel_filter(tile_info_key_to_list, edges_r)
    g_filter = solve_single_channel_filter(tile_info_key_to_list, edges_r)
    b_filter = solve_single_channel_filter(tile_info_key_to_list, edges_r)

    print("Updating the color_filters to tile_info_dict")
    for tile_info_key in tile_info_dict:
        tile_info_dict[tile_info_key].color_and_illumination_correction = \
            numpy.array([r_filter[tile_info_key_to_list[tile_info_key]],
                         g_filter[tile_info_key_to_list[tile_info_key]],
                         b_filter[tile_info_key_to_list[tile_info_key]]])
    print("Updated")
    return tile_info_dict
