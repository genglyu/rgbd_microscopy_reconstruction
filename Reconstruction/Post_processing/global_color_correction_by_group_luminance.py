import numpy
import scipy.optimize
import math
import TransformationData
import TileInfo
import TileInfoDict
import numba


def bgr_to_luminance(bgr_color):
    return 0.2126 * bgr_color[2] + 0.7152 * bgr_color[1] + 0.2126 * bgr_color[0]


def group_tiles(tile_info_dict, volum_size_by_m=0.01):
    bounding_box = [[0, 0, 0], [0, 0, 0]]
    for tile_info_key in tile_info_dict:
        if bounding_box[0][0] > tile_info_dict[tile_info_key].position[0]:
            bounding_box[0][0] = tile_info_dict[tile_info_key].position[0] - 0.0001
        if bounding_box[1][0] < tile_info_dict[tile_info_key].position[0]:
            bounding_box[1][0] = tile_info_dict[tile_info_key].position[0] + 0.0001

        if bounding_box[0][1] > tile_info_dict[tile_info_key].position[1]:
            bounding_box[0][1] = tile_info_dict[tile_info_key].position[1] - 0.0001
        if bounding_box[1][1] < tile_info_dict[tile_info_key].position[1]:
            bounding_box[1][1] = tile_info_dict[tile_info_key].position[1] + 0.0001

        if bounding_box[0][2] > tile_info_dict[tile_info_key].position[2]:
            bounding_box[0][2] = tile_info_dict[tile_info_key].position[2] - 0.0001
        if bounding_box[1][2] < tile_info_dict[tile_info_key].position[2]:
            bounding_box[1][2] = tile_info_dict[tile_info_key].position[2] + 0.0001

    groups_grid_structure = []
    x_cells = math.ceil((bounding_box[1][0] - bounding_box[0][0]) / volum_size_by_m)
    y_cells = math.ceil((bounding_box[1][1] - bounding_box[0][1]) / volum_size_by_m)
    z_cells = math.ceil((bounding_box[1][2] - bounding_box[0][2]) / volum_size_by_m)
    for i in range(x_cells):
        groups_grid_structure.append([])
        for j in range(y_cells):
            groups_grid_structure[i].append([])
            for k in range(z_cells):
                groups_grid_structure[i][j].append([])

    for tile_info_key in tile_info_dict:
        tile_info = tile_info_dict[tile_info_key]
        group_index_x = math.floor((tile_info.position[0] - bounding_box[0][0]) / volum_size_by_m)
        group_index_y = math.floor((tile_info.position[1] - bounding_box[0][1]) / volum_size_by_m)
        group_index_z = math.floor((tile_info.position[2] - bounding_box[0][2]) / volum_size_by_m)
        groups_grid_structure[group_index_x][group_index_y][group_index_z].append(tile_info_key)

    groups = []
    for i in range(x_cells):
        for j in range(y_cells):
            for k in range(z_cells):
                if len(groups_grid_structure[i][j][k]) != 0:
                    groups.append(groups_grid_structure[i][j][k])
    return groups


def generate_color_filters_in_group(tile_index_in_group_list, tile_info_dict,
                                    trans_data_manager: TransformationData.TransformationDataPool):

    if len(tile_index_in_group_list)==1:
        return {tile_index_in_group_list[0]: [1.0, 1.0, 1.0]}

    if len(tile_index_in_group_list)==2:
        real_s = tile_index_in_group_list[0]
        real_t = tile_index_in_group_list[1]
        trans_estimation_st = trans_data_manager.get_trans(real_s, real_t)

        mean_s = bgr_to_luminance(trans_estimation_st.mean_s)
        mean_t = bgr_to_luminance(trans_estimation_st.mean_t)
        luminance_filter_s = (mean_s + mean_t) / 2 / mean_s
        luminance_filter_t = (mean_s + mean_t) / 2 / mean_t

        filter_s = numpy.asarray([luminance_filter_s, luminance_filter_s, luminance_filter_s])
        filter_t = numpy.asarray([luminance_filter_t, luminance_filter_t, luminance_filter_t])
        return {real_s: filter_s,
                real_t: filter_t}
    else:
        edges_count = 0
        edges_in_group = []
        tile_index_to_list_dict = {}
        for real_s in tile_index_in_group_list:

            tile_info = tile_info_dict[real_s]
            tile_index_to_list_dict[real_s] = tile_index_in_group_list.index(real_s)

            # confirmed_neighbours_in_group = []
            for real_t in tile_info.confirmed_neighbour_list:
                if real_t in tile_index_in_group_list and real_s < real_t:
                    # confirmed_neighbours_in_group.append(real_t)
                    trans_estimation = trans_data_manager.get_trans(real_s, real_t)
                    if trans_estimation.success and trans_estimation.overlapping_pixels > (640 * 480 * 0.2):
                        edges_count += 1
                        edges_in_group.append((real_s, real_t))

        def color_equations(luminance_filters):
            equations = []
            for (real_s, real_t) in edges_in_group:
                trans_estimation_st = trans_data_manager.get_trans(real_s, real_t)
                s = tile_index_to_list_dict[real_s]
                t = tile_index_to_list_dict[real_t]

                if trans_estimation_st.success and trans_estimation_st.overlapping_pixels > (640 * 480 * 0.2):
                    mean_s = bgr_to_luminance(trans_estimation_st.mean_s)
                    mean_t = bgr_to_luminance(trans_estimation_st.mean_t)

                    eq_left = pow(mean_s * luminance_filters[s] - mean_t * luminance_filters[t], 2) / 100 + pow(1 - luminance_filters[s], 2) * 100
                    equations += [eq_left]
            return equations

        def color_equations_dfunction(luminance_filters):
            d_funs = numpy.zeros((edges_count, len(tile_index_in_group_list)))
            local_edges_count = 0

            # for (s, t) in trans_data_manager.trans_dict:
            #     trans_estimation_st = trans_data_manager.get_trans(s, t)
            for (real_s, real_t) in edges_in_group:
                trans_estimation_st = trans_data_manager.get_trans(real_s, real_t)
                s = tile_index_to_list_dict[real_s]
                t = tile_index_to_list_dict[real_t]
                if trans_estimation_st.success and trans_estimation_st.overlapping_pixels > (640 * 480 * 0.2):
                    mean_s = bgr_to_luminance(trans_estimation_st.mean_s)
                    mean_t = bgr_to_luminance(trans_estimation_st.mean_t)

                    eq_left_s = ((pow(mean_s, 2) * luminance_filters[s] - mean_s * mean_t * luminance_filters[t]) / 50 + (luminance_filters[s] - 1) * 50)
                    eq_left_t = (pow(mean_t, 2) * luminance_filters[t] - mean_s * mean_t * luminance_filters[s]) / 50

                    d_funs[local_edges_count][s] = eq_left_s
                    d_funs[local_edges_count][t] = eq_left_t

                    local_edges_count += 1
            return d_funs
        solve_result, _ = scipy.optimize.leastsq(func=color_equations, x0=numpy.ones(len(tile_index_in_group_list)), Dfun=color_equations_dfunction)
        color_filters_in_group = numpy.asarray(solve_result).reshape((-1))

        color_filters_dict = {}
        for i, tile_index in enumerate(tile_index_in_group_list):
            color_filters_dict[tile_index] = [color_filters_in_group[i], color_filters_in_group[i], color_filters_in_group[i]]
        return color_filters_dict
# ===================================================================================================================


def check_group_connection(group_source_index_list, group_target_index_list, tile_info_dict):
    edges_between_groups = []
    for tile_info_key in group_source_index_list:
        tile_info = tile_info_dict[tile_info_key]
        for confirmed_neighbours_key in tile_info.confirmed_neighbour_list:
            if confirmed_neighbours_key in group_target_index_list:
                if tile_info_key < confirmed_neighbours_key:
                    edges_between_groups.append((tile_info_key, confirmed_neighbours_key))
                else:
                    edges_between_groups.append((confirmed_neighbours_key, tile_info_key))
    connected = (len(edges_between_groups) != 0)
    return connected, edges_between_groups

# ===================================================================================================================


def generate_color_filter_between_groups(groups, tile_info_dict,
                                         trans_data_manager: TransformationData.TransformationDataPool):
    group_color_filters = []
    if len(groups) == 1:
        group_color_filters = [[1.0, 1.0, 1.0]]
    if len(groups) == 2:
        color_filter_dict = {}
        group_edges = {}
        group_edges_with_mean = {}
        for group_s_id, group_s in enumerate(groups):
            color_filter_dict_in_group = generate_color_filters_in_group(group_s, tile_info_dict, trans_data_manager)
            color_filter_dict.update(color_filter_dict_in_group)

        group_s = groups[0]
        group_t = groups[1]
        group_s_mean_sum = [0, 0, 0]
        group_s_mean_pixels = 0
        group_t_mean_sum = [0, 0, 0]
        group_t_mean_pixels = 0

        connected, edges_between_groups = check_group_connection(group_s, group_t, tile_info_dict)
        if connected:
            for (s, t) in edges_between_groups:
                trans_estimation_st = trans_data_manager.get_trans(s, t)
                if s in group_s:
                    group_s_mean_sum += trans_estimation_st.mean_s * color_filter_dict[
                        s] * trans_estimation_st.overlapping_pixels
                    group_s_mean_pixels += trans_estimation_st.overlapping_pixels
                if s in group_t:
                    group_t_mean_sum += trans_estimation_st.mean_s * color_filter_dict[
                        s] * trans_estimation_st.overlapping_pixels
                    group_t_mean_pixels += trans_estimation_st.overlapping_pixels

                if t in group_s:
                    group_s_mean_sum += trans_estimation_st.mean_t * color_filter_dict[
                        t] * trans_estimation_st.overlapping_pixels
                    group_s_mean_pixels += trans_estimation_st.overlapping_pixels
                if t in group_t:
                    group_t_mean_sum += trans_estimation_st.mean_t * color_filter_dict[
                        t] * trans_estimation_st.overlapping_pixels
                    group_t_mean_pixels += trans_estimation_st.overlapping_pixels
            group_s_mean = group_s_mean_sum / group_s_mean_pixels
            group_t_mean = group_t_mean_sum / group_t_mean_pixels

            mean_s = bgr_to_luminance(group_s_mean)
            mean_t = bgr_to_luminance(group_t_mean)
            luminance_filter_s = (mean_s + mean_t) / 2 / mean_s
            luminance_filter_t = (mean_s + mean_t) / 2 / mean_t

            filter_s = numpy.asarray([luminance_filter_s, luminance_filter_s, luminance_filter_s])
            filter_t = numpy.asarray([luminance_filter_t, luminance_filter_t, luminance_filter_t])
        group_color_filters = [filter_s, filter_t]
    else:
        color_filter_dict = {}
        group_edges = {}
        group_edges_with_mean = {}
        for group_s_id, group_s in enumerate(groups):
            color_filter_dict_in_group = generate_color_filters_in_group(group_s, tile_info_dict, trans_data_manager)
            color_filter_dict.update(color_filter_dict_in_group)
            if group_s_id < len(groups) -1:
                for group_t_id in range(group_s_id+1, len(groups)):
                    group_t = groups[group_t_id]
                    connected, edges_between_groups = check_group_connection(group_s, group_t, tile_info_dict)
                    if connected:
                        group_edges[(group_s_id, group_t_id)] = edges_between_groups
        # generate_group means
        for (group_s_id, group_t_id) in group_edges:
            real_edges = group_edges[(group_s_id, group_t_id)]
            group_s = groups[group_s_id]
            group_t = groups[group_t_id]
            group_s_mean_sum = [0, 0, 0]
            group_s_mean_pixels = 0
            group_t_mean_sum = [0, 0, 0]
            group_t_mean_pixels = 0
            for (s, t) in real_edges:
                trans_estimation_st = trans_data_manager.get_trans(s, t)
                if s in group_s:
                    group_s_mean_sum += trans_estimation_st.mean_s * color_filter_dict[s] * trans_estimation_st.overlapping_pixels
                    group_s_mean_pixels += trans_estimation_st.overlapping_pixels
                if s in group_t:
                    group_t_mean_sum += trans_estimation_st.mean_s * color_filter_dict[s] * trans_estimation_st.overlapping_pixels
                    group_t_mean_pixels += trans_estimation_st.overlapping_pixels

                if t in group_s:
                    group_s_mean_sum += trans_estimation_st.mean_t * color_filter_dict[t] * trans_estimation_st.overlapping_pixels
                    group_s_mean_pixels += trans_estimation_st.overlapping_pixels
                if t in group_t:
                    group_t_mean_sum += trans_estimation_st.mean_t * color_filter_dict[t] * trans_estimation_st.overlapping_pixels
                    group_t_mean_pixels += trans_estimation_st.overlapping_pixels
            group_s_mean = group_s_mean_sum / group_s_mean_pixels
            group_t_mean = group_t_mean_sum / group_t_mean_pixels
            group_edges_with_mean[(group_s_id, group_t_id)] = {"mean_s": group_s_mean,
                                                               "mean_t": group_t_mean}

        #  Solve the color filter between groups =============================================
        edges_count = len(group_edges_with_mean)
        # print("edges_count")
        print(groups)
        print(group_edges)

        def group_color_equations(luminance_filters):
            equations = []
            for (s, t) in group_edges_with_mean:
                mean_s = bgr_to_luminance(group_edges_with_mean[(s, t)]["mean_s"])
                mean_t = bgr_to_luminance(group_edges_with_mean[(s, t)]["mean_t"])

                eq_left = pow(mean_s * luminance_filters[s] - mean_t * luminance_filters[t], 2) / 100 \
                          + pow(1 - luminance_filters[s], 2) * 100
                equations += [eq_left]
            return equations

        def group_color_equations_dfunction(luminance_filters):
            d_funs = numpy.zeros((edges_count, len(groups)))
            local_edges_count = 0
            for (s, t) in group_edges_with_mean:
                mean_s = bgr_to_luminance(group_edges_with_mean[(s, t)]["mean_s"])
                mean_t = bgr_to_luminance(group_edges_with_mean[(s, t)]["mean_t"])

                eq_left_s = (pow(mean_s, 2) * luminance_filters[s] - mean_s * mean_t * luminance_filters[t]) / 50 \
                            + (luminance_filters[s] - 1) * 50
                eq_left_t = (pow(mean_t, 2) * luminance_filters[t]
                             - mean_s * mean_t * luminance_filters[s]) / 50
                d_funs[local_edges_count][s] = eq_left_s
                d_funs[local_edges_count][t] = eq_left_t

                local_edges_count += 1
            return d_funs

            # solve_result = scipy.optimize.fsolve(color_equations, numpy.ones(egdes_count * 3))

        solve_result, _ = scipy.optimize.leastsq(func=group_color_equations, x0=numpy.ones(len(groups)),
                                                 Dfun=group_color_equations_dfunction)
        group_luminance_filters = numpy.asarray(solve_result).reshape((-1))

    # print(group_color_filters)

    for group_id, group_index_list in enumerate(groups):
        for tile_key in group_index_list:
            print("group_luminance_filters[group_id]")
            print(group_luminance_filters[group_id])
            print(color_filter_dict[tile_key])
            color_filter_dict[tile_key] = color_filter_dict[tile_key] * numpy.array([group_luminance_filters[group_id],
                                                                                    group_luminance_filters[group_id],
                                                                                    group_luminance_filters[group_id]])

    return color_filter_dict, group_color_filters


def generate_color_filters(tile_info_dict, trans_data_manager: TransformationData.TransformationDataPool,
                           volum_size_by_m=0.01):
    groups = group_tiles(tile_info_dict, volum_size_by_m=volum_size_by_m)

    color_filter_dict, group_color_filters = \
        generate_color_filter_between_groups(groups, tile_info_dict, trans_data_manager)

    print("Updating the color_filters to tile_info_dict")
    for tile_info_key in color_filter_dict:
        tile_info_dict[tile_info_key].color_and_illumination_correction = color_filter_dict[tile_info_key]
    print("Updated")

    return tile_info_dict











