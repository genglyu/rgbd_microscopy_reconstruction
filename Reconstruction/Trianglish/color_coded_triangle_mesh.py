import TileInfo
import TileInfoDict
import open3d
import numpy
import plyfile
import triangle_texture_map
import file_managing
import cv2
import json


def generate_vertex_triangle_mesh_from_tile_info_dict(tile_info_dict, radii_list=[2]):
    vertices = []
    normals = []
    encoded_colors = []

    for tile_info_key in tile_info_dict:
        tile_pose_matrix = tile_info_dict[tile_info_key].pose_matrix
        color_code = [tile_info_dict[tile_info_key].tile_group / 255,
                      tile_info_dict[tile_info_key].tile_index % 10000 // 255 / 255,
                      tile_info_dict[tile_info_key].tile_index % 10000 % 255 / 255]

        vertices.append(numpy.dot(tile_pose_matrix, numpy.asarray([0, 0, 0, 1]).T).T[0:3])
        normals.append(numpy.dot(tile_pose_matrix, numpy.asarray([1, 0, 0, 0]).T).T[0:3])
        encoded_colors.append(color_code)

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(vertices)
    pcd.normals = open3d.utility.Vector3dVector(normals)
    pcd.colors = open3d.utility.Vector3dVector(encoded_colors)

    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd=pcd,
        radii=open3d.utility.DoubleVector(radii_list))

    return pcd, mesh


class MicroscopyTriangleInfo:
    def __init__(self):
        self.tile_ids = []  # The tile_id remains the same in tile_info_dict_all and all the group dicts.

        self.vertices = []
        self.vertex_normals = []

        self.merged_texture_name = ""
        self.merged_texture_coords = []


def process_color_coded_ply(ply_path, tile_info_dict,
                            path_data="",
                            dataset_folder_template="Dataset_%02d",
                            path_image_dir="color/",
                            merged_texture_dir="textures/",
                            merged_texture_file_name_template="tex_%07d.png"):
    triangle_infos = []
    ply_data = plyfile.PlyData.read(ply_path)

    # triangle_amount = len(ply_data['face'])
    for data in ply_data:
        print(data)
    print(ply_data['face'][492])

    for triangle_id, _ in enumerate(ply_data['face']):
        try:
            new_triangle_info = MicroscopyTriangleInfo()
            vertex_tile_ids = []
            vertex_texture_list = []

            vertex_ids, = ply_data['face'][triangle_id]

            for vertex_id in vertex_ids:
                (x, y, z, nx, ny, nz, group_id, tile_index_high, tile_index_low) = ply_data['vertex'][vertex_id]
                tile_index = group_id * 10000 + tile_index_high * 255 + tile_index_low
                vertex_tile_ids.append(tile_index)

                tile_info = tile_info_dict[tile_index]
                tile_pose_matrix = tile_info.pose_matrix

                # vertex_poses.append(tile_pose_matrix)

                new_triangle_info.tile_ids.append(int(tile_index))
                new_triangle_info.vertices.append(
                    numpy.dot(tile_pose_matrix, numpy.asarray([0, 0, 0, 1]).T).T[0:3].tolist())
                new_triangle_info.vertex_normals.append(
                    numpy.dot(tile_pose_matrix, numpy.asarray([1, 0, 0, 0]).T).T[0:3].tolist())

                # generating texture
                image_path = file_managing.join(path_data, dataset_folder_template % group_id,
                                                path_image_dir, tile_info.file_name) + ".png"

                image = (cv2.imread(image_path) * numpy.flip(tile_info.color_and_illumination_correction, 0)).astype(numpy.uint8)
                vertex_texture_list.append(image)

            print("Processing triangle %d. The tiles involved are (%d, %d, %d)" % (triangle_id,
                                                                                   vertex_tile_ids[0],
                                                                                   vertex_tile_ids[1],
                                                                                   vertex_tile_ids[2]))
            # print("Processing triangle %d / %d" % (triangle_id, triangle_amount))

            merged_texture, new_triangle_info.merged_texture_coords = \
                triangle_texture_map.merge_triangle_texture(images=vertex_texture_list)
            new_triangle_info.merged_texture_name = merged_texture_file_name_template % triangle_id

            file_managing.touch_folder(file_managing.join(path_data, merged_texture_dir))
            cv2.imwrite(filename=file_managing.join(path_data, merged_texture_dir, new_triangle_info.merged_texture_name),
                        img=merged_texture)

            triangle_infos.append(new_triangle_info)
        except:
            print("Failed for triangle %d" % triangle_id)
            continue
        # print(new_triangle_info.merged_texture_name)
    return triangle_infos


def save_triangle_info_list(triangle_info_list, save_path):
    data_to_save = []
    for triangle_info in triangle_info_list:
        triangle_info_json_format = {
                "tile_ids": triangle_info.tile_ids,
                "vertices": triangle_info.vertices,
                "vertex_normals": triangle_info.vertex_normals,
                "merged_texture_name": triangle_info.merged_texture_name,
                "merged_texture_coords": triangle_info.merged_texture_coords
            }
        print(triangle_info_json_format)
        data_to_save.append(triangle_info_json_format)
    json.dump(obj=data_to_save, fp=open(save_path, "w"), indent=4)


def read_triangle_info_list(path):
    triangle_info_list_json_format = json.load(open(path, "r"))
    triangle_infos = []
    for triangle_info_json_format in triangle_info_list_json_format:
        new_triangle_info = MicroscopyTriangleInfo()
        new_triangle_info.tile_ids = triangle_info_json_format["tile_ids"]
        new_triangle_info.vertices = triangle_info_json_format["vertices"]
        new_triangle_info.vertex_normals = triangle_info_json_format["vertex_normals"]
        new_triangle_info.merged_texture_name = triangle_info_json_format["merged_texture_name"]
        new_triangle_info.merged_texture_coords = triangle_info_json_format["merged_texture_coords"]

        triangle_infos.append(new_triangle_info)
    return triangle_infos


