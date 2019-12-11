import open3d

# path = "/home/lvgeng/writing/microscope_project/fruit/fragments/tile_group_00.ply"
# mesh_path = "/home/lvgeng/writing/microscope_project/fruit/fragments/tile_group_00_mesh.ply"
# pcd = open3d.io.read_point_cloud(filename=path, remove_nan_points=True, remove_infinite_points=True, print_progress=True)
#
# open3d.visualization.draw_geometries([pcd])
#
# mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     pcd=pcd,
#     radii=open3d.utility.DoubleVector([3]))
#
# open3d.visualization.draw_geometries([mesh])
#
# open3d.io.write_triangle_mesh(filename=mesh_path, mesh=mesh, write_ascii=True, compressed=False,
#                               write_vertex_normals=True,
#                               write_vertex_colors=True,
#                               write_triangle_uvs=True,
#                               print_progress=True)

import TileInfoDict
import color_coded_triangle_mesh
import open3d


dict_path = "/home/lvgeng/writing/microscope_project/fruit/tile_info_dict.json"
mesh_path = "/home/lvgeng/writing/microscope_project/fruit/tile_all_mesh.ply"

tile_info_dict = TileInfoDict.read_tile_info_dict(dict_path)

pcd, mesh = color_coded_triangle_mesh.generate_vertex_triangle_mesh_from_tile_info_dict(tile_info_dict, [5])

open3d.io.write_triangle_mesh(filename=mesh_path, mesh=mesh, write_ascii=True, compressed=False,
                              write_vertex_normals=True,
                              write_vertex_colors=True,
                              write_triangle_uvs=True,
                              print_progress=True)

open3d.visualization.draw_geometries([mesh])


