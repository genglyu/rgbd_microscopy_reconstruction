from panda3d.core import *
import color_coded_triangle_mesh
from direct.showbase.ShowBase import ShowBase
import cv2
from file_managing import *
import math
import numpy
import transforms3d
from scipy.spatial.transform import Rotation


class MicroscopyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.set_background_color(255, 255, 255)
        self.disable_mouse()

        window_settings = WindowProperties()
        window_settings.setSize(1080, 1080)
        self.win.requestProperties(window_settings)

        # Add global light
        alight = AmbientLight('alight')
        alight.setColor((1, 1, 1, 1))
        alight_in_render = self.render.attachNewNode(alight)
        self.render.setLight(alight_in_render)
        # self.render.setAntialias(AntialiasAttrib.MAuto)
        self.render.setAntialias(AntialiasAttrib.MNone)

        self.init_camera()

    def init_camera(self):
        self.disable_mouse()
        # self.camera.setPos(332.233, -430.504, 181.282)
        self.camera.setPos(295.154714427479, -179.06927339506692, 159.1025333387005)
        self.cam.node().getLens().setFov(30)
        self.camera.setHpr(0, -90, -90)

        mat = Mat4(self.camera.getMat())
        mat.invertInPlace()
        self.mouseInterfaceNode.setMat(mat)
        self.enableMouse()

    def init_virtual_microscope_camera(self, focal_distance_by_mm=50,
                                       tile_width_by_pixel=640, tile_height_by_pixel=480,
                                       tile_width_by_mm=9.5):
        window_settings = WindowProperties()
        window_settings.setSize(tile_width_by_pixel, tile_height_by_pixel)
        self.win.requestProperties(window_settings)
        self.disable_mouse()
        self.cam.node().getLens().setFov(math.degrees(math.atan(tile_width_by_mm/focal_distance_by_mm/2) * 2))
        self.cam.node().getLens().setNear(1)
        self.cam.node().getLens().setFar(10000)

    def set_virtual_camera_to_tile_pose(self, tile_pose=numpy.identity(4), focal_distance_by_mm=200):
        self.disable_mouse()
        camera_pose = numpy.dot(tile_pose, numpy.array([[0, -1, 0, focal_distance_by_mm],
                                                        [1, 0, 0, 0],
                                                        [0, 0, 1, 0],
                                                        [0, 0, 0, 1]]))
        # print("self.camera.getNetTransform().getMat()")
        # print(self.camera.getNetTransform().getMat())
        self.camera.setMat(Mat4(*camera_pose.T.flatten().tolist()))
        mat = Mat4(self.camera.getMat())
        mat.invertInPlace()
        self.mouseInterfaceNode.setMat(mat)
        self.enableMouse()

    def capture_rendered_image(self, save_path):
        self.graphicsEngine.renderFrame()
        self.screenshot(save_path, defaultFilename=0)
        return

    def add_triangle(self, triangle_info: color_coded_triangle_mesh.MicroscopyTriangleInfo, merged_texture_dir=""):
        vertex_format = GeomVertexFormat.getV3n3cpt2()
        vdata = GeomVertexData('square', vertex_format, Geom.UHDynamic)

        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')
        texcoord = GeomVertexWriter(vdata, 'texcoord')
        triangle_vertices = GeomTriangles(Geom.UHDynamic)

        for i, _ in enumerate(triangle_info.vertices):
            vertex.addData3(triangle_info.vertices[i][0], triangle_info.vertices[i][1], triangle_info.vertices[i][2])
            normal.addData3(triangle_info.vertex_normals[i][0],
                            triangle_info.vertex_normals[i][1],
                            triangle_info.vertex_normals[i][2])
            color.addData4i(255, 255, 255, 255)
            texcoord.addData2f(triangle_info.merged_texture_coords[i][0], triangle_info.merged_texture_coords[i][1])

        triangle_vertices.addVertices(0, 1, 2)

        triangle_mesh = Geom(vdata)
        triangle_mesh.addPrimitive(triangle_vertices)

        texture_cv = cv2.imread(join(merged_texture_dir, triangle_info.merged_texture_name))
        print(triangle_info.merged_texture_name)
        print(join(merged_texture_dir, triangle_info.merged_texture_name))

        w = texture_cv.shape[1]
        h = texture_cv.shape[0]
        texture_cv = cv2.flip(texture_cv, 0)

        texture = Texture()
        texture.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgb8)
        texture.setRamImage(texture_cv)

        # Add to render
        triangle_mesh_node = GeomNode("")
        triangle_mesh_node.addGeom(triangle_mesh)
        triangle_in_render = self.render.attachNewNode(triangle_mesh_node)
        triangle_in_render.setTwoSided(True)

        # Add texture
        triangle_in_render.setTexture(texture)

    def load_textured_triangles(self,
                                triangle_info_list,
                                merged_texture_dir=""):
        for triangle_info in triangle_info_list:
            # print(triangle_info.merged_texture_name)
            self.add_triangle(triangle_info=triangle_info, merged_texture_dir=merged_texture_dir)



