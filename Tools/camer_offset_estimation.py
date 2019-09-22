import numpy
import transforms3d
import open3d

point_a = numpy.array([0.24603, -0.554, 0.001992])
point_b = numpy.array([0.25392, -0.2739, -0.000292])
point_c = numpy.array([0.41612, -0.55694, 0.00324])

pos_robotic_b = numpy.array([0.25924, -0.30146, 0.13012])
rot_robotic_b = numpy.array([-0.65414, -0.27075, -0.26636, 0.6541])

trans_robotic_b = transforms3d.affines.compose(T=pos_robotic_b, R=transforms3d.quaternions.quat2mat(rot_robotic_b),
                                               Z=[1, 1, 1], S=[0, 0, 0])

print(numpy.asarray(transforms3d.euler.quat2euler(rot_robotic_b))*180/3.1415926)

vector_ab = point_b - point_a
vector_ac = point_c - point_a

vector_ab = vector_ab/numpy.linalg.norm(vector_ab)
vector_ac = vector_ac/numpy.linalg.norm(vector_ac)

vector_x = vector_ac
vector_z = numpy.cross(vector_ac, vector_ab)
vector_y = numpy.cross(vector_z, vector_x)

print("vector_z: ")
print(vector_z)
print("vector_y: ")
print(vector_y)

trans_camera_focal_point = numpy.identity(4)
trans_camera_focal_point[0:3, 0] = vector_y.T
trans_camera_focal_point[0:3, 1] = vector_x.T * (-1)
trans_camera_focal_point[0:3, 2] = vector_z.T
trans_camera_focal_point[0:3, 3] = point_b.T

print("trans_camera_focal_point")
print(trans_camera_focal_point)

camera_offset_matrix = numpy.dot(numpy.linalg.inv(trans_robotic_b), trans_camera_focal_point)

print("camera_offset_matrix")
print(camera_offset_matrix)
