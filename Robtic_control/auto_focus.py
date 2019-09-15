import cv2
import numpy


class AutoFocusingController:
    def __init__(self):
        self.init_pose = numpy.identity(4)
        self.current_pose = numpy.identity(4)

        self.known_pose_lapalacian_list = []
        # might need to normalize the camera position to a one dimension value
        self.normalized_position_lapalacian_list = []

    def suggest_sampling_pose(self):
        return self.init_pose

