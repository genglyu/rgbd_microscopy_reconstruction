import sys
sys.path.append("../../Utility")
sys.path.append("../Alignment")
sys.path.append("../Data_processing")
from open3d import *
import numpy
import math
from scipy.spatial.transform import Rotation
import transforms3d

import open3d
import numpy
from collections import defaultdict
from open3d import *
from DataConvert import *

# Load a list of poses, convert to list of points. If distance between points is less than r, make a edge.
# Traversal by depth first order.

class NavigationGraph:
    def __init__(self):
        self.graph = defaultdict(list)

        self.pose_list = []
        self.points = []
        self.pcd = None
        self.kd_tree = None
        self.trans_list = []

        self.dfs_order_index_list = []
        self.bfs_order_index_list = []

    def load_trans_list(self, trans_list, connection_range):
        self.points = trans_list_to_points(trans_list)
        self.pcd = PointCloud()
        self.pcd.points = Vector3dVector(self.points)
        self.kd_tree = KDTreeFlann(self.pcd)

        print("len(self.points): ")
        print(len(self.points))

        [_, idx, _] = self.kd_tree.search_radius_vector_3d([0, 0, 0], numpy.linalg.norm(self.points[0]) + 1)
        new_trans_list = adjust_order(trans_list, idx)
        print(idx)

        self.points = trans_list_to_points(new_trans_list)
        self.pcd = PointCloud()
        self.pcd.points = Vector3dVector(self.points)
        self.kd_tree = KDTreeFlann(self.pcd)

        self.trans_list = new_trans_list

        for i, point in enumerate(self.points):
            [_, idx, _] = self.kd_tree.search_radius_vector_3d(point, connection_range)
            for j in idx:
                self.add_edge(i, j)

    # def load_trans_list(self, trans_list):
    #     point_list = []
    #     for trans in trans_list:
    #         point = numpy.dot(trans, numpy.asarray([0, 0, 0, 1]).T).T[0:3].tolist()
    #         point_list.append(point)
    #     self.load_point_list(point_list)
    #
    # def load_robotic_pose_list(self, robotic_pose_list):
    #     point_list = []
    #     for pose in robotic_pose_list:
    #         trans = rob_pose_to_trans(pose)
    #         point = numpy.dot(trans, numpy.asarray([0, 0, 0, 1]).T).T[0:3].tolist()
    #         point_list.append(point)
    #     self.load_point_list(point_list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def bfs(self, s=0):
        # Mark all the vertices as not visited
        visited = [False] * (len(self.graph))
        # Create a queue for BFS
        queue = []
        # Mark the source node as
        # visited and enqueue it
        queue.append(s)
        visited[s] = True

        while queue:
            # Dequeue a vertex from
            # queue and print it
            s = queue.pop(0)
            self.bfs_order_index_list.append(s)
            # print(s, end=" ")

            # Get all adjacent vertices of the
            # dequeued vertex s. If a adjacent
            # has not been visited, then mark it
            # visited and enqueue it
            for i in self.graph[s]:
                if not visited[i]:
                    queue.append(i)
                    visited[i] = True

        self.trans_list = adjust_order(self.trans_list, self.bfs_order_index_list)
        return self.trans_list

    def dfs_util(self, v, visited):
        visited[v] = True
        self.dfs_order_index_list.append(v)

        for i in self.graph[v]:
            if not visited[i]:
                self.dfs_util(i, visited)

    def dfs(self):
        node_amount = len(self.graph)
        visited =[False]*(node_amount)
        for i in range(node_amount):
            if not visited[i]:
                self.dfs_util(i, visited)
        self.trans_list = adjust_order(self.trans_list, self.dfs_order_index_list)
        return self.trans_list


