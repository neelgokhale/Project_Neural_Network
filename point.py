"""
Created by Neel Gokhale at 2020-08-30
File point.py from project Project_NN_From_Scratch
Built using PyCharm

"""

import numpy as np


class Point:

    def __init__(self, x, y, test_point=False):
        self.x = x
        self.y = y
        self.is_centroid = False
        self.cluster_id = None
        self.test_point = test_point

    def distance_to(self, point2):
        return np.sqrt((self.x - point2.x) ** 2 + (self.y - point2.y) ** 2)

    def index_closest_centroid(self, centroid_list):
        if not self.is_centroid:
            dist_list = []
            for centroid in centroid_list:
                dist_list.append(self.distance_to(centroid))
            return dist_list.index(min(dist_list))

    def move_self(self, new_loc: tuple):
        if self.is_centroid:
            self.x = new_loc[0]
            self.y = new_loc[1]