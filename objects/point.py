"""
Created by Neel Gokhale at 2020-08-30
File point.py from project Project_NN_From_Scratch
Built using PyCharm

"""

import numpy as np


class Point(object):

    def __init__(self, x, y, test_point=False):
        """
        2-dimensional assets point object. Can be assigned to a cluster, identified as a centroid or as a test point

        :param x: x-dimension
        :param y: y-dimension
        :param test_point: True if instance of Point is part of testing assets
        """
        self.x = x
        self.y = y
        self.is_centroid = False
        self.cluster_id = None
        self.test_point = test_point

    def distance_to(self, point2):
        """
        Calculates the distance between itself and a target point

        :param point2: target point
        :return: distance between point objects
        """
        return np.sqrt((self.x - point2.x) ** 2 + (self.y - point2.y) ** 2)

    def index_closest_centroid(self, centroid_list):
        """
        Returns the index of the smallest distance from a list of distances to centroids

        :param centroid_list: list of centroids
        :return: list index of minimum distanced centroid
        """
        if not self.is_centroid:
            dist_list = []
            for centroid in centroid_list:
                dist_list.append(self.distance_to(centroid))
            return dist_list.index(min(dist_list))

    def move_self(self, new_loc: tuple):
        """
        `Point` object can be moved to another location

        :param new_loc: tuple with new location (x, y)
        """
        if self.is_centroid:
            if new_loc == (None, None):
                self.x = self.x
                self.y = self.y
            else:
                self.x = new_loc[0]
                self.y = new_loc[1]