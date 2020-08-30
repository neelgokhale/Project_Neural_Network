"""
Created by Neel Gokhale at 2020-08-28
File clustering_model.py from project Project_NN_From_Scratch
Built using PyCharm

"""

import matplotlib.pyplot as plt
import numpy as np
import random


class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.is_centroid = False
        self.cluster_id = None

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


def graph_data(point_list: list):
    color_list = ['green', 'orange', 'purple', 'pink', 'yellow', 'lightblue', 'brown']
    for point in point_list:
        if point.is_centroid:
            color = 'r'
            marker = 'D'
        elif point.cluster_id is not None:
            color = color_list[point.cluster_id]
            marker = '.'
        plt.scatter(point.x, point.y, c=color, marker=marker)
    plt.show()


def generate_rand_data(num_points: int, range_x: float = 1, range_y: float = 1, graph: bool = False):
    point_list = []
    for i in range(num_points):
        point_list.append(Point(random.random() * range_x,
                                random.random() * range_y))
    return point_list


def generate_centroids(num_centroids: int, num_points: int, point_list: list):
    centroid_list = []
    excluding = []
    for i in range(num_centroids):
        rand_num = random.choice([p for p in range(num_points) if p not in excluding])
        point_list[rand_num].is_centroid = True
        centroid_list.append(point_list[rand_num])
        excluding.append(rand_num)
    return centroid_list


def create_clusters(num_centroids: int):
    cluster_dict = {}
    for i in range(num_centroids):
        cluster_dict.update({i:[]})
    return cluster_dict


def populate_clusters(point_list: list, centroid_list: list, cluster_dict: dict):
    for point in point_list:
        if not point.is_centroid:
            point.cluster_id = point.index_closest_centroid(centroid_list)
            cluster_dict[point.cluster_id].append(point)


def center_of_mass(cluster_dict: dict):
    cm_list = []
    for i in cluster_dict:
        cluster_sum_x = 0
        cluster_sum_y = 0
        for point in cluster_dict[i]:
            cluster_sum_x += point.x
            cluster_sum_y += point.y
        cm_list.append((cluster_sum_x / len(cluster_dict[i]), cluster_sum_y / len(cluster_dict[i])))
    return cm_list


def relocate_centroids(centroid_list: list, cm_list: list):
    for i, centroid in enumerate(centroid_list):
        centroid.move_self((cm_list[i][0], cm_list[i][1]))


