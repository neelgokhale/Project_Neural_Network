"""
Created by Neel Gokhale at 2020-08-28
File clustering_model.py from project Project_NN_From_Scratch
Built using PyCharm

"""

import matplotlib.pyplot as plt
import numpy as np
import random
import csv
from data_class.point import Point
from tqdm import tqdm


def graph_data(point_list: list, x_lbl: str="x", y_lbl: str="y", plt_title: str=None):
    """
    Function to graph list of `Point` objects

    :param point_list: list of Point objects
    :param x_lbl: x-axis label
    :param y_lbl: y-axis label
    :param plt_title: plot title
    """
    color_list = ['green', 'orange', 'purple', 'pink', 'yellow', 'lightblue', 'brown', 'cyan', 'lightgreen']
    for point in point_list:
        if point.is_centroid:
            color = 'r'
            marker = 'd'
        elif point.cluster_id is not None:
            color = color_list[point.cluster_id]
            marker = '.'
        elif point.test_point:
            color = 'black'
            marker = '.'
        plt.scatter(point.x, point.y, c=color, marker=marker)
        if point.is_centroid:
            plt.annotate(str(point.cluster_id), (point.x, point.y))
        if point.test_point:
            plt.annotate("test_point", (point.x, point.y))
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    if plt_title is None:
        plt.title(x_lbl + "_vs_" + y_lbl)
    else:
        plt.title(plt_title)
    plt.show()


def generate_rand_data(num_points: int, range_x: float=1, range_y: float=1):
    """
    Generate random `Point` object data with x and y range

    :param num_points: number of points in dataset
    :param range_x: range of x values
    :param range_y: range of y values
    :return: list of randomly generate Point objects
    """
    point_list = []
    for i in range(num_points):
        point_list.append(Point(random.random() * range_x,
                                random.random() * range_y))
    return point_list


def generate_data(file_path: str, headers: bool=True):
    """
    Generate `Point` object data imported from values from 2D csv file

    :param file_path: path of csv file
    :param headers: set to True if csv file has headers
    :return: list of generated Point objects
    """
    point_list = []
    with open(file_path, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        if headers:
            next(data, None)
        for row in data:
            point_list.append(Point(float(row[0]), float(row[1])))
    return point_list


def assign_centroids(num_centroids: int, point_list: list):
    """
    Randomly assign centroid designation to a given number of points within dataset

    :param num_centroids: number of centroids required
    :param point_list: list of Point objects
    :return: list of randomly assigned centroids
    """
    centroid_list = []
    excluding = []
    for i in range(num_centroids):
        rand_num = random.choice([p for p in range(len(point_list)) if p not in excluding])
        point_list[rand_num].is_centroid = True
        centroid_list.append(point_list[rand_num])
        excluding.append(rand_num)
    return centroid_list


def generate_centroids(num_centroids: int, point_list: list, precision: int=3):
    """
    Randomly generate centroids within the bounds of dataset

    :param num_centroids: number of centroids required
    :param point_list: list of Point objects
    :param precision: number of decimal places of precision on location of centroids
    :return: list of randomly generated centroids
    """
    precision = 10 ** precision
    centroid_list = []
    x_vals = [i.x for i in point_list]
    y_vals = [i.y for i in point_list]
    x_max, x_min = max(x_vals), min(x_vals)
    y_max, y_min = max(y_vals), min(y_vals)
    for i in range(num_centroids):
        centroid = Point(random.randrange(int(precision * x_min), int(precision * x_max)) / precision,
                         random.randrange(int(precision * y_min), int(precision * y_max)) / precision)
        centroid.is_centroid = True
        centroid.cluster_id = i
        centroid_list.append(centroid)
        point_list.append(centroid)
    new_centroid_list = check_centroids(centroid_list=centroid_list, point_list=point_list, precision=3)
    return new_centroid_list


def check_centroids(centroid_list: list, point_list: list, precision: int=3):
    """
    Internal function used to relocate proximal centroids

    :param centroid_list: list of centroids
    :param point_list: list of point objects
    :param precision: number of decimal places of precision on location of centroids
    :return: new centroid list with re-located centroids
    """
    if len(centroid_list) > 1:
        precision = 10 ** precision
        x_vals = [i.x for i in point_list]
        y_vals = [i.y for i in point_list]
        x_max, x_min = max(x_vals), min(x_vals)
        y_max, y_min = max(y_vals), min(y_vals)
        max_diag = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
        for check_c in centroid_list:
            check_list = centroid_list.copy()
            check_list.remove(check_c)
            for centroid in check_list:
                if check_c.distance_to(centroid) < 1/4 * max_diag:
                    check_c.move_self((random.randrange(int(precision * x_min), int(precision * x_max)) / precision,
                                       random.randrange(int(precision * y_min), int(precision * y_max)) / precision))
    return centroid_list


def create_clusters(num_centroids: int):
    """
    Generate cluster dictionary based on number of centroids

    :param num_centroids: number of centroids required
    :return: dictionary of clusters assigned by ordinal ID
    """
    cluster_dict = {}
    for i in range(num_centroids):
        cluster_dict.update({i:[]})
    return cluster_dict


def populate_clusters(point_list: list, centroid_list: list, cluster_dict: dict):
    """
    Populate cluster dictionary based on the locality of centroids

    :param point_list: list of Point objects
    :param centroid_list: list of assigned centroids
    :param cluster_dict: dictionary of clusters
    """
    for point in point_list:
        if not point.is_centroid:
            point.cluster_id = point.index_closest_centroid(centroid_list)
            cluster_dict[point.cluster_id].append(point)
    return cluster_dict


def center_of_mass(cluster_dict: dict):
    """
    Calculate center of mass of all points in a cluster

    :param cluster_dict: dictionary of clusters
    :return: list of tuples (x, y) of center-of-mass locations of all clusters
    """
    cm_list = []
    for i in cluster_dict:
        cluster_sum_x = 0
        cluster_sum_y = 0
        for point in cluster_dict[i]:
            cluster_sum_x += point.x
            cluster_sum_y += point.y
        if len(cluster_dict[i]) == 0:
            cm_list.append((None, None))
        else:
            cm_list.append((cluster_sum_x / len(cluster_dict[i]), cluster_sum_y / len(cluster_dict[i])))
    return cm_list


def relocate_centroids(centroid_list: list, cm_list: list):
    """
    Relocation of centroids to new cluster center-of-mass location

    :param centroid_list: list of centroids
    :param cm_list: list of tuples (x, y) of center-of-mass locations of all clusters
    """
    for i, centroid in enumerate(centroid_list):
        centroid.move_self((cm_list[i][0], cm_list[i][1]))


def regenerate(epochs: int, point_list: list, centroid_list: list, cluster_dict: dict, graph: bool=True, document: bool=False):
    """
    Regenerate clusters after centroid location refinement for given number of epochs

    :param epochs: number of times refinement is engaged
    :param point_list: list of Point objects
    :param centroid_list: list of assigned centroids
    :param cluster_dict: dictionary of clusters
    :param graph: controls if point_list should be graphed after final epoch
    :return: point_list with refined clusters and centroid values
    """
    for epoch in tqdm(range(epochs), ncols=100, desc="Progress"):
        if document:
            print(f"Epoch {epoch}")
        new_cluster_dict = populate_clusters(point_list, centroid_list, cluster_dict)
        cm_list = center_of_mass(new_cluster_dict)
        relocate_centroids(centroid_list, cm_list)
        cluster_dict = create_clusters(len(centroid_list))
    if graph:
        graph_data(point_list)
    return point_list


def predict_cluster(test_point, centroid_list: list, point_list: list, graph: bool=True):
    """
    Used to predict the local cluster of a new test data point

    :param test_point: Point object with test_point = True
    :param centroid_list: list of centroids
    :param point_list: list of point objects
    :param graph: controls if point_list with test point should be graphed
    :return: a list of probabilities corresponding to cluster IDs
    """
    dist_list = []
    prob_list = []
    for centroid in centroid_list:
        dist_list.append(1 / test_point.distance_to(centroid))
    total_inv_value = sum(dist_list)
    prob_list = [i / total_inv_value for i in dist_list]
    if graph:
        print(prob_list)
        point_list.append(test_point)
        graph_data(point_list)
    return prob_list
