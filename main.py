"""
Created by Neel Gokhale at 2020-07-21
File main.py from project Project_NN_From_Scratch
Built using PyCharm

"""

# Main program to test / view outputs from the neural network

from neuralnetwork import NeuralNetwork
import clustering_model as cl
import point


if __name__ == '__main__':

    # Neural Network example

    INPUT_DATA = ([2, 9], [1, 5], [3, 6], [5, 10])
    OUTPUT_DATA = ([92], [86], [89])

    net = NeuralNetwork(input_data=INPUT_DATA, output_data=OUTPUT_DATA, scale_data=True)
    net.train(train_epochs=100, document=False, show_plt=True)
    net.save_weights()
    net.predict(pred_val=None)  # User input if needed for pred_val (should be in input arr format)

    # Clustering Model example

    file_path = "C:/Users/Owner/PycharmProjects/Project_NN_From_Scratch/test_data/test_data.csv"
    num_c = 5
    epochs = 1000
    test_point = point.Point(3.5, 1.5, test_point=True)

    point_list = cl.generate_data(file_path=file_path, headers=True)
    centroid_list = cl.generate_centroids(num_centroids=num_c, point_list=point_list)
    cluster_dict = cl.create_clusters(num_centroids=num_c)

    refined_point_list = cl.regenerate(epochs=epochs,
                                       point_list=point_list,
                                       centroid_list=centroid_list,
                                       cluster_dict=cluster_dict,
                                       graph=False)

    prob_list = cl.predict_cluster(test_point=test_point,
                                   centroid_list=centroid_list,
                                   point_list=point_list,
                                   graph=True)