"""
Created by Neel Gokhale at 2020-07-21
File NeuralNetwork.py from project Project_NN_From_Scratch
Built using PyCharm

"""

import numpy as np
from objects.data import Data
import matplotlib.pyplot as plt
from utils import logging


class NeuralNetwork(object):

    def __init__(self, input_data: any, output_data: any, scale_data: bool = True):
        """
        Neural network class, containing all the functions used to model, train and validate based on a dataset

        :param input_data: list of input dataset
        :param output_data: list of output dataset
        :param scale_data: boolean value used to normalize assets. Leave at True for better performance
        """
        self.data = Data(input_data=input_data, output_data=output_data)
        self.scale_data = scale_data
        if self.scale_data:
            self.data.scale_data()
        self.data.split_data(where=-1)

        self.input_size = len(input_data[0])
        self.output_size = len(output_data[0])
        self.hidden_size = len(input_data[0]) + 1

        self.w_1 = np.random.randn(self.input_size, self.hidden_size)  # input -> weight layers
        self.w_2 = np.random.randn(self.hidden_size, self.output_size)  # weight layers -> output

    @staticmethod
    def sigmoid(x: np.ndarray):
        """
        Sigmoid activation function. Scales all inputs between 1 and 0.

        :param x: input value is a numpy array set from the Data class
        :return: input value scaled between 1 and 0
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def d_sigmoid(x: np.ndarray):
        """
        Derivative of sigmoid activation function. Used to calculate gradient descent

        :param x: input value is a numpy array set from back-propagation function
        :return: slopes of sigmoid function for array x
        """
        return x * (1 - x)

    # TODO: fully implement softmax function with variable output size
    @staticmethod
    def softmax(x: np.ndarray):
        return np.exp(x) / np.sum(np.exp(x))

    def forward(self, prediction_val: any = None):
        """
        Feed-forward function to predict the next output value based on a prediction value

        :param prediction_val: default param set to None
        :return: predicted output
        """
        if prediction_val is None:
            prediction_val = self.data.X
        self.dot1_input = np.dot(prediction_val, self.w_1)
        self.dot1_output = self.sigmoid(self.dot1_input)
        self.dot2_input = np.dot(self.dot1_output, self.w_2)
        self.output = self.sigmoid(self.dot2_input)

        return self.output

    def backward(self):
        """
        Main back-propagation function
        """
        self.output_error = self.data.Y - self.output  # error in input
        self.output_delta = self.output_error * self.d_sigmoid(
            self.output)  # scaling output error by sigmoid derivative

        self.w_2_error = self.output_delta.dot(self.w_2.T)  # w_2 error: error by hidden layer
        self.w_2_delta = self.w_2_error * self.d_sigmoid(self.dot1_output)  # scaling w_2 error by sigmoid derivative

        self.w_1 += self.data.X.T.dot(self.w_2_delta)  # adjusting w_1 (input -> hidden)
        self.w_2 += self.dot1_output.T.dot(self.output_delta)  # adjusting w_2 (hidden -> output)

    @logging.my_logger
    @logging.my_timer
    def train(self, train_epochs: int, document: bool = True, show_plt: bool = True):
        """
        Training function used to build the relationship matrix in the neural network object

        :param train_epochs: number of times the input dataset is trained
        :param document: controls the documentation of each step in the terminal. Set to True by default
        :param show_plt: draws plot of ongoing loss calculations by epoch count
        """
        for i in range(train_epochs):
            count = []
            loss_history = []
            loss = np.mean(np.square(self.data.Y - self.forward(self.data.X)))
            count.append(i)
            loss_history.append(np.round(loss, 6))

            print(f"count: {i}, loss: {loss}")

            if document:
                print("> epoch_" + str(i) + "\n")
                print("Input (scaled): \n" + str(self.data.X))
                print("Actual Output: \n" + str(self.data.Y))
                print("Predicted Output: \n" + str(self.forward()))
                print("Loss: \n" + str(loss))  # mean squared error
                print("\n")

            if show_plt:
                plt.title("Loss over Epochs")
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.plot(count, loss_history, 'r')
                plt.pause(0.001)

                self.output = self.forward()
                self.backward()

    def save_weights(self):
        """
        Used to save the current calculated weights
        """
        np.savetxt("/Users/Owner/PycharmProjects/Project_NN_From_Scratch/assets/train_snapshots/w1.txt", self.w_1, fmt='%s')
        np.savetxt("/Users/Owner/PycharmProjects/Project_NN_From_Scratch/assets/train_snapshots/w2.txt", self.w_2, fmt='%s')

    def predict(self, pred_val: any = None):
        """
        Prediction function used to predict a value retrieved from the input assets split (if provided by user)

        * If user defined input array is larger than output array, the default prediction value is the first separated input value from the split
        * If user wants to test a unique input value, set pred_val to that value

        :param pred_val: input prediction value. Set to desired value using the proper format, or leave to None if contained in input array
        """
        if pred_val is None and self.data.X_pred_val:
            print("> Predicting assets based on training... ")
            print("> Input (scaled): \n" + str(self.data.X_prediction))
            print("> Output: \n" + str(self.forward(prediction_val=self.data.X_prediction)))
        else:
            # FIXME: user inputted prediction value should be scaled by the right max values from dataset
            pred_val = pred_val / np.max(self.X_all, axis=0)
            print("> Predicting assets based on training... ")
            print("> Input (scaled): \n" + str(self.data.X_prediction))
            print("> Output: \n" + str(self.forward(prediction_val=pred_val)))
