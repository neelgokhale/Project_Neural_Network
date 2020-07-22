"""
Created by Neel Gokhale at 2020-07-21
File main.py from project Project_NN_From_Scratch
Built using PyCharm

"""

# Main program to test / view outputs from the neural network

from neuralnetwork import NeuralNetwork


if __name__ == '__main__':

    INPUT_DATA = ([2, 9], [1, 5], [3, 6], [5, 10])
    OUTPUT_DATA = ([92], [86], [89])

    net = NeuralNetwork(input_data=INPUT_DATA, output_data=OUTPUT_DATA, scale_data=True)
    net.train(train_epochs=150000, document=False)
    net.save_weights()
    net.predict(pred_val=None)