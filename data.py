"""
Created by Neel Gokhale at 2020-07-21
File data.py from project Project_NN_From_Scratch
Built using PyCharm

"""

import numpy as np


class Data:

    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.X_all = np.array(self.input_data, dtype=float)
        self.X = None
        self.Y = np.array(self.output_data, dtype=float)
        self.X_prediction = None
        self.X_pred_val = False

    def scale_data(self):
        self.X_all = self.X_all / np.max(self.X_all, axis=0)
        self.Y = self.Y / 100

    def split_data(self, where: int = -1):
        if where == -1:
            if len(self.input_data) == 1 + len(self.output_data):
                self.X = np.split(self.X_all, [len(self.Y)])[0]
                self.X_prediction = np.split(self.X_all, [len(self.Y)])[1]
        else:
            self.X = np.split(self.X_all, [where])[0]
            self.X_prediction = np.split(self.X_all, [where])[1][0]

        self.X_pred_val = True
