# Author: Daniel Eynis
# ML HW2: Multilayer Neural Network

import pandas
import numpy as np


class MultilayerNetwork:

    def __init__(self, hidden_nodes, learn_rate, momentum):
        # initialize number of hidden nodes, learn rate, and momentum as well as create weight matrices for
        # input to hidden and hidden to output. Weights were randomly initialized to [-0.05, 0.05] with uniform distribution
        self.hidden_nodes = hidden_nodes
        self.hidden_output_w = np.random.uniform(-0.05, 0.05, size=(self.hidden_nodes + 1, 10))
        self.input_hidden_w = np.random.uniform(-0.05, 0.05, size=(785, self.hidden_nodes))
        self.learn_rate = learn_rate
        self.momentum = momentum

    def load_data(self, train_data_path, test_data_path):
        # read in csv and convert to numpy matrix
        self.test_data = pandas.read_csv(test_data_path, header=None).as_matrix()
        self.train_data = pandas.read_csv(train_data_path, header=None).as_matrix() # [np.random.choice(60000, 30000, replace=False)]
                                                                                    # use the above to filter a certain amount of examples

        # extract target data for test and train telling which number it is
        self.test_target = self.test_data[:, 0]
        self.test_target = self.test_target.reshape(1, len(self.test_target))
        self.train_target = self.train_data[:, 0]
        self.train_target = self.train_target.reshape(1, len(self.train_target))

        # extract pixel data excluding classification data which is first column
        self.test_data = self.test_data[:, 1:]
        self.train_data = self.train_data[:, 1:]

        # scale each input value to be [0, 1]
        self.test_data = self.test_data / 255
        self.train_data = self.train_data / 255

        # insert the bias column of 1s to beginning of input array
        self.test_data = np.insert(self.test_data, 0, 1, axis=1)
        self.train_data = np.insert(self.train_data, 0, 1, axis=1)

    # this is the learn function which will do forward and backwards propogation to learn
    def learn(self):
        test_acc_history = []  # stores history of test accuracies
        train_acc_history = []  # stores history of train accuracies
        prev_delta_ho_w = np.zeros((10, self.hidden_nodes + 1))  # matrix to store previous delta weights for hidden to output and input to hidden
        prev_delta_ih_w = np.zeros((self.hidden_nodes + 1, 785))
        for k in range(0, 50):  # this for loop sets the number of epochs to run for (current is 50)
            for m in range(0, self.train_target.shape[1]):  # for loop will cycle through training examples
                data = self.train_data[m]  # get the data at position m
                data = data.reshape(1, len(data))
                target = self.train_target[0][m]  # get the target value
                # start forward propogation
                hidden_activation = 1.0 / (1.0 + np.exp(-data.dot(self.input_hidden_w)))  # calculate hidden activation
                hidden_activation_wbias = np.insert(hidden_activation, 0, 1, axis=1)  # hidden activations with bias
                output_activation = 1.0 / (1.0 + np.exp(-hidden_activation_wbias.dot(self.hidden_output_w)))  # calculate output activations
                # start backward propogation
                target_array = np.full((1, 10), 0.1)  # create target array and set target index to 0.9, 0.1 for the rest
                target_array[0][target] = 0.9
                # calculate the error for the output nodes
                error_output = np.multiply((np.multiply(output_activation, (1-output_activation))), (target_array-output_activation))
                # calculate the error for the hidden layer nodes
                error_hidden = np.multiply((np.multiply(hidden_activation, (1-hidden_activation))), (error_output.dot(self.hidden_output_w.T[:, 1:])))
                # loop through each error and using it to calculate the delta weight for hidden to output weights
                for i in range(0, error_output.shape[1]):
                    delta_ho_w = ((self.learn_rate*error_output[0][i])*hidden_activation_wbias[0]) + (self.momentum*prev_delta_ho_w[i])
                    prev_delta_ho_w[i] = delta_ho_w  # store the calculated delta weight for the next iteration
                    self.hidden_output_w.T[i] += delta_ho_w  # adjust the weights
                # loop through each error and using it to calculate the delta weight for input to hidden weights
                for j in range(0, error_hidden.shape[1]):
                    delta_ih_w = ((self.learn_rate*error_hidden[0][j])*data[0]) + (self.momentum*prev_delta_ih_w[j])
                    prev_delta_ih_w[j] = delta_ih_w # store the calculated delta weight for the next iteration
                    self.input_hidden_w.T[j] += delta_ih_w  # adjust the weights
            ta, tra = self.test_accuracy()  # calculate the accuracies on the test and training data
            test_acc_history.append(ta)  # append the accuracies to the list of accuracy history
            train_acc_history.append(tra)
            print('Epoch ', k+1, ': ', ta, tra)
            # return the test and training accuracy history as lists and a confusion matrix as numpy array
        return test_acc_history, train_acc_history, self.confusion()

    # this function computes the confusion matrix by using the final weights to calculate predicted vs actual classification
    def confusion(self):
        confusion_matrix = np.zeros((10, 10), dtype=np.int64)  # create a 10x10 array of zeros
        for i in range(0, self.test_target.shape[1]):  # go through all test data
            data = self.test_data[i]
            data = data.reshape(1, len(data))
            hidden_activation = 1.0 / (1.0 + np.exp(-data.dot(self.input_hidden_w)))
            hidden_activation_wbias = np.insert(hidden_activation, 0, 1, axis=1)
            output_activation = 1.0 / (1.0 + np.exp(-hidden_activation_wbias.dot(self.hidden_output_w)))
            # add a 1 to the position in the matrix to predicted vs actual class
            confusion_matrix[self.test_target[0][i]][np.argmax(output_activation)] += 1
        return confusion_matrix

    # this function calculates the accuracy of the network on test and training data
    def test_accuracy(self):
        num_test_correct = 0  # number of correct classifications
        for i in range(0, self.test_target.shape[1]):  # loop through test data
            data = self.test_data[i]
            data = data.reshape(1, len(data))
            # forward propagate input
            hidden_activation = 1.0 / (1.0 + np.exp(-data.dot(self.input_hidden_w)))
            hidden_activation_wbias = np.insert(hidden_activation, 0, 1, axis=1)
            output_activation = 1.0 / (1.0 + np.exp(-hidden_activation_wbias.dot(self.hidden_output_w)))
            if np.argmax(output_activation) == self.test_target[0][i]:  # add 1 if correct classification
                num_test_correct += 1

        # do the same as above for training data
        num_train_correct = 0
        for i in range(0, self.train_target.shape[1]):
            data = self.train_data[i]
            data = data.reshape(1, len(data))
            hidden_activation = 1.0 / (1.0 + np.exp(-data.dot(self.input_hidden_w)))
            hidden_activation_wbias = np.insert(hidden_activation, 0, 1, axis=1)
            output_activation = 1.0 / (1.0 + np.exp(-hidden_activation_wbias.dot(self.hidden_output_w)))
            if np.argmax(output_activation) == self.train_target[0][i]:
                num_train_correct += 1

        # return the accuracies of for the training and test data
        return num_test_correct/self.test_target.shape[1], num_train_correct/self.train_target.shape[1]
