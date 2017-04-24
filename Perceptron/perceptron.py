# Author: Daniel Eynis
# ML HW1: Perceptrons

import pandas
import numpy as np


class Perceptron:

    def __init__(self, learn_rate=0.01):
        # set the learn rate given through constructor
        self.learn_rate = learn_rate

    # this is the learn function which implements the perceptron learning algorithm
    def learn(self):
        # read in csv and convert to numpy matrix
        test_data = pandas.read_csv('mnist_test.csv', header=None).as_matrix()
        train_data = pandas.read_csv('mnist_train.csv', header=None).as_matrix()

        # extrct pixel data excluding classification data which is first column
        mnist_test_data = test_data[:, 1:]
        mnist_train_data = train_data[:, 1:]

        # extract target data for test and train telling which number it is
        test_target_data = test_data[:, 0]
        test_target_data = test_target_data.reshape(1, len(test_target_data))
        train_target_data = train_data[:, 0]
        train_target_data = train_target_data.reshape(1, len(train_target_data))

        # scale each input value to be [0, 1]
        mnist_test_data = mnist_test_data/255
        mnist_train_data = mnist_train_data/255

        # insert the bias column of 1s to beginning of input array
        mnist_test_data = np.insert(mnist_test_data, 0, 1, axis=1)
        mnist_train_data = np.insert(mnist_train_data, 0, 1, axis=1)

        # create weight matrix with initial random weights [-0.5, 0.5]
        weight_matrix = np.random.uniform(-0.5, 0.5, size=(785, 10))

        # take an epoch 0 accuracy test with both train and test data
        cur_accuracy = self.test_accuracy(mnist_test_data, weight_matrix, test_target_data)
        train_data_acc = self.test_accuracy(mnist_train_data, weight_matrix, train_target_data)
        prev_accuracy = 0  # store previous accuracy
        epoch = 0  # store current epoch counter
        test_acc_history = [cur_accuracy]  # list of accuracies correponding to epoch
        train_acc_history = [train_data_acc]
        print("learn rate: " + str(self.learn_rate))
        print("epoch " + str(epoch) + " accuracy: " + str(test_acc_history[epoch]*100) + "%")

        # while the difference in accuracy is about 0.01% or we are still not at epoch 70
        while epoch < 70 and abs(cur_accuracy-prev_accuracy)*100 > 0.01:
            prev_accuracy = cur_accuracy
            # loop through mnist train data and update weights accordigly
            for i in range(0, mnist_train_data.shape[0]):
                weight_result = mnist_train_data[i].dot(weight_matrix)  # take dot product with weight matrix
                num_choice = np.argmax(weight_result)  # take highest number in list (perceptron prediction)
                tk = yk = 0
                if train_target_data[0][i] == num_choice:  # if there is a correct prediction set tk to 1
                    tk = 1
                if weight_result[num_choice] > 0:  # if the weight for the prediction is bigger than 0 set yk to 1
                    yk = 1
                if tk != 1:  # if the prediction was wrong update the weights according to the perceptron learning algorithm
                    delta_weight = self.learn_rate * (tk - yk) * mnist_train_data[i]
                    weight_matrix.T[train_target_data[0][i]] -= delta_weight  # update the weights for incorrectly fired neuron and the one that was supposed to fire
                    weight_matrix.T[num_choice] += delta_weight
            epoch += 1
            # do an accuracy test with test and training data and record
            cur_accuracy = self.test_accuracy(mnist_test_data, weight_matrix, test_target_data)
            train_data_acc = self.test_accuracy(mnist_train_data, weight_matrix, train_target_data)
            test_acc_history.append(cur_accuracy)
            train_acc_history.append(train_data_acc)
            print("epoch " + str(epoch) + " accuracy: " + str(test_acc_history[epoch]*100) + "%, Accuracy diff: " + str(abs(cur_accuracy-prev_accuracy)*100))
        # this function returns the final weight matrix, test and train accuracy history and a confusion matrix
        return weight_matrix, test_acc_history, train_acc_history, self.confusion_matrix(mnist_test_data, weight_matrix, test_target_data)

    # this function takes a test matrix, weight matrix, and target results to check the accuracy of the weights for
    # the perceptron
    def test_accuracy(self, test_matrix, weight_matrix, target_data):
        num_correct = 0  # tracks number of correct predictions
        for i in range(0, test_matrix.shape[0]):  # go through all test examples in matrix
            weight_result = test_matrix[i].dot(weight_matrix)
            choice = np.argmax(weight_result)
            if target_data[0][i] == choice:  # if correct prediction increment
                num_correct += 1
        return num_correct/test_matrix.shape[0]  # return the accuracy by dividing number of correct by total examples

    # this function returns a confusion matrix given a test matrix, weight matrix, and target data to check against
    def confusion_matrix(self, test_matrix, weight_matrix, target_data):
        confusion_matrix = np.zeros((10, 10), dtype=np.int64)  # create a 10x10 array of zeros
        for i in range(0, test_matrix.shape[0]):  # go through all training examples
            weight_result = test_matrix[i].dot(weight_matrix)
            choice = np.argmax(weight_result)
            confusion_matrix[target_data[0][i]][choice] += 1  # insert result into confusion matrix [predicted][actual]
        return confusion_matrix  # return the 10x10 matrix
