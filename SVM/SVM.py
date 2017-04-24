# Author: Daniel Eynis
# ML HW3: SVMs

import pandas
import numpy as np
from numpy.random import choice
from sklearn import svm, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt


class SVM:
    def __init__(self):
        # read in the data file containing the different features and targets
        self.data = pandas.read_csv('spambase.data', header=None).as_matrix()
        # split the data half and half into test and training
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.5, train_size=0.5)
        # get the train and test target data
        self.train_target_data = self.train_data[:, -1]
        self.test_target_data = self.test_data[:, -1]
        # get the actual feature data for training and test
        self.train_data = self.train_data[:, :-1]
        self.test_data = self.test_data[:, :-1]
        # scale the data using the standard scalar (with mean and std dev)
        self.scalar = preprocessing.StandardScaler()
        self.train_data = self.scalar.fit_transform(self.train_data)
        self.test_data = self.scalar.transform(self.test_data)
        # fit the linear SVM to the training data
        self.linear_svm = svm.LinearSVC()
        self.linear_svm.fit(self.train_data, self.train_target_data)

    def experiment1(self):
        # classify the test examples
        predicted = self.linear_svm.predict(self.test_data)
        # get the confidence for the set of features
        confidence = self.linear_svm.decision_function(self.test_data)
        # calculate the accuracy, recall, and precision of SVM classification
        accuracy = self.linear_svm.score(self.test_data, self.test_target_data)
        recall = metrics.recall_score(self.test_target_data, predicted)
        precision = metrics.precision_score(self.test_target_data, predicted)
        # calculate the roc curve by getting false positive and true positive rates for different thresholds
        fp_rate, tp_rate, thresholds = metrics.roc_curve(self.test_target_data, confidence)
        # save collected data to a file
        f = open('data.txt', 'w')
        f.write('acc: ' + str(accuracy) + '\n')
        f.write('rec: ' + str(recall) + '\n')
        f.write('prec: ' + str(precision) + '\n')
        f.write('thresh: ' + str(len(thresholds)) + '\n')
        f.close()

        # create an ROC curve using the data collected
        plt.clf()
        plt.figure()
        plt.plot(fp_rate, tp_rate, color='red', lw=2)
        plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.savefig('exp1')  # save the plot to a file to use for report

    def experiment2(self):
        # indirectly sort the array of weights in descending order
        sorted_features = np.argsort(self.linear_svm.coef_[0])[::-1]
        # resort the train and test matrices columns to be in the new order (high weight feat. -> low weight)
        sorted_train = self.train_data[:, sorted_features]
        sorted_test = self.test_data[:, sorted_features]
        lin_svm_feat = svm.LinearSVC()
        feat_acc = []

        # loop adding a new feature to the set each time until including all possible features, for m = 2 ... 57 feats.
        for i in range(1, 57):
            # fit the SVM to the new data and then calculate accuracy and append to history array
            lin_svm_feat.fit(sorted_train[:, :i], self.train_target_data)
            feat_acc.append(lin_svm_feat.score(sorted_test[:, :i], self.test_target_data))

        # plot the accuracy vs features selected
        plt.clf()
        plt.figure()
        plt.plot([x for x in range(2, 58)], feat_acc, color='red', lw=2)
        plt.xlim([0, 57])
        plt.ylim([min(feat_acc) - 0.01, max(feat_acc) + 0.01])
        plt.xlabel('Number of features (m)')
        plt.ylabel('Accuracy rate on test data')
        plt.title('Accuracy of SVM on test data in relation to the number of features selected')
        plt.savefig('exp2')

    def experiment3(self):
        lin_svm = svm.LinearSVC()
        feat_acc = []
        # loop for m = 2 ... 57 feats in each random set generated
        for i in range(2, 58):
            # choose columns representing features from the matrix randomly
            columns = choice(57, size=i, replace=False)
            # fit the SVM and then calculate accuracy
            lin_svm.fit(self.train_data[:, columns], self.train_target_data)
            feat_acc.append(lin_svm.score(self.test_data[:, columns], self.test_target_data))

        # plot the data collected for accuracy vs random features selected
        plt.clf()
        plt.figure()
        plt.plot([x for x in range(2, 58)], feat_acc, color='red', lw=2)
        plt.xlim([0, 58])
        plt.ylim([min(feat_acc) - 0.05, max(feat_acc) + 0.05])
        plt.xlabel('Number of features randomly selected (m)')
        plt.ylabel('Accuracy rate on test data')
        plt.title('Accuracy rate of SVM on test data in relation to the number of random features selected')
        plt.savefig('exp3')
