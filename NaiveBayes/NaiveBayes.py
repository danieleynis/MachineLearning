# Author: Daniel Eynis
# ML HW4: Naive Bayes

import pandas
import numpy as np
from sklearn.cross_validation import train_test_split
import math
from sklearn import metrics

# read in the data file containing the different features and targets
data = pandas.read_csv('spambase.data', header=None).as_matrix()

# split the data half and half into test and training
train_data, test_data = train_test_split(data, test_size=0.5, train_size=0.5)

# get the train and test target data
train_target_data = train_data[:, -1]
test_target_data = test_data[:, -1]

# get the actual feature data for training and test
train_data = train_data[:, :-1]
test_data = test_data[:, :-1]

# split in pos and neg matrices
pos_train_data = train_data[[x for x, m in enumerate(train_target_data) if m == 1]]
neg_train_data = train_data[[x for x, m in enumerate(train_target_data) if m == 0]]

# get each class probability
pos_prob = len(pos_train_data)/len(train_data)
neg_prob = len(neg_train_data)/len(train_data)

# get the mean and standard deviation for each feature for neg and pos class
pos_mean = np.mean(pos_train_data, axis=0)
pos_std = np.std(pos_train_data, axis=0)
neg_mean = np.mean(neg_train_data, axis=0)
neg_std = np.std(neg_train_data, axis=0)

predicted = []  # store the predicted values for each test example
for i in range(len(test_data)):  # loop through test data
    # calculate the probability density function and use the result to do Gaussian Naive Bayes and class classification
    neg_pdf = np.nan_to_num(np.log((1/(math.sqrt(2*math.pi)*neg_std))*np.exp(-(test_data[i]-neg_mean)**2/(2*neg_std**2))))
    neg = np.log(neg_prob) + np.sum(neg_pdf)  # get the result for neg class (0)
    pos_pdf = np.nan_to_num(np.log((1/(math.sqrt(2*math.pi)*pos_std))*np.exp(-(test_data[i]-pos_mean)**2/(2*pos_std**2))))
    pos = np.log(pos_prob) + np.sum(pos_pdf)  # get the result for pos class (1)
    predicted.append(np.argmax([neg, pos]))  # take the argmax of both calculated values to determine class

# print accuracy, precision, recall, and confusion matrix
print('accuracy:\t' + str(metrics.accuracy_score(test_target_data, predicted)))
print('precision:\t' + str(metrics.precision_score(test_target_data, predicted)))
print('recall:\t\t' + str(metrics.recall_score(test_target_data, predicted)))
print('confusion:\n' + str(metrics.confusion_matrix(test_target_data, predicted)))



