"""
Author: Daniel Eynis
HW5: K-Means

Instructions to run:
You can just run this python file as is if you already have the train and test data sets in the same folder. Also
remember to have the necessary libraries loaded (see imports)
The only thing that should be adjusted is the k value below
"""

import numpy as np
import pandas as pd
from numpy import random
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# This is the euclidean distance function that can take two numpy arrays
# The a value selects across which axis to do the distance calculation
# the r value specifies if to reshape the resulting array (depends on dimensions)
def distance(x, y, a=1, r=True):
    d = np.sqrt(np.sum((np.subtract(x, y)**2), axis=a))
    if r is True:
        return np.reshape(d, (len(d), 1))
    return d

# This is the Mean Square Error function that will sum the distances and divide by the k value
def mse(x):
    return np.sum(x)/len(x)

# Calculate mean Square Separation between cluster centers
def mss(x):
    s = 0
    for y in range(len(x)):  # loop through same array creating pairs (y, w) where y != w
        for w in range(y+1, len(x)):
            s += distance(x[y], x[w], 0, False)  # get the distance and add it to running sum
    return s/(len(x)*(len(x)-1)/2)  # plug into MSS formula to get final result


# load in the train and test data sets
train_data = pd.read_csv('optdigits.train', header=None).as_matrix()
test_data = pd.read_csv('optdigits.test', header=None).as_matrix()

# get the train and test target data
train_target_data = train_data[:, -1]
test_target_data = test_data[:, -1]

# get the actual feature data for training and test
train_data = train_data[:, :-1]
test_data = test_data[:, :-1]

# training section
k = 10  # the k value for the number of cluster centers to generate
centroids_stats = []  # keeps a list of generated cluster centers and their corresponding info (mse, mss, etc...)

for _ in range(5):
    dist = None
    centroid_class = None
    centroid_buckets = None
    centroid_target_buckets = None
    centroids = random.randint(17, size=(k, 64)).astype(dtype=float)  # generate an array of random k centers
    prev_centroids = np.copy(centroids)

    while True:
        # get the distances from each training point to each centroid
        dist = distance(train_data, centroids[0])
        for j in range(1, k):
            dist = np.concatenate((dist, distance(train_data, centroids[j])), axis=1)

        centroid_class = np.argmin(dist, axis=1)  # assign each example to its closest centroid
        centroid_buckets = [[] for cb in range(k)]  # store examples that correspond to a particular centroid
        centroid_target_buckets = [[] for ctb in range(k)]  # store corresponding target values

        # for each example put it into its centroid bucket and target bucket creating a distribution
        for n in range(len(centroid_class)):
            centroid_buckets[centroid_class[n]].append(train_data[n])
            centroid_target_buckets[centroid_class[n]].append(train_target_data[n])

        # Go through each bucket and find the mean for the values in that bucket and then use that to update the
        # corresponding centroid that those examples belonged to
        for m in range(len(centroid_buckets)):
            if len(centroid_buckets[m]) != 0:
                new_center = np.mean(np.array(centroid_buckets[m]), axis=0)
                centroids[m] = new_center

        # if the centroid centers have stopped changing then break out of the training loop
        if np.array_equal(prev_centroids, centroids):
            break
        prev_centroids = np.copy(centroids)

    mse_clustering = []  # store mse data for each center
    sel_centroids = []  # store the non empty centers
    # get the mse for each cluster center and record the non empty centers
    for mc in range(len(centroid_buckets)):
        if len(centroid_buckets[mc]) != 0:
            c = mse(distance(np.array(centroid_buckets[mc]), centroids[mc]))
            mse_clustering.append(c)
            sel_centroids.append(mc)

    classes = []  # store the classification info for each center
    # Get the most prevalent target value and assign it to that center for future classification
    for buck in centroid_target_buckets:
        if len(buck) != 0:
            classes.append(np.bincount(buck).argmax())

    # record all info for this run for the centers
    centroids_stats.append((mse(mse_clustering), mss(centroids), centroids[sel_centroids], classes))

final_centroid = None
# find the run with the lowest MSE and select those centers for testing
for tup in centroids_stats:
    if final_centroid is None:
        final_centroid = tup
    elif tup[0] < final_centroid[0]:
        final_centroid = tup

# testing section
# get the distance for each test example to the centers
t_dist = distance(test_data, final_centroid[2][0])
for j in range(1, len(final_centroid[2])):
    t_dist = np.concatenate((t_dist, distance(test_data, final_centroid[2][j])), axis=1)

# Assign each example to its closest center
test_centroids = np.argmin(t_dist, axis=1)
prediction = []
# predict that examples class based on its associated center
for p in range(len(test_centroids)):
    prediction.append(final_centroid[3][test_centroids[p]])

# print the MSE, MSS, and Accuracy for the testing as well as confusion matrix
print("MSE: " + str(final_centroid[0]))
print("MSS: " + str(final_centroid[1]))
print("ACC: " + str(accuracy_score(test_target_data, prediction)))
print(confusion_matrix(test_target_data, prediction))

# Create the gray scale images using the cluster centers
for ctr in range(len(final_centroid[2])):
    plt.clf()
    im = np.reshape(final_centroid[2][ctr].astype(int), (8, 8))
    plt.imshow(im, cmap='gray')
    plt.savefig(str(ctr) + "_" + str(final_centroid[3][ctr]))
