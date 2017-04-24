# Author: Daniel Eynis
# ML HW2: Multilayer Neural Network

import MultilayerNetwork
import numpy as np

mln = MultilayerNetwork.MultilayerNetwork(100, 0.1, 0.9)
mln.load_data('mnist_train.csv', 'mnist_test.csv')
ta, tra, cm = mln.learn()
f = open("data.txt", "w")
f.write(np.array_str(cm) + "\n")
f.write("Test acc: " + str(ta) + "\n")
f.write("Train acc: " + str(tra) + "\n")
f.close()
