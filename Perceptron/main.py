# Author: Daniel Eynis
# ML HW1: Perceptrons

import perceptron
import numpy as np

p = perceptron.Perceptron(learn_rate=0.1)

w_m, t_a_h, tr_a_h, c_m = p.learn()

f = open("data.txt", "w")
f.write(np.array_str(c_m) + "\n")
f.write("Test acc: " + str(t_a_h) + "\n")
f.write("Train acc: " + str(tr_a_h) + "\n")
f.close()
