"""Plots data corresponding to Figure 2 in

Playing Atari with Deep Reinforcement Learning
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis
Antonoglou, Daan Wierstra, Martin Riedmiller

Usage:

plot_results.py RESULTS_CSV_FILE
"""

import numpy as np
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
import sys

# Modify this to do some smoothing...
kernel = np.array([1.] * 1)
kernel = kernel / np.sum(kernel)

results = np.loadtxt(open(sys.argv[1], "rb"), delimiter=",", skiprows=1)
plt.subplot(1, 2, 1)
plt.plot(results[:, 0], np.convolve(results[:, 3], kernel, mode='same'), '-*')
#plt.ylim([0, 250])
plt.subplot(1, 2, 2)
plt.plot(results[:, 0], results[:, 4], '--')
#plt.ylim([0, 4])
plt.show()
