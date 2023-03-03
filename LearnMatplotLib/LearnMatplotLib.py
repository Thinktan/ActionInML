
# https://www.runoob.com/w3cnote/matplotlib-tutorial.html

from pylab import *
import numpy as np

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)

#print(X)

C, S = np.cos(X), np.cos(X)
