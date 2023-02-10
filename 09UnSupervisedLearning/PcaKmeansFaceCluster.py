
# learn from
# https://www.kaggle.com/code/annatu/pca-kmeans-face-cluster/data

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imshow
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# load dataset.
faces_image = np.load('./input/olivetti_faces.npy')

print(faces_image.shape)
