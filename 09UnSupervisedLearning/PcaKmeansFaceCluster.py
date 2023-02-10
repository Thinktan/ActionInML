
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

# show dataset
fig, axes = plt.subplots(3, 4, figsize=(9, 4), subplot_kw={'xticks':[], 'yticks': []},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
print(axes)