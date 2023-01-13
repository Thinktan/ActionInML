# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
# default: macOS
PROJECT_ROOT_DIR = "/Users/thinktan/ActionInMlImage"

import platform
if platform.system().lower() == "linux":
    PROJECT_ROOT_DIR = "/home/thinktan/ActionInMlImage"
elif platform.system().lower() == "windows":
    PROJECT_ROOT_DIR = "D:/ActionInMlImage" # TODO

CHAPTER_ID = "ensembles"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR,  CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
print(IMAGES_PATH)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)