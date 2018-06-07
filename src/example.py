''' Example of semi-boost implementation '''

import os
# Set working directory to file directory
os.chdir('/data/home/joliver/github/semi_boost/src')

import sys
import importlib
import SemiBoost
import utils
import importlib

importlib.reload(SemiBoost)
importlib.reload(utils)

# make_classification, make_gaussian_quantiles, make_blobs, make_moons, make_circles
utils.plot_classification(data_simulation = 'make_moons',ratio_unsampled = 0.7)

utils.plot_classification(data_simulation = 'make_moons',ratio_unsampled = 0.7)


utils.mejora_semiboost(n_features = 10, ratio_unsampled = 0.8, n_samples = 1000,
                 data_simulation = 'make_moons', )
