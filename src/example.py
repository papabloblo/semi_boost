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

utils.plot_classification(data_simulation = 'make_circles',ratio_unsampled = 0.9)

utils.plot_classification(data_simulation = 'make_moons',ratio_unsampled = 0.9)

utils.plot_classification(data_simulation = 'make_blobs',ratio_unsampled = 0.9)

utils.plot_classification(data_simulation = 'make_classification',ratio_unsampled = 0.9)

utils.plot_classification(data_simulation = 'make_gaussian_quantiles',ratio_unsampled = 0.9)

utils.mejora_semiboost(n_features = 10, ratio_unsampled = 0.95, n_samples = 1000,
                 data_simulation = 'make_moons', )
