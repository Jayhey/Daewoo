import pandas as pd
import numpy as np
import sklearn.metrics as skm
import os
import re


from checkpoints_ensemble_module import ensemble_a_of_b

__author__ = "Gyubin Son"
__copyright__ = "Copyright 2018, Daewoo Shipbuilding & Marine Engineering Co., Ltd."
__credits__ = ["Minsik Park", "Jaeyun Jeong", "Heejeong Choi"]
__version__ = "1.0"
__maintainer__ = "Gyubin Son"
__email__ = "gyubin_son@korea.ac.kr"
__status__ = "Develop"


if __name__ == "__main__":
    print('ensemble 20 of 50')
    m2050 = ensemble_a_of_b(top_k=20, upper_bound_epoch=50)
    print('===========\nensemble 30 of 50')
    m3050 = ensemble_a_of_b(top_k=30, upper_bound_epoch=50)
    print('===========\nensemble 20 of 100')
    m20100 = ensemble_a_of_b(top_k=20, upper_bound_epoch=100)
    print('===========\nensemble 30 of 100')
    m30100 = ensemble_a_of_b(top_k=30, upper_bound_epoch=100)
    print('===========\nensemble 50 of 100')
    m50100 = ensemble_a_of_b(top_k=50, upper_bound_epoch=100)

