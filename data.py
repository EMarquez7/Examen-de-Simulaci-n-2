"""
# -- ------------------------------------------------------------------------------------------------------------------------ -- #   
# -- project: S & P 500 Risk Optimized Portfolios from 14 Apr (2020-2023)                                                     -- #           
# -- script: data.py : Python script with data functionality for the project                                                  -- #                 
# -- author: EstebanMqz                                                                                                       -- #  
# -- license: CC BY 3.0                                                                                                       -- #
# -- repository: https://github.com/EstebanMqz/SP500-Risk-Optimized-Portfolios/blob/main/data.py                              -- #           
# -- ------------------------------------------------------------------------------------------------------------------------ -- #  
"""

from os import path
#Dependencies
import functions as fn
import visualizations as vs

#Libraries in data.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import scipy.stats as st
from scipy import optimize

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

from yahoofinancials import YahooFinancials 
import datetime 
import time

from tabulate import tabulate
import IPython.display as d

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


def library_install(requirements_txt):
    """Install requirements.txt file."""
    import os
    import warnings
    warnings.filterwarnings("ignore")
    os.system(f"pip install -r {requirements_txt}")
    print("Requirements installed.")
    with open("requirements.txt", "r") as f:
        print(f.read())





