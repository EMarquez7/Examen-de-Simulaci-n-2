"""
# -- ------------------------------------------------------------------------------------------------------------------------ -- #   
# -- project: S & P 500 Risk Optimized Portfolios from 14 Apr (2020-2023)                                                     -- #           
# -- script: data.py : Python script with general functions.                                                                  -- #                 
# -- author: EstebanMqz                                                                                                       -- #  
# -- license: CC BY 3.0                                                                                                       -- #
# -- repository: https://github.com/EstebanMqz/SP500-Risk-Optimized-Portfolios/blob/main/functions.py                         -- #           
# -- ------------------------------------------------------------------------------------------------------------------------ -- #  
"""

from os import path
#Dependencies
import visualizations as vs
import data as dt

#Libraries in functions.py
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

#Define empty docstring
docstring = ""

def get_requirements(docstring):
    #MODIFY libraries:
    import numpy as np
    import pandas as pd
    import matplotlib as plt
    #LaTeX
    from tabulate import tabulate
    from random import randrange

    """
    Get versions of imported libraries and create "requirements.txt" file for a project environment setup.
    Note: Libraries and requirements dictionary must be imported WITHIN the function and modified accordingly (MODIFY)  

    Parameters
    ----------
    docstring: str
        Docstring of requirements.txt script in a project.
    Returns
    -------
    requirements.txt
        File with libraries and their respective versions for a project environment setup.
    """

    #MODIFY: with libs. with .__version__ attribute (check e.g: np.__version__)
    requirements = {
        "numpy >=": np.__version__,
        "pandas >=": pd.__version__,
        "matplotlib >=": plt.__version__,
    }

    with open("requirements.txt", "w") as f:
        f.write(docstring)
        for key, value in requirements.items():
            f.write(f"{key} {value} \n")

        #MODIFY: with libs. without .__version__ attribute (Refs on: https://pypi.org)
        f.write("jupyter >= 1.0.0 \n") #nb lib not imported. 
        f.write("tabulate >= 0.8.9 \n")
        f.write("IPython >= 8.12.0 \n")

    print("requirements.txt file created in local path:", path.abspath("requirements.txt"))
    

