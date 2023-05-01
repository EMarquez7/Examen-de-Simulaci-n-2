"""
# -- ------------------------------------------------------------------------------------------------------------------------ -- #   
# -- project: S&P500-Risk-Optimized-Portfolios-PostCovid-ML.                                                                  -- #           
# -- script: data.py : Python script with general functions for the proj.                                                     -- #                 
# -- author: EstebanMqz                                                                                                       -- #  
# -- license: CC BY 3.0                                                                                                       -- #
# -- repository: https://github.com/EstebanMqz/SP500-Risk-Optimized-Portfolios-PostCovid-ML/blob/main/functions.py            -- #           
# -- ------------------------------------------------------------------------------------------------------------------------ -- #  
"""

from os import path
#Dependencies
import visualizations as vs
import data as dt

#Libraries in functions.py
import numpy as np
import pandas as pd
import matplotlib as plt

import scipy
import scipy.stats as st
from scipy import optimize

import sklearn
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

from yahoofinancials import YahooFinancials 
from tabulate import tabulate
import IPython.display as d

import datetime 
import time

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

# -- ----------------------------------------------------------------------------------------------- functions ------------------------------------------------------------------------------- -- #

#Define empty docstring
docstring = ""

def get_requirements(docstring):
    #MODIFY libraries:
    import numpy as np
    import pandas as pd
    import matplotlib as plt

    import scipy
    import scipy.stats as st
    from scipy import optimize

    import sklearn
    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV
    from sklearn import metrics

    from yahoofinancials import YahooFinancials
    from tabulate import tabulate

    import datetime
    import time

    """
    Get versions of imported libraries and create "requirements.txt" file for a project environment setup.
    Note: Libraries and requirements dictionary must be imported WITHIN the function and modified accordingly 
    where (MODIFY)  

    Parameters
    ----------
    docstring: str
        Docstring of requirements.txt script in a project.
    Returns
    -------
    requirements.txt
        File with libraries and their respective versions for a project environment setup.
    """

    #modif.: libs version used in proj. accordingly.
    requirements = {
        "numpy >=": np.__version__,
        "pandas >=": pd.__version__,
        "matplotlib >=": plt.__version__,
        "scipy >=": scipy.version.version,
        "sklearn >=": sklearn.__version__,

    }

    with open("requirements.txt", "w") as f:
        f.write(docstring)
        for key, value in requirements.items():
            f.write(f"{key} {value} \n")

        #modif.: libs without version attributes and jupyter if used in proj.
        f.write("jupyter >= 1.0.0 \n") 
        
        f.write("yahoofinanicals >= 1.14 \n")
        f.write("tabulate >= 0.8.9 \n")
        f.write("IPython >= 8.12.0 \n")

    print("requirements.txt file created in local path:", path.abspath("requirements.txt"))
    

