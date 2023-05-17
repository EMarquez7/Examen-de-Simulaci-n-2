"""
# -- -------------------------------------------------------------------------------------------------------------- -- #   
# -- project: S&P500-Risk-Optimized-Portfolios-PostCovid-ML.                                                        -- #           
# -- script: data.py : Python script with general functions for the proj.                                           -- #                 
# -- author: EstebanMqz                                                                                             -- #  
# -- license: CC BY 3.0                                                                                             -- #
# -- repository: https://github.com/EstebanMqz/SP500-Risk-Optimized-Portfolios-PostCovid-ML/blob/main/functions.py  -- #           
# -- ----------------------------------------------------------------------------------------------------------------- #  
"""

from os import path
#Dependencies
import visualizations as vs
import data as dt

# Libraries in functions.py
import numpy as np
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None
              ,"display.max_colwidth", None, "display.width", None)

import matplotlib
import matplotlib.pyplot as plt
plt.style.use("dark_background")

import scipy
import scipy.stats as st
from scipy import optimize
from scipy.optimize import minimize

import sklearn
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import ast

from yahoofinancials import YahooFinancials 
from tabulate import tabulate
import IPython.display as d

from io import StringIO
from fitter import Fitter, get_common_distributions, get_distributions 
import logging
logging.getLogger().setLevel(logging.ERROR)

import datetime 
import time
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

# -- ------------------------------------------------------------------------------------------------------------------------------ Functions ----------------------------------------------------------------------------------------------------------------------------- -- #

##############################################################################################################################################################################################################################################################################

docstring = ""

def get_requirements(docstring):
    #MODIFY libraries:
    import numpy as np
    import pandas as pd
    
    import matplotlib
    import matplotlib.pyplot as plt

    import scipy
    import scipy.stats as st
    from scipy import optimize

    import sklearn
    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV
    from sklearn import metrics
    import ast

    from yahoofinancials import YahooFinancials
    from tabulate import tabulate
    import IPython.display as d

    from io import StringIO
    from fitter import Fitter, get_common_distributions, get_distributions 
    import logging
    logging.getLogger().setLevel(logging.ERROR)

    import datetime
    import time
    import warnings
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=UserWarning)
    """
    Function to create requirements.txt file in order to setup environment. Libraries and requirements dictionary
    must be imported within the function: 1° section libraries with versions attributes, 2° section libraries without
    versions attribute. Note: Prebuilt libraries should not be included. Include `jupyter` if there are .ipynb's in proj.


    Parameters
    ----------
    docstring: str
        Docstring of requirements.txt script in a project.
    Returns
    -------
    requirements.txt
        File with libraries and their respective versions for a project environment setup.
    """

    #1° SECTION: Libs. with version attributes used in project.
    requirements = {
        "numpy >=": np.__version__  ,
        "pandas >=": pd.__version__ ,
        "matplotlib >=": matplotlib.__version__ , 
        "scipy >=": scipy.version.version   , 
        "sklearn >=": sklearn.__version__   ,
        "logging >=": logging.__version__   ,
    }

    with open("requirements.txt", "w") as f:
        f.write(docstring)
        for key, value in requirements.items():
            f.write(f"{key} {value} \n")

        #2° SECTION: Libs. without version attributes in project.
        f.write("jupyter >= 1.0.0 \n") 

        f.write("yahoofinanicals >= 1.14 \n")
        f.write("tabulate >= 0.8.9 \n")
        f.write("IPython >= 8.12.0 \n")
        f.write("fitter >= 1.5.2 \n")

    print("requirements.txt file created in local path:", path.abspath("requirements.txt"))
    return path.abspath("requirements.txt")

##############################################################################################################################################################################################################################################################################

def library_install(requirements_txt):
    """Install requirements.txt file in project created with fn.get_requirements. """
    import os
    import warnings
    warnings.filterwarnings("ignore")
    os.system(f"pip install -r {requirements_txt}")
    print("Requirements installed.")
    with open("requirements.txt", "r") as f:
        print(f.read())

##############################################################################################################################################################################################################################################################################

def SP500_tickers(batches):
    """
    Function to fetch tickers from S&P500 .html by batches or (N) lists of lists. Undivisible n batches sizes slice list[-1] 
    Parameters:
    ----------
    batches : int
        N tickers in lists of lists in function S&P 500 tickers.
    Returns:
    -------
    list
        Lists of lists for all quotes listed in the S&P 500 acc0rding to refs. 
    """

    list = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]["Symbol"].values.tolist()
    data = [[x.replace(".", "-") for x in list[x:x+batches]] for x in range(0, len(list), batches)]
    
    return data

##############################################################################################################################################################################################################################################################################

def format_table(describe):
    """
    format_table is a function to format the output of vs.Stats function in order to show dist_fit in vs.Stats[1] containing Xi params, AIC and BIC.
    Parameters:
    ----------
    describe : list
        List of lists with output of vs.Stats[1] in .ipynb.
    Returns:
    -------
    df : dataframe
        Formatted Dataframe with Xi dictionaries with rows & cols. [Params., AIC & BIC].
    """

    df = pd.DataFrame([ast.literal_eval(describe[i][-1][j]) for i in range(len(describe)) for j in range(len(describe[i][-1]))][0:])
    df = df.apply(lambda row: pd.Series(row).drop_duplicates(keep='first'),axis='columns')
    df.apply(lambda x : pd.Series(x[x.notnull()].values.tolist()+x[x.isnull()].values.tolist()),axis='columns')
    df.columns = ["params", "AIC", "BIC"]
    df.index = pd.DataFrame(np.array([describe[0][0].index.values + str(" Wk"), describe[1][0].index.values + str(" Mo"), describe[2][0].index.values + str(" Qt")]).reshape(len(describe[0][0]) * 3, 1))
    
    return df

##############################################################################################################################################################################################################################################################################