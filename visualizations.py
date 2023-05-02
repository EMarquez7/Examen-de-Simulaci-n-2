"""
# -- ------------------------------------------------------------------------------------------------------------------------ -- #   
# -- project: S&P500-Risk-Optimized-Portfolios-PostCovid-ML.                                                                  -- #           
# -- script: data.py : Python script with visualizations functionalities for the proj.                                        -- #                 
# -- author: EstebanMqz                                                                                                       -- #  
# -- license: CC BY 3.0                                                                                                       -- #
# -- repository: https://github.com/EstebanMqz/SP500-Risk-Optimized-Portfolios-PostCovid-ML/blob/main/visualizations.py       -- #           
# -- ------------------------------------------------------------------------------------------------------------------------ -- #  
"""

from os import path
#Dependencies
import functions as fn
import data as dt

#Libraries in visualizations.py
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

# -- --------------------------------------------------------------------------------------- visualizations ------------------------------------------------------------------------------- -- #

def summary(adjcloses,rf):
    """
    Function to get summary statistics of a dataframe of adjcloses, risk-free rates on a yearly basis are used from the end of the period.
    Parameters:
    ----------
    adjcloses : dataframe
        Dataframe of adjcloses.
    Returns:
    -------
    dataframe
        Dataframe with summary statistics with sharpe/sortino ratios of adjcloses.
    """
    returns = adjcloses.pct_change()
    mean_ret = returns.mean() * 252 
    sharpe = (mean_ret - rf) / ( returns.std() * np.sqrt(252) )
    sortino = (mean_ret - rf) / ( returns[returns < 0].std() * np.sqrt(252) )
    #Summary table
    summary = pd.DataFrame({"Annualized Return" : mean_ret, "Volatility" : returns.std()*np.sqrt(252), "Sharpe Ratio" : sharpe, "Sortino Ratio" : sortino})
    summary.index.name = "Financial Data"
    summary.head()

    return returns, summary
