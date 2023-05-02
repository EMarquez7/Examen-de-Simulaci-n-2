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

def selection_data(data, rf, title):
    """
    Function to get selection summary and data with given metrics.
    Parameters:
    ----------
    data : dataframe
        Prices to model.
    rf : float
        Daily Treasury Par Yield Curve Rates for sharpe/sortino selection from the end of the period on a yearly basis. 
    Returns:
    -------
    dataframe
        Summary statistics of data selection with sharpe/sortino ratios with given data.
    """
    returns = data.pct_change()
    mean_ret = returns.mean() * 252 
    sharpe = (mean_ret - rf) / ( returns.std() * np.sqrt(252) )
    sortino = (mean_ret - rf) / ( returns[returns < 0].std() * np.sqrt(252) )
    summary = pd.DataFrame({"Annualized Return" : mean_ret, "Volatility" : returns.std() * np.sqrt(252),
                            "Sharpe Ratio" : sharpe, "Sortino Ratio" : sortino})
    summary = summary.nlargest(30, "Sharpe Ratio").nlargest(30, "Sortino Ratio")
    bars = summary.plot.bar(figsize=(20, 10), rot=90, title=title, fontsize=15, grid=True, edgecolor="black", linewidth=1) 

    return summary, bars

