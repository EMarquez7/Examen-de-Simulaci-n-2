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
    
    #For Indivual Assets:
    # 1. Add rf daily data on a yearly basis.
    # 2. Obtain beta (lambda data, rf: np.cov(data, rf)[0][1] / np.var(rf)) with np.dot & np.divide 
    # 3. Add Traynor (np.divide((mean_ret - rf), beta(data, rf))) & Jensen (mean_ret - np.dot((rf + beta(data, rf)), (mean_ret - rf))) Ratios.
    # 4. Modify Sharpe and Sortino to do calculations with rf daily data on a yearly basis.
    # 5. Display summary with all ratios in markdown df.

    summary = summary.nlargest(30, "Sharpe Ratio").nlargest(30, "Sortino Ratio")
    
    bars = summary.plot.bar(figsize=(20, 10), rot=90, title=title, fontsize=15, grid=True, edgecolor="black", linewidth=1)
    plt.grid(color='gray', linestyle='--')

    return summary, bars

def Optimizer(SP, rf):
    returns = (SP.pct_change()).iloc[1:, :].dropna(axis = 1)
    mean_ret = returns.mean() * 252
    cov = returns.cov() * 252
    N = len(mean_ret)
    bnds = ((0, None), ) * N
    cons = {"type" : "eq", "fun" : lambda weights : weights.sum() - 1}


    def Min_Var(weights, cov):
        return np.dot(weights.T, np.dot(cov, weights))

    def Max_Sharpe(w, er, rf, cov):
        erp = np.dot(w.T, er)
        sp = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        RS = (erp - rf) / sp
        return -RS if sp > 0 else -np.inf
    
    Wopt_MinVar = optimize.minimize(Min_Var, np.ones(N) / N, (cov,), 'SLSQP', bounds = bnds,
                    constraints = cons, options={"tol": 1e-10})
    Ropt_MinVar = np.dot(Wopt_MinVar.x.T, mean_ret)
    Vopt_MinVar = np.sqrt(np.dot(Wopt_MinVar.x.T, np.dot(cov, Wopt_MinVar.x)))
    Popt_MinVar = pd.DataFrame({"$\mu$" : Ropt_MinVar, "$\sigma$" : Vopt_MinVar, "$Sharpe-R_{max}$" :
                                                (Ropt_MinVar - rf) / Vopt_MinVar}, index = ["$Min_{Var{Arg_{max}}}$"])
                                                    
    Wopt_EMV = optimize.minimize(Max_Sharpe, np.ones(N) / N, (mean_ret, rf, cov), 'SLSQP', bounds = bnds,
                                 constraints = cons, options={"tol": 1e-10})   
    Ropt_EMV = np.dot(Wopt_EMV.x.T, mean_ret)
    Vopt_EMV = np.sqrt(np.dot(Wopt_EMV.x.T, np.dot(cov, Wopt_EMV.x)))
    Popt_EMV = pd.DataFrame({"$\mu$" : Ropt_EMV, "$\sigma$" : Vopt_EMV, "$Sharpe-R_{max}$" : 
                                                 (Ropt_EMV - rf) / Vopt_EMV}, index = ["$Sharpe_{Arg_{max}}$"])
    
    #For Optimized Portfolios:
    # 1. Add rf daily data on a yearly basis.
    # 2. Obtain beta (lambda data, rf: np.cov(data, rf)[0][1] / np.var(rf)) with np.dot & np.divide 
    # 3. Add Traynor (np.divide((mean_ret - rf), beta(data, rf))) & Jensen (mean_ret - np.dot((rf + beta(data, rf)), (mean_ret - rf))) Ratios.
    # 4. Modify Sharpe and Sortino to do calculations with rf daily data on a yearly basis.
    # 5. Display summary with all ratios in markdown df.
    # 6. Plot with legends with style.
    
    Argmax = d.Markdown(tabulate(pd.concat([Popt_EMV, Popt_MinVar], axis = 0), headers = "keys", tablefmt = "pipe"))
    
    return Argmax