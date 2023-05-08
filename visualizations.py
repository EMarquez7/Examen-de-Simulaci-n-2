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
    plt.grid(color='gray', linestyle='--')

    return summary, bars


def Optimizer(SP, rf, title):
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
    Popt_MinVar.index.name = title
                                                    
    Wopt_EMV = optimize.minimize(Max_Sharpe, np.ones(N) / N, (mean_ret, rf, cov), 'SLSQP', bounds = bnds,
                                 constraints = cons, options={"tol": 1e-10})   
    Ropt_EMV = np.dot(Wopt_EMV.x.T, mean_ret)
    Vopt_EMV = np.sqrt(np.dot(Wopt_EMV.x.T, np.dot(cov, Wopt_EMV.x)))
    Popt_EMV = pd.DataFrame({"$\mu$" : Ropt_EMV, "$\sigma$" : Vopt_EMV, "$Sharpe-R_{max}$" : 
                                                 (Ropt_EMV - rf) / Vopt_EMV}, index = ["$Sharpe_{Arg_{max}}$"])
    Popt_EMV.index.name = title
    
    Ratios = pd.concat([Popt_EMV, Popt_MinVar], axis = 0)    
    Argmax = d.Markdown(tabulate(Ratios, headers = "keys", tablefmt = "pipe"))
    
    return Argmax

def Accum_ts(accum):
    """
    Accum_ts is a function that plots time-series in a dataframe with 3 strategies as cols with matplot.
    Given dates in X-axis labels are formatted on a monthly / yearly basis for visualization purposes.
    Parameters:
    ----------
    accum : dataframe
        Dataframe with time-series to plot.
    Returns:
    -------
    Plot
        Plot of time-series in dataframe.
    """
    fig, ax = plt.subplots(figsize = (15, 7))
    ax.plot(accum.index, accum.iloc[:, 0], color = "red", label = accum.columns[0])
    ax.plot(accum.index, accum.iloc[:, 1], color = "green", label = accum.columns[1])
    ax.plot(accum.index, accum.iloc[:, 2], color = "blue", label = accum.columns[2])
    ax.set_title("Cumulative Returns", fontsize = 20)
    ax.set_xlabel("Date", fontsize = 15)
    ax.set_ylabel("Cumulative Returns", fontsize = 15)
    ax.legend(loc = "upper left", fontsize = 15)
    ax.grid(True)
    ax.grid(which='major', color='gray', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=90)
    plt.show()