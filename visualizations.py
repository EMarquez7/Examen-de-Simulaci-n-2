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



def Optimizer(SP, rf):
    returns18_20 = (SP.loc["2018-01-02":"2020-01-02"].pct_change()).iloc[1:, :].dropna(axis = 1)
    mean_ret = returns18_20.mean() * 252
    cov = returns18_20.cov() * 252
    N = len(mean_ret)
    w0 = np.ones(N) / N
    bnds = ((0, None), ) * N
    cons = {"type" : "eq", "fun" : lambda w : w.sum() - 1}


    def Var(w, cov):
        return np.dot(w.T, np.dot(cov, w))

    def Minus_RatioSharpe(w, er, rf, cov):
        erp = np.dot(w.T, er)
        sp = np.dot(w.T, np.dot(cov, w))**0.5
        RS = (erp - rf) / sp
        return -RS if sp > 0 else -np.inf

    pmv = minimize(fun = Var, x0 = w0, args = (cov,), bounds = bnds, constraints = cons, tol = 1e-10)
    pemv = minimize(fun = Minus_RatioSharpe, x0 = w0, args = (mean_ret, rf, cov), bounds = bnds, constraints = cons, tol = 1e-10)

    Er_pemv = np.dot(pemv.x.T, mean_ret)
    s_pemv = (np.dot(pemv.x.T, np.dot(cov, pemv.x)))**0.5
    summary = pd.DataFrame({"Returns" : Er_pemv, "Volatility" : s_pemv,
                                "Sharpe Ratio" : (Er_pemv - rf) / s_pemv}, index = ["EMV"])

    w_pemv = pd.DataFrame(np.round(pemv.x.reshape(1, N), 4), columns = returns18_20.columns, index = ["Weights"])
    w_pemv[w_pemv <= 0.0] = np.nan
    w_pemv.dropna(axis = 1, inplace = True)

    w = np.linspace(0, 1, 100)
    Er_pmv = np.dot(pmv.x, mean_ret)
    s_pmv = (np.dot(pmv.x.T, np.dot(cov, pmv.x)))**0.5
    cov_pmv_pemv = np.dot(pmv.x.T, np.dot(cov, pemv.x))

    minvar_frontier = pd.DataFrame({"Volatility" : ((w*s_pemv)**2 + 2*w*(1-w)*cov_pmv_pemv + ((1-w)*s_pmv)**2)**0.5, "Returns" : w*Er_pemv + (1 - w)*Er_pmv})

    minvar_frontier["Sharpe Ratio"] = (minvar_frontier["Returns"] - rf) / minvar_frontier["Volatility"]
    sp = np.linspace(0, summary["Volatility"].max())
    lac = pd.DataFrame({"Volatility" : sp, "Returns" : summary["Sharpe Ratio"].values[0]*sp + rf})

    plt.style.use('dark_background')
    plt.figure(figsize = (22, 12))
    plt.rc('grid', linestyle="--", color='gray')
    plt.rc('ytick', labelsize=13, color='lightgreen')
    plt.rc('xtick', labelsize=13, color='red')
    plt.plot(minvar_frontier["Volatility"], minvar_frontier["Returns"], color = "lightgreen", linewidth = 2.5)
    plt.plot(s_pemv, Er_pemv, "*r", ms=16, label = ("EMV:", 'E(r)=',Er_pemv.round(2),'σ=',s_pemv.round(2)))
    plt.plot(s_pmv, Er_pmv, "*b", ms=13, color = "dodgerblue",label = ("Port. min. var.", 'E(r)=',Er_pmv.round(2),'σ=',s_pmv.round(2)))
    plt.plot(lac["Volatility"], lac["Returns"], "--", color = "royalblue", label = "Capital Allocation Line")
    plt.scatter(minvar_frontier["Volatility"], minvar_frontier["Returns"],
                c = minvar_frontier["Sharpe Ratio"], cmap = "coolwarm")
    plt.colorbar(orientation = "horizontal", pad=0.13).set_label(label='Sharpe Ratio',size='14', weight='roman', family="Bell MT")

    for i in range(len(summary)):
        if summary.index[i] in w_pemv.columns:
            plt.plot(summary.iloc[i, 1], summary.iloc[i, 0], "*", ms=10, 
            label=('W=', round(w_pemv.loc["Weights", summary.index[i]],2), 'r=', round(summary.iloc[i, 0],2),'s=',round(summary.iloc[i, 1],2)))
            plt.text(summary.iloc[i, 1], summary.iloc[i, 0], summary.index[i])

    plt.title("Efficient Frontier 18-20",size='17', weight='bold', family="Constantia")
    plt.xlabel("$\sigma$",size='15', weight='roman', family="Georgia")
    plt.ylabel("$\mu$",size='15', weight='roman', family="Georgia")
    plt.grid(True)
    plt.legend(loc = "best")
    #Obtain legends in a dataframe 
    handles, labels = plt.gca().get_legend_handles_labels()
    df = pd.DataFrame(labels, columns = ["Assets"])

    return df, plt.show()


