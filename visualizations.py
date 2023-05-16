"""
# -- ------------------------------------------------------------------------------------------------------------------- -- #   
# -- project: S&P500-Risk-Optimized-Portfolios-PostCovid-ML.                                                             -- #           
# -- script: data.py : Python script with visualizations functionalities for the proj.                                   -- #                 
# -- author: EstebanMqz                                                                                                  -- #  
# -- license: CC BY 3.0                                                                                                  -- #
# -- repository: https://github.com/EstebanMqz/SP500-Risk-Optimized-Portfolios-PostCovid-ML/blob/main/visualizations.py  -- #           
# -- ------------------------------------------------------------------------------------------------------------------- -- #  
"""

from os import path
#Dependencies
import functions as fn
import data as dt

#Libraries in visualizations.py
import numpy as np
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None
              ,"display.max_colwidth", None, "display.width", None)

from io import StringIO
import ast
from fitter import Fitter, get_common_distributions, get_distributions 
import logging

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

from io import StringIO
from fitter import Fitter, get_common_distributions, get_distributions 
import logging
logging.getLogger().setLevel(logging.ERROR)

import datetime 
import time
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

# -- ---------------------------------------------------------------------------------------------------------------------------------- Visualizations --------------------------------------------------------------------------------------------------------- -- #

def selection_data(dataframe, rf, title, best):
    """
    Function that calculates Annualized Returns and Std. Deviation for a given dataframe in order to obtain 
    n_best Sharpe & Sortino Ratios with a risk-free rate.
    Parameters:
    ----------
    dataframe : dataframe
        Dataframe for the model.
    rf : float
        Yield Curve Rates (see Refs.) from model's end date on a Yearly basis for n_best Sharpe & Sortino Ratios. 
    best: int
        No° of best Sharpe & Sortino Ratios to integrate in selection (if dataframe cols. <= best, then all cols. are selected). 
        
    Returns:
    -------
    dataframe
        Annualized Returns and Std. deviations of best Sharpe & Sortino displayed in a table and a plot bar for each column in dataframe.
            Note: Markdown's pandas dataframe located in pos. [2] of return.
    """
    
    returns = (dataframe.pct_change()).iloc[1:, :].dropna(axis = 1)
    mean_ret = returns.mean() * 252 
    sharpe = (mean_ret - rf) / ( returns.std() * np.sqrt(252) )
    sortino = (mean_ret - rf) / ( returns[returns < 0].std() * np.sqrt(252) )
      
    summary = pd.DataFrame({"$\mu_{i{yr}}$" : mean_ret, "$\sigma_{yr}$" : returns.std() * np.sqrt(252),
                            "$R_{Sharpe}$" : sharpe, "$R_{Sortino}$" : sortino})
    
    summary = summary.nlargest(best, "$R_{Sharpe}$").nlargest(best, "$R_{Sortino}$")
    bars = summary.plot.bar(figsize=(22, 10), rot=45, fontsize=15, grid=True, linewidth=1)
    
    plt.title(title, fontsize=20)
    plt.yticks(fontsize=10)
    plt.grid(color='gray', linestyle='--')
    for i, t in enumerate(bars.get_xticklabels()):
        if i % 2 == 0:
            t.set_color('lightgreen')
        else:
            t.set_color('white')
    markdown = d.Markdown(tabulate(summary, headers = "keys", tablefmt = "pipe"))

    return display(markdown), bars, summary


##############################################################################################################################################################################################################################################################################


def Stats(dataframe, Selection, P,  title, start, end, percentiles, dist, color):
    """
    Stats is a function that resamples data from a Selection performed over a dataframe.
    Parameters:
    ----------
    dataframe : dataframe
        Dataframe from which the Selection is made, in order to acess Selection's original data.
    Selection : list
        Selection to resample on a W, M, Q or Y basis whose period is longer than original data.
    P : str
        Period of resample (e.g. "W" for Weekly, "M" for monthly, "Q" for quarterly, "Y" for yearly).
    title : str
        Title of the boxplot
    start : str
        Start date for box plot title.
    end : str
        End date of the dataframe for the boxplot title.
    percentiles : list
        List of percentiles to return in statistics dataframe (e.g. [.05, .25, .5, .75, .95]).
    dist : list
        Continous Distributions to fit on datasets Xi
    color : str
        Color of the boxplot.
    Returns:
    -------
    describe : dataframe
        Stats returns summary statistics (mean, std, min, max, percentiles, skewness and kurtosis) in a 
        markdown object callable as a dataframe by assigning a variable to the function in pos. [2].  
    """
    
    Selection = (dataframe[Selection.index].pct_change()).iloc[1:, :].dropna(axis = 1)
    Selection.index = pd.to_datetime(Selection.index)
    
    Selection_Mo_r = Selection.resample(P).sum()


    Selection_Mo_r.plot(kind = "box", figsize = (22, 13),
                      title = title + str(start) + " to " + str(end), color = color, fontsize = 13)
    
    for i in range(0, len(Selection_Mo_r.columns)):
        plt.text(x = i + 0.96 , y = Selection_Mo_r.iloc[:, i].mean() + .0075, s = str("$\mu$ = +") + str(round(Selection_Mo_r.iloc[:, i].mean(), 4)), fontsize = 7, fontweight = "bold", color = "lightgreen")
        plt.text(x = i + 0.98 , y = Selection_Mo_r.iloc[:, i].max() + .010, s = str("+") + str(round(Selection_Mo_r.iloc[:, i].max(), 3)), fontsize = 9, color = "green")
        plt.text(x = i + 0.98 , y = Selection_Mo_r.iloc[:, i].min() - .015, s = str(round(Selection_Mo_r.iloc[:, i].min(), 3)), fontsize = 9, color = "red")

    describe = Selection_Mo_r.describe(percentiles).T
    describe["mode"] = Selection_Mo_r.mode().iloc[0, :]
    describe["skewness"] = st.skew(Selection_Mo_r)
    describe["kurtosis"] = st.kurtosis(Selection_Mo_r)

    logging.getLogger().setLevel(logging.ERROR)
    dist_fit = np.empty(len(Selection_Mo_r.columns), dtype=object)
    
    for i in range(0, len(Selection.columns)):
        f = Fitter(pd.DataFrame(Selection_Mo_r.iloc[:, i]), distributions = dist, timeout=5)
        f.fit()
        params, AIC, BIC = [StringIO() for i in range(3)]
        (print(f.get_best(), file=params)), (print(f.get_best(method="aic"), file=AIC)), (print(f.get_best(method="bic"), file=BIC))
        params, AIC, BIC = [i.getvalue() for i in [params, AIC, BIC]]
        dist_fit[i] = (params + AIC + BIC).replace("\n", ", ")
    
    display(describe)
    
    plt.title(title + str(start) + " to " + str(end), fontsize = 20)
    plt.axhline(0, color = "red", lw = .5, linestyle = "--")
    plt.axhspan(0, Selection_Mo_r.min().min(), facecolor = "red", alpha = 0.2) 
    plt.axhspan(0, Selection_Mo_r.max().max(), facecolor = "green", alpha = 0.2)
    plt.xticks(rotation = 45)
    for i, t in enumerate(plt.gca().xaxis.get_ticklabels()):
        if (i % 2) != 0:
            t.set_color("lightgreen")
        else:
            t.set_color("white")
    plt.yticks(np.arange(round(Selection_Mo_r.min().min(), 1), round(Selection_Mo_r.max().max(), 1), 0.05))
    plt.grid(alpha = 0.5, linestyle = "--", color = "grey")
    plt.show()

    return describe, dist_fit

##############################################################################################################################################################################################################################################################################`

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

##############################################################################################################################################################################################################################################################################

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


##############################################################################################################################################################################################################################################################################