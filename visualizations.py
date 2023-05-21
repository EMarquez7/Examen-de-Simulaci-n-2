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
import visualizations as vs

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
import seaborn as sns
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
import IPython.core.display

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

def BoxHist(data, bins, color, label, title):
    """Boxplot and Histogram for given data
    ----------
    data : DataFrame
        Data to plot.
    bins : int
        Number of bins for histogram.
    color : str
        Color for plots.
    x1_label : str
        x1_label for boxplot.
    x2_label : str
        x2_label for histogram.
    title : str
        Title for both plots.
    Returns
    -------
    Boxplot and Histogram of data
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    data.plot.box(ax=ax1, color=color, vert=False)
    plt.text(0.05, 0.05, data.describe().round(6).to_string(), transform=ax1.transAxes)
    ax1.set_xlabel(label)
    sns.histplot(data, bins=bins, kde=True, alpha=0.5, ax=ax2).legend().remove()
    for patch in ax2.patches:
        patch.set_facecolor(color)
    ax2.set_yticklabels(["{:.2f}%".format(x/10000) for x in ax2.get_yticks()])
    ax2.set_ylabel("Probability")
    ax2.set_xlabel(label)
    fig.suptitle(str(label) + title, fontsize=12, fontweight="bold")
    ax1.grid(color="gray", linestyle="--"), ax2.grid(color="lightgray", linestyle="--")

    plt.show()


##############################################################################################################################################################################################################################################################################
def selection_data(dataframe, r, rf, best, start, end):
    """
    Function that calculates Annualized Returns and Std. Deviation for a given dataframe in order to obtain 
    n_best Sharpe & Sortino Ratios with a risk-free rate.
    Parameters:
    ----------
    dataframe : dataframe
        Dataframe for the model.
    r : str
        Type of return for the model: "Simple" or "Log".
    rf : float
        Yield Curve Rates (see refs.) from model's end date on a Yearly basis for n_best Sharpe & Sortino Ratios. 
    best: int
        No° of best Sharpe & Sortino Ratios to integrate in selection (if dataframe cols. <= best (No.°), then all cols. are selected).
    start : str
        Start date selected from dataframe.
    end : str
        End date selected from dataframe.
        
    Returns:
    -------
    summary : dataframe
            Annualized Returns, Std. deviations and best Ratios for Sharpe & Sortino with a Sortino Selection for Xi with dataframe and dates.
    """
    dataframe_date = dataframe.loc[start:end]
    if  r == "Simple" :
        returns = dataframe_date.pct_change().iloc[1:, :].dropna(axis = 1)
    if  r == "Log" :
        returns = np.log(dataframe_date).diff().iloc[1:, :].dropna(axis = 1)   
        
    mean_ret = returns.mean() * 252
    sigma = returns.std() * np.sqrt(252)

    if r != "Simple" and r != "Log" :
        print("Aborted: Please select a valid Return type: 'Simple' or 'Log'. selection_data help command: help(vs.selection_data)")

    sharpe = (mean_ret - rf) / sigma 
    sortino = (mean_ret - rf) / ( (returns[returns < 0].std()) * np.sqrt(252) )
      
    summary = pd.DataFrame({"$\mu_{i{yr}}$" : mean_ret, "$\sigma_{yr}$" : returns.std() * np.sqrt(252),
                            "$R_{Sharpe}$" : sharpe, "$R_{Sortino}$" : sortino})
    summary = summary.nlargest(best, "$R_{Sharpe}$").nlargest(best, "$R_{Sortino}$")
    
    return dataframe_date, returns, summary

##############################################################################################################################################################################################################################################################################

def Selection_R_SLog_Plot(data, rf, best, start, execution_date, r_jump):
    Sortino25_S = vs.selection_data(data, "Simple", rf, best, start, execution_date)[1]
    Sortino25_Log = vs.selection_data(data, "Log", rf, best, start, execution_date)[1]

    fig, ax = plt.subplots(1, 2, figsize = (30, 12))
    Sortino25_S.plot.bar(ax = ax[0], rot = 45, fontsize = 15, grid = True, linewidth = 1)
    Sortino25_Log.plot.bar(ax = ax[1], rot = 45, fontsize = 15, grid = True, linewidth = 1)

    ax[0].set_title("Selection of " + str(best) + " $X_i$ datasets from $S&P 500$ Population with $R_{t}$", fontsize = 20)
    ax[1].set_title("Selection of " + str(best) + " $X_i$ datasets from $S&P 500$ Population with $Ln(r_{t})$", fontsize = 20)

    ax[0].set_yticks(np.arange(0, Sortino25_S.max().max() + r_jump, r_jump))
    ax[1].set_yticks(np.arange(0, Sortino25_Log.max().max() + r_jump, r_jump))

    ax[0].set_yticklabels(np.arange(0, Sortino25_S.max().max() + r_jump, r_jump).round(2), fontsize = 9)
    ax[1].set_yticklabels(np.arange(0, Sortino25_Log.max().max() + r_jump, r_jump).round(2), fontsize = 9)

    ax[0].set_xticklabels(Sortino25_S.index, rotation = 45, fontsize = 12)
    ax[1].set_xticklabels(Sortino25_Log.index, rotation = 45, fontsize = 12)

    ax[0].grid(color='gray', linestyle='--')
    ax[1].grid(color='gray', linestyle='--')

    for i, t in enumerate(ax[0].get_xticklabels()):
        if i % 2 == 0:
            t.set_color('lightgreen')
        else:
            t.set_color('white')
    for i, t in enumerate(ax[1].get_xticklabels()):
        if i % 2 == 0:
            t.set_color('lightgreen')
        else:
            t.set_color('white')
    #Show figure
    return plt.show()

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
        Continuous Distributions to fit on datasets Xi
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
    describe.replace("\n", "")

    dist_fit = np.empty(len(Selection_Mo_r.columns), dtype=object)
    
    for i in range(0, len(Selection.columns)):
        f = Fitter(pd.DataFrame(Selection_Mo_r.iloc[:, i]), distributions = dist, timeout=5)
        f.fit()
        params, AIC, BIC = [StringIO() for i in range(3)]
        (print(f.get_best(), file=params)), (print(f.get_best(method="aic"), file=AIC)), (print(f.get_best(method="bic"), file=BIC))
        params, AIC, BIC = [i.getvalue() for i in [params, AIC, BIC]]
        dist_fit[i] = (params + AIC + BIC).replace("\n", ", ")
    
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
    IPython.core.display.clear_output() 
    return describe, dist_fit, plt.show()

##############################################################################################################################################################################################################################################################################`

def Optimizer(Assets, index, rf, title):
    Asset_ret = (Assets.pct_change()).iloc[1:, :].dropna(axis = 1)
    index_ret = index.pct_change().iloc[1:, :].dropna(axis = 1)
    index_ret = index_ret[index_ret.index.isin(Asset_ret.index)]

    mean_ret = Asset_ret.mean() * 252
    cov = Asset_ret.cov() * 252

    N = len(mean_ret)
    w0 = np.ones(N) / N
    bnds = ((0, None), ) * N
    cons = {"type" : "eq", "fun" : lambda weights : weights.sum() - 1}

    def Max_Sharpe(weights, Asset_ret, rf, cov):
        rp = np.dot(weights.T, Asset_ret)
        sp = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        RS = (rp - rf) / sp
        return -(np.divide(np.subtract(rp, rf), sp))
    
    def Min_Var(weights, cov):
        return np.dot(weights.T, np.dot(cov, weights)) 
    
    # def Min_Traynor(weights, Asset_ret, rf, cov):
    #     rp = np.dot(weights.T, Asset_ret)
    #     varp = np.dot(weights.T, np.dot(cov, weights))
    #     RT = (rp - rf) / sp
    #     return -(np.divide(np.subtract(rp, rf), sp))
    
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    opt_EMV = optimize.minimize(Max_Sharpe, w0, (mean_ret, rf, cov), 'SLSQP', bounds = bnds,
                                constraints = cons, options={"tol": 1e-10})
    
    W_EMV = pd.DataFrame(np.round(opt_EMV.x.reshape(1, N), 4), columns = Asset_ret.columns, index = ["Weights"])
    W_EMV[W_EMV <= 0.0] = np.nan
    W_EMV.dropna(axis = 1, inplace = True)

    RAssets = Asset_ret[Asset_ret.columns[Asset_ret.columns.isin(W_EMV.columns)]]
    # MuAssets = mean_ret[mean_ret.index.isin(W_EMV.columns)]
    R_EMV = pd.DataFrame((RAssets*W_EMV.values).sum(axis = 1), columns = ["$r_{Sharpe_{Arg_{max}}}$"])
    index_ret.rename(columns={index_ret.columns[0]: "$r_{mkt}$" }, inplace=True)
    R_EMV.insert(1, index_ret.columns[0], index_ret.values)

    Muopt_EMV = np.dot(opt_EMV.x.T, mean_ret) 
    Sopt_EMV = np.sqrt(np.dot(opt_EMV.x.T, np.dot(cov, opt_EMV.x)))
    Beta_EMV = np.divide((np.cov(R_EMV.iloc[0], R_EMV.iloc[1])[0][1]), R_EMV.iloc[1].var())
    SR_EMV = (Muopt_EMV - rf) / Sopt_EMV

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    opt_MinVar = optimize.minimize(Min_Var, np.ones(N) / N, (cov,), 'SLSQP', bounds = bnds,
                                   constraints = cons, options={"tol": 1e-10})

    W_MinVar = pd.DataFrame(np.round(opt_MinVar.x.reshape(1, N), 4), columns = Asset_ret.columns, index = ["Weights"])
    W_MinVar[W_MinVar <= 0.0] = np.nan
    W_MinVar.dropna(axis = 1, inplace = True)

    RAssets_MinVar = Asset_ret[Asset_ret.columns[Asset_ret.columns.isin(W_MinVar.columns)]]
    R_MinVar = pd.DataFrame((RAssets_MinVar*W_MinVar.values).sum(axis = 1), columns = ["$r_{Var_{Arg_{min}}}$"])
    R_EMV.insert(2, R_MinVar.columns[0], R_MinVar.values)

    Muopt_MinVar = np.dot(opt_MinVar.x.T, mean_ret) 
    Sopt_MinVar = np.sqrt(np.dot(opt_MinVar.x.T, np.dot(cov, opt_MinVar.x)))
    Beta_MinVar = np.divide((np.cov(R_EMV.iloc[2], R_EMV.iloc[1])[0][1]), R_EMV.iloc[1].var())
    SR_MinVar = (Muopt_MinVar - rf) / Sopt_MinVar 

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #opt_Traynor = 
    
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    Mu, Sigma, Beta, SR = [Muopt_EMV, Muopt_MinVar], [Sopt_EMV, Sopt_MinVar], [Beta_EMV, Beta_MinVar], [SR_EMV, SR_MinVar]
    index = ["$r_{P{Sharpe_{Arg_{max}}}}$", "$r_{Var_{Arg_{min}}}$"]
    Popt = [pd.DataFrame({"$\mu_P$" : Mu[i], "$\sigma_P$" : Sigma[i], "$\Beta_{P}$": Beta[i], "$r_{Sharpe_{Arg_{max}}}$" : SR[i]},
                          index = [index[i]]) for i in range(0, len(Mu))]
    
    Popt[0].index.name = title
    Popt[1].index.name = title
    R_EMV = R_EMV[[R_EMV.columns[1], R_EMV.columns[2], R_EMV.columns[0]]]
    #Get the cumulative returns with cumsum for rmkt, rEMV and rMinVar
    accum = R_EMV.cumsum()

    Argmax = [d.Markdown(tabulate(Popt[i], headers = "keys", tablefmt = "pipe")) for i in range(0, len(Popt))]
    R_EMV = d.Markdown(tabulate(R_EMV, headers = "keys", tablefmt = "pipe"))
    
    return Argmax, R_EMV, accum

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
    # ax.legend(loc = "upper left", fontsize = 15)
    # ax.grid(True)
    # ax.grid(which='major', color='gray', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=90)
    plt.show()


##############################################################################################################################################################################################################################################################################