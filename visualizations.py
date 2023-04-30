"""
# -- ------------------------------------------------------------------------------------------------------------------------ -- #   
# -- project: S & P 500 Risk Optimized Portfolios from 14 Apr (2020-2023)                                                     -- #           
# -- script: data.py : Python script with visualizations functionalities                                                      -- #                 
# -- author: EstebanMqz                                                                                                       -- #  
# -- license: CC BY 3.0                                                                                                       -- #
# -- repository: https://github.com/EstebanMqz/SP500-Risk-Optimized-Portfolios-ML-Models/blob/main/visualizations.py          -- #           
# -- ------------------------------------------------------------------------------------------------------------------------ -- #  
"""

from os import path
#Dependencies
import functions as fn
import data as dt

#Libraries in visualizations.py
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


def MC_plot(simulations, E_V, xlabel , ylabel, i_capital, f_capital):
    """Plots a line plot of events for n Monte Carlo simulations.
    Parameters
    ----------
    simulation : pandas.DataFrame
        Dataframe of Monte Carlo simulations.
    E_V : pandas.DataFrame
        The expected value of the simulation.
    xlabel : str
        The label of the x-axis.
    ylabel : str
        The label of the y-axis.
    i_capital : int
        The initial value of the variable to be simulated.
    f_capital : int
        The final value of the variable simulated.

    Returns
    -------
    line_plot : matplotlib.pyplot
        Plot of events time-series for N MonteCarlo simulations.
    """

    ROI = round((E_V.iloc[-1][0] / E_V.iloc[0][0]) - 1, 8)
    color = ["red" if ROI < 0 else "green"][0]

    plt.style.use('dark_background')
    plt.rc('grid', linestyle="--", color='gray')
    plt.rc('ytick', labelsize=13, color='blue')
    plt.rc('xtick', labelsize=13, color = color)
    
    fig, ax = plt.subplots(figsize = (18, 10))
    ax.set_facecolor('black')
   
    sim.plot(ax = ax, xlabel = xlabel, ylabel = ylabel, title = ("Montecarlo Simulations"), linewidth = 0.15).legend().remove()
    ax.title.set_color('teal'), ax.title.set_size(20)

    #I&F_capital, negative-line & axhspan sim W/L. 
    plt.axhline(y = i_capital, color = "white", linewidth = .8)
    plt.axhline(y = f_capital, color = color, linewidth = .8)
    plt.axhline(y = 0, color = color, linewidth = 1.2)
    plt.axhspan(i_capital, f_capital, facecolor = color, alpha = 0.2)

    #E(V) and ROI.
    E_V.plot(ax = ax, color = color, linewidth = 1)
    plt.text(0.5, 0.5, "Expected ROI ≈ " + str(ROI), fontsize=13, color=color, transform=ax.transAxes, position = (0.8, 0.65))

    #Style.
    plt.grid(True)
    plt.grid(which='both', color='gray', linestyle='--', alpha = 0.8)

    #x-y labels and ticks.
    ax.xaxis.label.set_size(15), ax.yaxis.label.set_size(15)
    ax.xaxis.label.set_color('teal'), ax.yaxis.label.set_color('teal')
    
    #plt.xticks(plt.xticks(range(((sim.index[0]), sim.shape[0], freq="1d")).tolist()), [str(i) for i in range(0, len(sim), 10000)])
    #plt.yticks(range(0, int(round(simulations.max().max(), 0)), 10000), [str(i) for i in range(0, int(round(simulations.max().max(), 0)), 10000)])
    
    return plt.show()
