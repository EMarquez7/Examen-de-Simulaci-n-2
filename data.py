"""
# -- -------------------------------------------------------------------------------------------------------- -- #   
# -- project: S&P500-Risk-Optimized-Portfolios-PostCovid-ML.                                                  -- #           
# -- script: data.py : Python script with data functionality for the proj.                                    -- #                 
# -- author: EstebanMqz                                                                                       -- #  
# -- license: CC BY 3.0                                                                                       -- #
# -- repository: https://github.com/EstebanMqz/SP500-Risk-Optimized-Portfolios-PostCovid-ML/blob/main/data.py -- #           
# -- -------------------------------------------------------------------------------------------------------- -- #  
"""

from os import path
#Dependencies
import functions as fn
import visualizations as vs

#Libraries in data.py
import numpy as np
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None
              ,"display.max_colwidth", None, "display.width", None)

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("dark_background")


import plotly.graph_objects as go 
import plotly.express as px

import scipy
import scipy.stats as st
from scipy import optimize
from scipy.optimize import minimize

import sklearn
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import re
from yahoofinancials import YahooFinancials 
from tabulate import tabulate
import IPython.display as d
import IPython.core.display

import ast
from io import StringIO
from fitter import Fitter, get_common_distributions, get_distributions 
import logging

import datetime 
import time
import warnings

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

# -- ------------------------------------------------------------------------------------------------------------------- Data ------------------------------------------------------------------------------------------------------------------ -- #


###########################################################################################################################################################################################################################################

def get_historical_price_data(ticker, years):
    """
    Function to retrieve Adj. Closes data from OHLCV of ticker(s) from Yahoo_Financials for n years backwards from today's date.
    It returns a dataframe with the Adj. Close(s) of ticker(s) with datetime as index.
    Parameters:
    ----------
    ticker : str
        Ticker of the stock(s) to be downloaded as a string or str list, e.g: ["ticker_1", "ticker_2", ... , "ticker_n"]
    years : int
        Number of years for data download from today's date backwards.

    Returns:
    -------
    data : pandas.DataFrame
        Adj. Close(s) of ticker(s) with datetime as index, modify function to obtain        
    """
    start = (datetime.datetime.now() - datetime.timedelta(days = 365 * years)).strftime("%Y-%m-%d") 
    end = (datetime.datetime.now()).strftime("%Y-%m-%d") #Today

    try:
                data = pd.DataFrame(YahooFinancials(ticker).get_historical_price_data(start_date = start,
                                                                        end_date = end, time_interval="daily")[ticker]["prices"])
                data["formatted_date"] = pd.to_datetime(data["formatted_date"])
                data = data.set_index("formatted_date")
                data = data.drop(["date", "high", "low", "open", "close", "volume"], axis = 1)               
                data = data.rename(columns = {"adjclose" : ticker})

    except KeyError:
        pass
    except TypeError:
        pass
    except ValueError:
        pass

    return data

###########################################################################################################################################################################################################################################
def DQR(data):
    """
    Function to create a data quality report for a given dataframe.
    It returns a dataframe with the following columns:
    columns=[['Column_Name','Data_Type', 'Unique_Values', 'Missing_Values', 'Zero_Values', 'Outliers', 'Unique_Outliers']]    
    Parameters:
    ----------
    data: pd.DataFrame() :
         Dataframe with the data to be analyzed.
    Returns:
    -------
    data_quality_report: pd.DataFrame() :
                         Dataframe with the data quality report with given columns as variables.
    """

    data = data.copy()
    data_quality_report = pd.DataFrame(columns=['Data_Type', 'Unique_Values', 'Missing_Values', 'Zero_Values', 'Outliers', 'Unique_Outliers'])
    data_quality_report['Columns'] = data.columns
    data_quality_report['Data_Type'] = data.dtypes.values
    data_quality_report['Unique_Values'] = [data[col].nunique() for col in data.columns]
    data_quality_report['Missing_Values'] = [data[col].isna().sum() for col in data.columns]
    data_quality_report['Zero_Values'] = [(data[col] == 0).sum() for col in data.columns]
    data_quality_report['Outliers'] = [len(data[col][data[col] > data[col].mean() + 3*data[col].std()]) for col in data.columns]
    data_quality_report['Unique_Outliers'] = [data[col][data[col] > data[col].mean() + 3*data[col].std()].nunique() for col in data.columns]
    data_quality_report = data_quality_report.set_index('Columns')
    return data_quality_report


###########################################################################################################################################################################################################################################
def data_describe(df, output, rf, start, end):
    """ Function that returns an adavnaced describe dataframee for time series values simple_returns or log_returns with a start and end date in case it is a subset.
    Otherwise, dates would be the original start and end dates of the dataframe. 

    Parameters:
    ----------
    df: pd.DataFrame
        prices dataframe to get stats for.
    output: str
        'prices', 'log_returns' or 'Simple' to get stats for and plot.
    rf: float
        risk free rate or None if Sharpe, Sortino, Calmar and Burker ratios are not needed.
    start: str
        start date to retrieve from df.
    end: str
        end date to implement from df.

    Returns:
    -------
    'Prices': If output == 'Prices', returns: stats for prices.
    'Simple': If output == 'Simple', returns: stats for simple_returns, simple_returns
    'log_returns': If output == 'Log_returns', returns: stats for log_returns, log_returns  
    
    """
    df = df.loc[start:end]
    description = df.describe()
    description = description.iloc[3:, :]
    description.index.names = [output]
    description.columns.names = ['Companies']

    if output == 'Prices' :
        description.loc['Mean'] = df.mean()
        description.loc['Yr_Std'] = df.std()
        description.loc['Total_Change'] = (df.iloc[-1]/ df.iloc[0]) - 1
        description.loc['var97.5(-)'] = fn.VaR(df, alpha = 0.025)
        description.loc['var2.5(+)'] = fn.VaR(df, alpha = 0.975)
        description.loc['Price_skew'] = df.skew()
        description.loc['Price_kurtosis'] = df.kurtosis()

    if  output == "Simple" :
        simple_returns = df.pct_change().iloc[1:, :].dropna(axis = 1)
        description.loc['Simple_skew'] = simple_returns.skew()
        description.loc['Simple_kurtosis'] = simple_returns.kurtosis()
        r_simple_acum = ((1+simple_returns).cumprod()-1).iloc[-1]
        description.loc['Accum_Simple'] = r_simple_acum
        description.loc['Yr_Return'] = simple_returns.mean() *252
        description.loc['Yr_Std'] = simple_returns.std() * np.sqrt(252)
        description.loc['var97.5(-)'] = fn.VaR(simple_returns, alpha = 0.025)
        description.loc['var2.5(+)'] = fn.VaR(simple_returns, alpha = 0.975)
        description.loc['Yr_MaxDrawdown'] = description.loc['Accum_Simple'] / (1 + description.loc['Accum_Simple']).cummax() - 1
        if rf is not None:
            description.loc['Sharpe'] = (description.loc['Yr_Return'] - rf) / (simple_returns.std() * np.sqrt(252))
            description.loc['Sortino'] = (description.loc['Yr_Return'] - rf) / (simple_returns[simple_returns < 0].std() * np.sqrt(252))
            description.loc['Calmar'] = description.loc['Yr_Return'] / abs(description.loc['Yr_MaxDrawdown'])
            description.loc['Burke'] = (description.loc['Yr_Return'] - rf) / abs(description.loc['Yr_MaxDrawdown'])

        if rf is None:
            description.loc['Sharpe'] = np.nan
            description.loc['Sortino'] = np.nan
            description.loc['Calmar'] = np.nan
            description.loc['Burke'] = np.nan
            description = description.drop(['Sharpe', 'Sortino', 'Calmar', 'Burke'], axis = 0)


    if  output == "Log_returns" :
        log_returns = np.log(df).diff().iloc[1:, :].dropna(axis = 1)   
        description.loc['Logret_skew'] = log_returns.skew()
        description.loc['Logret_kurtosis'] = log_returns.kurtosis()
        description.loc['Accum_Logret'] = (log_returns.cumsum().apply(np.exp)-1).iloc[-1]
        description.loc['Yr_Return'] = log_returns.mean() *252
        description.loc['Yr_Std'] = log_returns.std() * np.sqrt(252)
        description.loc['var97.5(-)'] = fn.VaR(log_returns, alpha = 0.025)
        description.loc['var2.5(+)'] = fn.VaR(log_returns, alpha = 0.975)
        description.loc['Yr_MaxDrawdown'] = description.loc['Accum_Logret'] / (1 + description.loc['Accum_Logret']).cummax() - 1
        if rf is not None:
            description.loc['Sharpe'] = (description.loc['Yr_Return'] - rf) / (log_returns.std() * np.sqrt(252))
            description.loc['Sortino'] = (description.loc['Yr_Return'] - rf) / (log_returns[log_returns < 0].std() * np.sqrt(252))
            description.loc['Calmar'] = description.loc['Yr_Return'] / abs(description.loc['Yr_MaxDrawdown'])
            description.loc['Burke'] = (description.loc['Yr_Return'] - rf) / abs(description.loc['Yr_MaxDrawdown'])
        if rf is None:
            description.loc['Sharpe'] = np.nan
            description.loc['Sortino'] = np.nan
            description.loc['Calmar'] = np.nan
            description.loc['Burke'] = np.nan
            description = description.drop(['Sharpe', 'Sortino', 'Calmar', 'Burke'], axis = 0)

    if output == 'Prices':
        return description
    elif output == 'Simple':
        return description, simple_returns
    elif output == 'Log_returns':
        return description, log_returns
    
    else:
        return print("Error: output must be 'Prices', 'Simple', 'Log_returns'.")



