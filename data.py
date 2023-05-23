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

# -- ----------------------------------------------------------------------------------------------- Data ------------------------------------------------------------------------------- -- #


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
    """ Function to describe dataframes
    Parameters:
    ----------
    df: pd.DataFrame
        prices dataframe to get stats for.
    output: str
        'prices', 'log_returns' or 'simple_returns' to get stats for and plot.
    rf: float
        risk free rate.
    start: str
        start date to retrieve from df.
    end: str
        end date to implement from df.

    Returns:
    -------
    'prices': in case prices is given as output, it returns [0]: prices_description, [1]: prices_df.iloc(start:end).
    'log_returns': in case log_returns is given as output, it returns [0]: logret_description, [1]: prices_df.iloc(start:end), [2]: logrets.
    'simple_returns': in case simple_returns is given as output, it returns [0]: simple_ret_description}, [1]: prices_df.iloc(start:end), [2]: simple_rets.
    """
    df = df.loc[start:end]
    description = df.describe()
    description = description.iloc[3:, :]
    description.index.names = [output]
    description.columns.names = ['Companies']

    if output == 'prices' :
        description.loc['Mean'] = df.mean()
        description.loc['Yr_Std'] = df.std()
        description.loc['Total_Change'] = (df.iloc[-1]/ df.iloc[0]) - 1
        description.loc['var97.5(-)'] = fn.VaR(df, alpha = 0.025)
        description.loc['var2.5(+)'] = fn.VaR(df, alpha = 0.975)
        description.loc['Price_skew'] = df.skew()
        description.loc['Price_kurtosis'] = df.kurtosis()

    if  output == "simple_returns" :
        returns = df.pct_change().iloc[1:, :].dropna(axis = 1)
        description.loc['Simple_skew'] = returns.skew()
        description.loc['Simple_kurtosis'] = returns.kurtosis()
        r_simple_acum = ((1+returns).cumprod()-1).iloc[-1]
        description.loc['Accum_Simple'] = r_simple_acum
        description.loc['Yr_Return'] = returns.mean() *252
        description.loc['Yr_Std'] = returns.std() * np.sqrt(252)
        description.loc['var97.5(-)'] = fn.VaR(returns, alpha = 0.025)
        description.loc['var2.5(+)'] = fn.VaR(returns, alpha = 0.975)
        description.loc['sharpe'] = (description.loc['Yr_Return'] - rf) / (returns.std() * np.sqrt(252))
        description.loc['sortino'] = (description.loc['Yr_Return'] - rf) / (returns[returns < 0].std() * np.sqrt(252))
        description.loc['Yr_MaxDrawdown'] = description.loc['Accum_Simple'] / (1 + description.loc['Accum_Simple']).cummax() - 1

    if  output == "log_returns" :
        returns = np.log(df).diff().iloc[1:, :].dropna(axis = 1)   
        description.loc['Logret_skew'] = returns.skew()
        description.loc['Logret_kurtosis'] = returns.kurtosis()
        description.loc['Accum_Logret'] = (returns.cumsum().apply(np.exp)-1).iloc[-1]
        description.loc['Yr_Return'] = returns.mean() *252
        description.loc['Yr_Std'] = returns.std() * np.sqrt(252)
        description.loc['var97.5(-)'] = fn.VaR(returns, alpha = 0.025)
        description.loc['var2.5(+)'] = fn.VaR(returns, alpha = 0.975)
        description.loc['sharpe'] = (description.loc['Yr_Return'] - rf) / (returns.std() * np.sqrt(252))
        description.loc['sortino'] = (description.loc['Yr_Return'] - rf) / (returns[returns < 0].std() * np.sqrt(252))
        description.loc['Yr_MaxDrawdown'] = description.loc['Accum_Logret'] / (1 + description.loc['Accum_Logret']).cummax() - 1


    if output == 'prices':
        return description.T.sort_values(by = 'Total_Change', ascending = False), df
    elif output == 'simple_returns':
        return description.T.sort_values(by = 'sortino', ascending = False), df, returns
    elif output == 'log_returns':
        return description.T.sort_values(by = 'sortino', ascending = False), df, returns
    
    else:
        return print("Error: output must be 'prices' or 'log_returns'.")



