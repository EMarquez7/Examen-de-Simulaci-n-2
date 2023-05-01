"""
# -- ------------------------------------------------------------------------------------------------------------------------ -- #   
# -- project: S&P500-Risk-Optimized-Portfolios-PostCovid-ML.                                                                  -- #           
# -- script: data.py : Python script with data functionality for the proj.                                                    -- #                 
# -- author: EstebanMqz                                                                                                       -- #  
# -- license: CC BY 3.0                                                                                                       -- #
# -- repository: https://github.com/EstebanMqz/SP500-Risk-Optimized-Portfolios-PostCovid-ML/blob/main/data.py                 -- #           
# -- ------------------------------------------------------------------------------------------------------------------------ -- #  
"""

from os import path
#Dependencies
import functions as fn
import visualizations as vs

#Libraries in data.py
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

# -- ----------------------------------------------------------------------------------------------- data ------------------------------------------------------------------------------- -- #

def library_install(requirements_txt):
    """Install requirements.txt file."""
    import os
    import warnings
    warnings.filterwarnings("ignore")
    os.system(f"pip install -r {requirements_txt}")
    print("Requirements installed.")
    with open("requirements.txt", "r") as f:
        print(f.read())


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
    """
    start = (datetime.datetime.now() - datetime.timedelta(days = 365 * years)).strftime("%Y-%m-%d") 
    end = (datetime.datetime.now()).strftime("%Y-%m-%d") 
    try:
                data = pd.DataFrame(YahooFinancials(ticker).get_historical_price_data(start_date = start,
                                                                        end_date = end, time_interval="daily")[ticker]["prices"])
                data["formatted_date"] = pd.to_datetime(data["formatted_date"])
                data = data.set_index("formatted_date")
                data = data.drop(["date", "high", "low", "open", "close", "volume"], axis = 1)  #OHLCV               
                data = data.rename(columns = {"adjclose" : ticker}) 

    except KeyError:
        pass
    except TypeError:
        pass
    except ValueError:
        pass

    return data



