##  S&P 500 Risk Optimizations Forecast

### Description: 

Since Covid, data has changed in most industries with few exceptions and the markets are just another example.<br>

For this reason, the present repository automates tasks that deliver a full understanding until the user *(you)* execution's date. <br>
Furthermore, it generates a variety of optimizations whose purpose is to forecast the mentioned period while being able to incorporate newly generated data with the usage of the snippets provided by the scripts, they could be executed by clicking: <br>

[![S&P500-Optimizations-Forecast](https://img.shields.io/badge/Notebook-Run>All-black?style=square&logo=github&logoColor=black)](https://github.com/EstebanMqz/SP500-Risk-Optimizations-Forecast/blob/main/SP500-Risk-Optimized-Portfolios-ML.ipynb) <br>

#### **Repository Tools:**
<font>
<Details>
<Summary> <b>Click to expand</b> </Summary>

###### Actions: [![Repo-Visualization-Badge](https://img.shields.io/badge/Action-Visualization-020521?style=square&logo=github&logoColor=white)](https://githubnext.com/projects/repo-visualization)
###### Main Text-Editor: [![VSCode-Badge](https://img.shields.io/badge/VSCode-007ACC?style=square&logo=visual-studio-code&logoColor=white)](https://code.visualstudio.com/)&nbsp;[![Jupyter-Badge](https://img.shields.io/badge/Jupyter-F37626?style=square&logo=Jupyter&logoColor=white)](https://jupyter.org/try) 

###### Language: [![Python-Badge](https://img.shields.io/badge/Python-3776AB.svg?style=square&logo=Python&logoColor=green)](https://www.python.org)[![Markdown-Badge](https://img.shields.io/badge/Markdown-000000.svg?style=square&logo=Markdown&logoColor=white)](https://www.markdownguide.org)[![yaml-Badge](https://img.shields.io/badge/YAML-000000?style=square&logo=yaml&logoColor=red)](https://yaml.org)

###### Libraries:  [![Numpy-Badge](https://img.shields.io/badge/Numpy-013243?style=square&logo=numpy&logoColor=white)](https://numpy.org)  [![Pandas-Badge](https://img.shields.io/badge/Pandas-150458?style=square&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![Scipy-Badge](https://img.shields.io/badge/Scipy-darkblue?style=square&logo=scipy&logoColor=white)](https://www.scipy.org)  [![Sklearn-Badge](https://img.shields.io/badge/Sklearn-red?style=square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)  [![Fitter-Badge](https://img.shields.io/badge/Fitter-000000?style=square&logo=python&&logoColor=yellow)](https://fitter.readthedocs.io/en/latest/)  [![Matplotlib-Badge](https://img.shields.io/badge/Matplotlib-40403f?style=square&logo=python&logoColor=blue)](https://matplotlib.org)  [![Seaborn-Badge](https://img.shields.io/badge/Seaborn-40403f?style=square&logo=python&logoColor=blue)](https://seaborn.pydata.org)

###### Interface: [![React-Badge](https://img.shields.io/badge/React-61DAFB?style=square&logo=react&logoColor=black)](https://create-react-app.dev)

###### Version Control: [![GitHub-Badge](https://img.shields.io/badge/GitHub-100000?style=square&logo=github&logoColor=white)](https://github.com)&nbsp;[![Git-Badge](https://img.shields.io/badge/Git-F05032.svg?style=square&logo=Git&logoColor=white)](https://git-scm.com)  
[![Git-Commads](https://img.shields.io/badge/Git%20Commands-gray?style=square&logo=git&logoColor=white)](https://github.com/EstebanMqz/Git-Commands)

###### License: [![Creative Commons BY 3.0](https://img.shields.io/badge/License-CC%20BY%203.0-yellow.svg?style=square&logo=creative-commons&logoColor=white)](https://creativecommons.org/licenses/by/3.0/)

</Details>

##### Work Contact:
<font>
<Details>
<Summary> <b>Click to expand</b> </Summary>

[![Website](https://img.shields.io/badge/Website-ffffff?style=square&logo=opera&logoColor=red)](https://estebanmqz.com) [![LinkedIn](https://img.shields.io/badge/LinkedIn-041a80?style=square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/esteban-m65381722210212839/) [![Portfolio](https://img.shields.io/badge/Github-Portfolio-010b38?style=square&logo=github&logoColor=black)](https://estebanmqz.github.io/Portfolio/) [![E-mail](https://img.shields.io/badge/Business-Mail-052ce6?style=square&logo=mail&logoColor=white)](mailto:esteban@esteban.com)
<br>

![GitHub Logo](https://github.com/EstebanMqz.png?size=50) [![Github](https://img.shields.io/badge/Github-000000?style=square&logo=github&logoColor=white)](https://github.com/EstebanMqz) 

</Details>

##### **Diagram:**
[![Repository](https://img.shields.io/badge/Repository-0089D6?style=square&logo=microsoft-azure&logoColor=white)](https://mango-dune-07a8b7110.1.azurestaticapps.net/?repo=EstebanMqz%2FSP500-Risk-Optimizations-Forecast)

<img src="diagram.svg" width="280" height="280">

#### Table of Contents: 
The processes made are illustrated by the following:

![Alt text](images/ToC.jpg)

*Note: Newly generated dataframes is saved in `Data` and visualizations in `Images` folders respectively.* <br>

### Methodology: 

After the Virtual Environment [fn.get_requirements](https://github.com/EstebanMqz/SP500-Risk-Optimized-Portfolios-PostCovid-ML/blob/main/functions.py) is set up and the libraries installed by executing: <br>

```python
fn.library_install("requirements.txt")
```
1. Individual Datasets:

Data Extraction techniques are made for its cleaning and Exploration followed by Descriptive Statistics<br>
$x_i\in [x_1,x_{500}]$ $\hookrightarrow$ $S$&$P$ $500$ and $x_{j\neq i}\in$ $[x_1,x_{25}]_{{R_{Sortino_{+_{25}}}}}$ $\subset$ $x_i$, to show what and how it has happened. <br>

They incorporate to [`pandas.DataFrame.describe`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html) the following variables for Simple $R_t$ and Log Returns $r_t$ in order to be visualized:<br>
+ $\mu_{y}$
+ $\sigma_{y}$
+ $\sum_{t=1}^{n} R_t$ $\&$ $r_t$
+ $VaR_{97.5}$ - $VaR_{2.5}$
+ $MDD$
+ $R_{Sharpe}$, $R_{Sortino}$, $R_{Calmar}$, $R_{Burke}$.<br>

*Visualizations include common periodicity resampling periods.* <br>

As result, the following optimizations are made to subsequently generate simulations with what would have been its past behavior, concluding with the optimization's forecast out of the simulated data: <br>

![Alt text](images/Methodology.jpg)
---
#### **References:**
<font>
<Details>
<Summary> <b>Click to expand</b> </Summary>

##### </u> Pandas: </u> <br>

[`pd.isin`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isin.html) [`pd.df.sample`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html) [`pd.df.fillna`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html) [`pd.df.resample`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html) [`pandas.DataFrame.describe`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)

##### </u> Numpy: </u> <br>

[`np.quantile`](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html) [`np.arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html) [`np.add`](https://numpy.org/doc/stable/reference/generated/numpy.add.html) [`np.subtract`](https://numpy.org/doc/stable/reference/generated/numpy.subtract.html) [`np.dot`](https://numpy.org/doc/stable/reference/generated/numpy.dot.html) [`np.divide`](https://numpy.org/doc/stable/reference/generated/numpy.divide.html) [`np.cov`](https://numpy.org/doc/stable/reference/generated/numpy.cov.html) [`np.power`](https://numpy.org/doc/stable/reference/generated/numpy.power.html) <br>

##### </u> Stats: </u> <br>

[`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html) [`scipy.stats.rv_continuous`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html) [`scipy.stats.rv_discrete`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html) [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)

##### </u> Sklearn: </u> <br>

[`sklearn.model_selection.GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) [`Hyper-parameters Exhaustive GridSearchCV`](https://scikit-learn.org/stable/modules/grid_search.html) <br>

[`sklearn.neighbors.KernelDensity`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html) [`sklearn.neighbors.KernelDensity.fit`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity.fit) <br>

[`sklearn.neighbors.KernelDensity.score_samples`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity.score_samples) [`sklearn.metrics`](https://scikit-learn.org/stable/modules/model_evaluation.html)

###### Other Libraries: <br>

+ [`fitter`](https://fitter.readthedocs.io/en/latest/index.html)<br>

+ [`statsmodels`](https://www.statsmodels.org/stable/index.html)<br>

###### *Other References*: <br>
+ [`Convolution of Distributions`](https://en.wikipedia.org/wiki/Convolution_of_probability_distributions)
+ [`*i.i.d* $x \sim X$`](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)<br>
+ [`LaTeX`](https://en.wikipedia.org/wiki/List_of_mathematical_symbols_by_subject)</br>
+ [`Daily Treasury Par Yield Curve Rates`](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value_month=202304)<br>
+ [`$S$&$P$ $500$ Symbols`](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)</br>
