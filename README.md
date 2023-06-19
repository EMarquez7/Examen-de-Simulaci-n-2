<h1><div align="center"><font color= '#318f17'><b> Optimized Indexes Forecasts </b></font></div></h1>

<div align="right">

[![Creative Commons BY 3.0](https://img.shields.io/badge/License-CC%20BY%203.0-yellow.svg?style=square&logo=creative-commons&logoColor=white)](https://creativecommons.org/licenses/by/3.0/)

<Details> <Summary> <i> <font color= '#ff9100'> Click to expand: </font> </i>

![S&P500-Optimizations-Forecast](https://img.shields.io/badge/Author's_Contact-Financial_Eng._Esteban_Márquez-black?style=square&logo=github&logoColor=black) </Summary>

[![Website](https://img.shields.io/badge/Website-ffffff?style=square&logo=opera&logoColor=red)](https://estebanmqz.com) [![LinkedIn](https://img.shields.io/badge/LinkedIn-041a80?style=square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/esteban-m65381722210212839/) [![Portfolio](https://img.shields.io/badge/Github-Portfolio-010b38?style=square&logo=github&logoColor=black)](https://estebanmqz.github.io/Portfolio/) [![E-mail](https://img.shields.io/badge/Business-Mail-052ce6?style=square&logo=mail&logoColor=white)](mailto:esteban@esteban.com)</br>

![GitHub Logo](https://github.com/EstebanMqz.png?size=50) [![Github](https://img.shields.io/badge/Github-000000?style=square&logo=github&logoColor=white)](https://github.com/EstebanMqz)
</Details>

<Details> <Summary> <i> <font color= '#ff9100'> Repository Tools: </font> </i> </Summary>

###### Actions: [![Repo-Visualization-Badge](https://img.shields.io/badge/Action-Visualization-020521?style=square&logo=github&logoColor=white)](https://githubnext.com/projects/repo-visualization)
###### Main Text-Editor: [![VSCode-Badge](https://img.shields.io/badge/VSCode-007ACC?style=square&logo=visual-studio-code&logoColor=white)](https://code.visualstudio.com/)&nbsp;[![Jupyter-Badge](https://img.shields.io/badge/Jupyter-F37626?style=square&logo=Jupyter&logoColor=white)](https://jupyter.org/try)
###### Language: [![Python-Badge](https://img.shields.io/badge/Python-3776AB.svg?style=square&logo=Python&logoColor=green)](https://www.python.org)[![Markdown-Badge](https://img.shields.io/badge/Markdown-000000.svg?style=square&logo=Markdown&logoColor=white)](https://www.markdownguide.org)[![yaml-Badge](https://img.shields.io/badge/YAML-000000?style=square&logo=yaml&logoColor=red)](https://yaml.org)  
###### Libraries: [![Numpy-Badge](https://img.shields.io/badge/Numpy-013243?style=square&logo=numpy&logoColor=white)](https://numpy.org)  [![Pandas-Badge](https://img.shields.io/badge/Pandas-150458?style=square&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![Scipy-Badge](https://img.shields.io/badge/Scipy-darkblue?style=square&logo=scipy&logoColor=white)](https://www.scipy.org)  [![Fitter-Badge](https://img.shields.io/badge/Fitter-000000?style=square&logo=python&&logoColor=yellow)](https://fitter.readthedocs.io/en/latest/)  [![Matplotlib-Badge](https://img.shields.io/badge/Matplotlib-40403f?style=square&logo=python&logoColor=blue)](https://matplotlib.org)  [![Seaborn-Badge](https://img.shields.io/badge/Seaborn-40403f?style=square&logo=python&logoColor=blue)](https://seaborn.pydata.org)
###### Interface: [![React-Badge](https://img.shields.io/badge/React-61DAFB?style=square&logo=react&logoColor=black)](https://create-react-app.dev)
###### Version Control: [![Git-Badge](https://img.shields.io/badge/Git-F05032.svg?style=square&logo=Git&logoColor=white)](https://git-scm.com) [![Git-Commads](https://img.shields.io/badge/Git%20Commands-gray?style=square&logo=git&logoColor=white)](https://github.com/EstebanMqz/Git-Commands)
</Details>
</div>
<Details> <Summary> <i> Files Visualization: </i> </Summary>

[![Repository](https://img.shields.io/badge/Repository-0089D6?style=square&logo=microsoft-azure&logoColor=white)](https://mango-dune-07a8b7110.1.azurestaticapps.net/?repo=EstebanMqz%2FSP500-Risk-Optimizations-Forecast) [![Jupyter](https://img.shields.io/badge/Render-nbviewer-000000?style=square&logo=jupyter&logoColor=orange)](https://nbviewer.jupyter.org/github/EstebanMqz/SP500-Risk-Optimizations-Forecast/blob/main/SP500-Risk-Optimized-Portfolios-ML.ipynb)

<img src="diagram.svg" width="280" height="280">

</Details> 

<Details> <Summary> <h3> <font color= '#3d74eb'> 1.1 Table of Contents:  </font> </h3> </Summary>

Sections and processes are illustrated:

<img src="images/ToC.jpg" width="494" height="440">

</Details>

<Details> <Summary> <h3> <font color= '#3d74eb'> 1.2 Description:  </font> </h3> </Summary> 

[web-app (parser)](https://github.com/EstebanMqz/Optimized_Indexes_Forecasts/blob/c414ca71b86219725a8d55bc03674573faa02087/images/ToC.jpg)

Data has changed since Covid in most industries and the markets project the public's sentiment, financially speaking.<br>
In this regard, from indexes $OHLCVs$, Volumes $V_{t}$ are standarized and compared dynamically in a web-app to show the big picture of $I$ to the user, as well as to note important insights.<br> 
Moreover, from $Adj_{Closes} \rightarrow P_t$ <i>, its Compounded Returns</i> are manipulated as: $ln({P_t})-ln({P_{t-1}})$ because of their <b>additive nature</b> (instead of multiplicative)<br>
among other characteristics <i>(see 3.2)</i>, which makes the following true:<br>
$$\prod_{t=1}^{n}(1+r_t) \implies \bigg[{\mathrm{e}^{\sum_{t=1}^{n} ln (1+r_t) }} \bigg]$$ 

From <i><b>Logarithmic Compounded Returns</b></i> the following are calculated and interpreted <i>(sections 3-7)</i><br>

+ Estimators: $\mu_{j_d}, \mu_{j_{Yr}} \mu_{P_d}, \mu_{P_{Yr}}$.
+ Disperssion measures: $Q_n$, $IQR$, $\sigma_{j_d}, \sigma_{j_{Yr}} \sigma_{P_d}, \sigma_{P_{Yr}}$, Correlation $\rho_{i,j}$ and Covariance $\sigma_{i,j}$ matrices.<br>
+ Optimizations: $R_{Sharpe}$, $R_{Sortino}$, $R_{Calmar}$, $R_{Burke}$, $R_{Kappa}$, $R_{\Omega}$, $R_{Traynor}$, $R_{Jensen}$.
+ Risk measures: $VaR_{\alpha}$, $ES_{\alpha}$, $MDD$.<br>

Out of the indexes supported, the $S\&P500$ is generally used as benchmark because:<br>

1. It's the most commonly used index to determine the <i><u>overall state of the economy.</u></i><br>
2. Because it has the <i><u>most liquid derivatives markets</u></i> (the same generally applies for $j$ components).<br>
 *Note: Market Risk exposure hedging won't be covered in this repository.*<br>
1. Its $j$ components provide a <i><u>broader scope</u></i> to different industries.<br>

Therefore, $\mathbb{R}^{500} = x_j\in [x_1,x_{500}] \hookrightarrow S\&P500$ is modelled as an example of its usage *(sections 4-7).*

Optimizations Accumulated Returns are obtained, simulated and forecasted from:

$$R^{n \times m} = \sum_{t=1}^{n} \sum_{j=1}^{m} w_{j}\ ln(1+r_t)$$
+ $m$ = No° of components.
+ $j$ = Component
+ $n$ = No° of periods. 
+ $t$ = Period.

</Details>


<Details> <Summary> <h4> <font color= '#3d74eb'> 8. References </font> </h4> </Summary> 

<font color= 'white'><h6>

#### Libraries

+ ##### </u> Pandas: </u> <br>

[`pd.isin`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isin.html) [`pd.df.sample`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html) [`pd.df.fillna`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html) [`pd.df.resample`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html) [`pandas.DataFrame.describe`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)

+ ##### </u> Numpy: </u> <br>

[`np.quantile`](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html) [`np.arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html) [`np.add`](https://numpy.org/doc/stable/reference/generated/numpy.add.html) [`np.subtract`](https://numpy.org/doc/stable/reference/generated/numpy.subtract.html) [`np.dot`](https://numpy.org/doc/stable/reference/generated/numpy.dot.html) [`np.divide`](https://numpy.org/doc/stable/reference/generated/numpy.divide.html) [`np.cov`](https://numpy.org/doc/stable/reference/generated/numpy.cov.html) [`np.power`](https://numpy.org/doc/stable/reference/generated/numpy.power.html) <br>

+ ##### </u> Stats: </u> <br>

[`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html) [`scipy.stats.rv_continuous`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html) [`scipy.stats.rv_discrete`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html) [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)

+ ##### </u> Sklearn: </u> <br>

[`sklearn.model_selection.GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) [`Hyper-parameters Exhaustive GridSearchCV`](https://scikit-learn.org/stable/modules/grid_search.html) <br>
[`sklearn.neighbors.KernelDensity`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html) [`sklearn.neighbors.KernelDensity.fit`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity.fit) <br>
[`sklearn.neighbors.KernelDensity.score_samples`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity.score_samples) [`sklearn.metrics`](https://scikit-learn.org/stable/modules/model_evaluation.html)

+ ##### Other: <br>

[`fitter`](https://fitter.readthedocs.io/en/latest/index.html)<br>
[`statsmodels`](https://www.statsmodels.org/stable/index.html)<br><br>

+ ##### Other References: <br>

###### *Indexes Supported*:<br>

+ [`S&P`](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) [`Dow Jones`](https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average) [`NASDAQ 100`](https://en.wikipedia.org/wiki/NASDAQ-100) [`Russell 1000`](https://en.wikipedia.org/wiki/Russell_1000_Index) [`FTSE 100`](https://en.wikipedia.org/wiki/FTSE_100_Index) [`IPC`](https://en.wikipedia.org/wiki/Indice_de_Precios_y_Cotizaciones) [`DAX`](https://en.wikipedia.org/wiki/DAX) [`IBEX 35`](https://en.wikipedia.org/wiki/IBEX_35) [`CAC 40`](https://en.wikipedia.org/wiki/CAC_40) [`EURO STOXX 50`](https://en.wikipedia.org/wiki/EURO_STOXX_50) [`FTSE MIB`](https://en.wikipedia.org/wiki/FTSE_MIB) [`Hang Seng Index`](https://en.wikipedia.org/wiki/Hang_Seng_Index)

###### *Official Market Risks Data:*

+ [`Daily Treasury Par Yield Curve Rates`](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value_month=202304)<br>
+ [`Bank of International Settlements (BIS)`](https://www.bis.org/statistics/index.htm)<br>

###### *Other:*

+ [`LaTeX`](https://en.wikipedia.org/wiki/List_of_mathematical_symbols_by_subject)</br>
+ [`Expected Shortfall (ES)`](https://en.wikipedia.org/wiki/Expected_shortfall) [`Value at Risk (VaR)`](https://en.wikipedia.org/wiki/Value_at_risk)
+ [`Convolution of Distributions`](https://en.wikipedia.org/wiki/Convolution_of_probability_distributions)<br>
+ [`i.i.d`](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)<br>

+ [`Expected Shortfall (ES)`](https://en.wikipedia.org/wiki/Expected_shortfall)<br>
