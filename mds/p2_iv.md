

```python
import pandas as pd
import math
from IPython import display
import statsmodels.api as sm

df = sm.add_constant(pd.read_stata('game2_data.dta'))
df = df.dropna()

display.display(df.head())
```

    /usr/local/lib/python3.6/dist-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const</th>
      <th>y</th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>x4</th>
      <th>x5</th>
      <th>x6</th>
      <th>x7</th>
      <th>x8</th>
      <th>x9</th>
      <th>x10</th>
      <th>x11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.438777</td>
      <td>-9.163060</td>
      <td>-1.588530</td>
      <td>-0.802902</td>
      <td>-3.990557</td>
      <td>-0.802902</td>
      <td>1.584558</td>
      <td>-3.111471</td>
      <td>-0.451257</td>
      <td>2.154801</td>
      <td>-5.849712</td>
      <td>-1.384668</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.629828</td>
      <td>-1.438437</td>
      <td>1.177610</td>
      <td>-0.196004</td>
      <td>0.446565</td>
      <td>-0.196004</td>
      <td>-1.218185</td>
      <td>0.690861</td>
      <td>-2.143279</td>
      <td>0.121307</td>
      <td>-0.341499</td>
      <td>-0.467438</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-3.428959</td>
      <td>11.462125</td>
      <td>2.887286</td>
      <td>1.358113</td>
      <td>3.876879</td>
      <td>1.358113</td>
      <td>-0.815247</td>
      <td>2.638416</td>
      <td>0.158565</td>
      <td>-1.563668</td>
      <td>4.190841</td>
      <td>-0.548659</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-1.320286</td>
      <td>5.854400</td>
      <td>-0.552525</td>
      <td>1.233984</td>
      <td>1.719736</td>
      <td>1.233984</td>
      <td>2.150350</td>
      <td>0.896650</td>
      <td>-0.827780</td>
      <td>4.008009</td>
      <td>1.085393</td>
      <td>1.486160</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.773177</td>
      <td>-7.031788</td>
      <td>-1.952549</td>
      <td>-0.887858</td>
      <td>-2.296057</td>
      <td>-0.887858</td>
      <td>-0.705950</td>
      <td>0.054110</td>
      <td>-0.355626</td>
      <td>-2.779588</td>
      <td>-1.299902</td>
      <td>-0.291565</td>
    </tr>
  </tbody>
</table>
</div>


## IV-2SLS 预测

- IV2SLS
- IVGMM
- IVGMMCUE
- IVLIML


```python
import linearmodels.iv as iv
import statsmodels.api as sm

dependent = df['y']
exdog = df[['const', 'x9', 'x10', 'x11']]
endog = df[['x1']]
instruments = df[['x2', 'x3', 'x4', 'x6', 'x7', 'x8']]

model = iv.IV2SLS(dependent, exdog, endog, instruments)
result = model.fit()
print(result)

print("~" * 100)

model = iv.IVGMM(dependent, exdog, endog, instruments)
result = model.fit()
print(result)

print("~" * 100)

model = iv.IVGMMCUE(dependent, exdog, endog, instruments)
result = model.fit()
print(result)

print("~" * 100)

model = iv.IVLIML(dependent, exdog, endog, instruments)
result = model.fit()
print(result)
```

                              IV-2SLS Estimation Summary                          
    ==============================================================================
    Dep. Variable:                      y   R-squared:                      0.9982
    Estimator:                    IV-2SLS   Adj. R-squared:                 0.9982
    No. Observations:               10000   F-statistic:                 5.474e+06
    Date:                Wed, Jul 31 2019   P-value (F-stat)                0.0000
    Time:                        01:42:49   Distribution:                  chi2(4)
    Cov. Estimator:                robust                                         
                                                                                  
                                 Parameter Estimates                              
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    const          0.0003     0.0011     0.3107     0.7560     -0.0018      0.0025
    x9             0.0864     0.0005     159.81     0.0000      0.0853      0.0875
    x10            0.1193     0.0010     124.86     0.0000      0.1175      0.1212
    x11            0.0988     0.0011     90.767     0.0000      0.0967      0.1010
    x1            -0.3329     0.0003    -1253.2     0.0000     -0.3334     -0.3324
    ==============================================================================
    
    Endogenous: x1
    Instruments: x2, x3, x4, x6, x7, x8
    Robust Covariance (Heteroskedastic)
    Debiased: False
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                              IV-GMM Estimation Summary                           
    ==============================================================================
    Dep. Variable:                      y   R-squared:                      0.9982
    Estimator:                     IV-GMM   Adj. R-squared:                 0.9982
    No. Observations:               10000   F-statistic:                 5.489e+06
    Date:                Wed, Jul 31 2019   P-value (F-stat)                0.0000
    Time:                        01:42:49   Distribution:                  chi2(4)
    Cov. Estimator:                robust                                         
                                                                                  
                                 Parameter Estimates                              
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    const         -0.0018     0.0011    -1.6647     0.0960     -0.0039      0.0003
    x9             0.0866     0.0005     160.18     0.0000      0.0855      0.0876
    x10            0.1209     0.0010     126.63     0.0000      0.1190      0.1228
    x11            0.0999     0.0011     91.689     0.0000      0.0978      0.1020
    x1            -0.3334     0.0003    -1257.3     0.0000     -0.3339     -0.3329
    ==============================================================================
    
    Endogenous: x1
    Instruments: x2, x3, x4, x6, x7, x8
    GMM Covariance
    Debiased: False
    Robust (Heteroskedastic)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                              IV-GMM Estimation Summary                           
    ==============================================================================
    Dep. Variable:                      y   R-squared:                      0.9977
    Estimator:                     IV-GMM   Adj. R-squared:                 0.9977
    No. Observations:               10000   F-statistic:                 4.434e+06
    Date:                Wed, Jul 31 2019   P-value (F-stat)                0.0000
    Time:                        01:42:50   Distribution:                  chi2(4)
    Cov. Estimator:                robust                                         
                                                                                  
                                 Parameter Estimates                              
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    const         -0.0271     0.0012    -23.275     0.0000     -0.0294     -0.0249
    x9             0.0858     0.0006     141.11     0.0000      0.0846      0.0870
    x10            0.1400     0.0011     129.66     0.0000      0.1378      0.1421
    x11            0.1338     0.0013     104.80     0.0000      0.1313      0.1363
    x1            -0.3398     0.0003    -1124.9     0.0000     -0.3404     -0.3392
    ==============================================================================
    
    Endogenous: x1
    Instruments: x2, x3, x4, x6, x7, x8
    GMM Covariance
    Debiased: False
    Robust (Heteroskedastic)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                              IV-LIML Estimation Summary                          
    ==============================================================================
    Dep. Variable:                      y   R-squared:                      0.9981
    Estimator:                    IV-LIML   Adj. R-squared:                 0.9981
    No. Observations:               10000   F-statistic:                 5.119e+06
    Date:                Wed, Jul 31 2019   P-value (F-stat)                0.0000
    Time:                        01:42:50   Distribution:                  chi2(4)
    Cov. Estimator:                robust                                         
                                                                                  
                                 Parameter Estimates                              
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    const          0.0001     0.0011     0.0924     0.9264     -0.0021      0.0023
    x9             0.0824     0.0006     148.64     0.0000      0.0814      0.0835
    x10            0.1090     0.0010     111.46     0.0000      0.1071      0.1109
    x11            0.0988     0.0011     89.019     0.0000      0.0966      0.1010
    x1            -0.3288     0.0003    -1194.5     0.0000     -0.3294     -0.3283
    ==============================================================================
    
    Endogenous: x1
    Instruments: x2, x3, x4, x6, x7, x8
    Robust Covariance (Heteroskedastic)
    Debiased: False
    Kappa: 2722423926149.400

