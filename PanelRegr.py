from linearmodels.panel import PanelOLS
from linearmodels.datasets import wage_panel
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

# data = wage_panel.load()
# data = data.set_index(['nr','year'])
# dependent = data.lwage
# exog = sm.add_constant(data[['expersq','married','union']])
# mod = PanelOLS(dependent, exog)
# res = mod.fit(cov_type='unadjusted')
# print(res)

#ticker_list = [ 'AAPL', 'ABBV', 'AMGN', 'AXP', 'CSCO']

df = pd.read_csv(r"D:\MFE\HQAM Project\RegrData.csv")
df = df.loc[df['Year'] != 2021]
#df = df.loc[df['Form'] == '10K']
#df = df.loc[df['Ticker'].isin(ticker_list)]
df['MarketPrem'] = df['MarketPrem'] * 100
df = df.set_index(['Ticker', 'Year'])
df = df.iloc[:, 1:]
df = df.drop_duplicates()
dependent = df['MarketPrem']
exog = sm.add_constant(df[['ESG Words']])
mod = PanelOLS(dependent, exog)
res = mod.fit()
print(res)

factor_corr = df[['Positive Percentage','Subjectivity','ESG Words','ESG Percentage']]
factor_corr = factor_corr.corr()
#factor_corr.to_csv(r"D:\MFE\HQAM Project\FactorCovTrim.csv", index = True)



