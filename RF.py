from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.tree import export_graphviz
import pydot
import datetime
from time import time
from mpl_toolkits.mplot3d import Axes3D
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingRegressor



df = pd.read_csv(r"D:\MFE\HQAM Project\RegrData.csv")
df = df.loc[df['Year'] != 2021]
df = df.loc[df['Year'] != 2016]
df = df.loc[df['Form'] == '10K']
df = df.drop('Form', axis = 1)
df['MarketPrem'] = df['MarketPrem'] * 100


df = df.drop_duplicates()

temp = df['Ticker']
df.iloc[:, 0] = df['Year']
df.iloc[:, 1] = temp

col_list = list(df)
col_list[0], col_list[1] = col_list[1], col_list[0]
df.columns = col_list

labels = np.array(df['MarketPrem'])

features = df[['Year', 'Positive Percentage','Subjectivity','ESG Words','ESG Percentage']]


exog = features
dependent = labels

feature_list = list(features.columns)

features = np.array(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

rf.fit(train_features, train_labels)
predictions = rf.predict(test_features)

rf_small = RandomForestRegressor(n_estimators= 100 , max_depth = 4)


rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)


years = df['Year']

dates = [str(int(year)) for year in years]
dates = [datetime.datetime.strptime(date, '%Y') for date in dates]

true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})

y = labels
X = df[['Year', 'Positive Percentage','Subjectivity','ESG Words','ESG Percentage']]
y -= y.mean()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=0)

fig = plt.figure()
pd.plotting.scatter_matrix(X,figsize =(40,40),alpha=0.9,diagonal="hist",marker="o");
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.rcParams['font.size'] = '24'

      
      
