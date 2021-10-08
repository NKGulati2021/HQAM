import statsmodels.api as sm
import math 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection  import train_test_split
from tqdm import tqdm
# from sklearn.datasets import load_breast_cancer
# cancer = load_breast_cancer()
# print(cancer.keys())

df = pd.read_csv(r"D:\MFE\HQAM Project\RegrData.csv")
df = df.loc[df['Year'] != 2021]

#cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

X = sm.add_constant(df[['Positive Percentage','Subjectivity','ESG Words','ESG Percentage']]) 
Y = df['MarketPrem']

X_train,X_test,y_train,y_test=train_test_split(X,Y, test_size=0.3, random_state=31)

lasso = Lasso()
lasso.fit(X_train,y_train)
train_score=lasso.score(X_train,y_train)
test_score=lasso.score(X_test,y_test)
coeff_used = np.sum(lasso.coef_!=0)



lasso001 = Lasso(alpha=0.01, max_iter=10e5)
lasso001.fit(X_train,y_train)

lasso00001 = Lasso(alpha=0.0001, max_iter=10e5)
lasso00001.fit(X_train,y_train)
train_score00001=lasso00001.score(X_train,y_train)
test_score00001=lasso00001.score(X_test,y_test)
coeff_used00001 = np.sum(lasso00001.coef_!=0)
print("training score for alpha=0.0001:", train_score00001 )
print("test score for alpha =0.0001: ", test_score00001)
print("number of features used: for alpha =0.0001:", coeff_used00001)

lr = LinearRegression()
lr.fit(X_train,y_train)
lr_train_score=lr.score(X_train,y_train)
lr_test_score=lr.score(X_test,y_test)
print("LR training score:", lr_train_score )
print("LR test score: ", lr_test_score)
plt.subplot(1,2,1)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency

plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
#plt.legend(fontsize=13,loc=4)
plt.subplot(1,2,2)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency
plt.plot(lasso00001.coef_,alpha=0.8,linestyle='none',marker='v',markersize=6,color='black',label=r'Lasso; $\alpha = 0.00001$') # alpha here is for transparency
plt.plot(lr.coef_,alpha=0.7,linestyle='none',marker='o',markersize=5,color='green',label='Linear Regression',zorder=2)
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
#plt.legend(fontsize=13,loc=4)
plt.tight_layout()
plt.show()

# print('Coefficients: ' + str(lasso00001.coef_))

def run_lasso(penalty, X_train, y_train, X_test, y_test):
    
    lasso_reg = Lasso(alpha = penalty, max_iter = 10e5)
    
    lasso_reg.fit(X_train, y_train)
    
    train_score = lasso_reg.score(X_train, y_train)
    test_score = lasso_reg.score(X_test, y_test)
    
    coefs = np.sum(lasso_reg.coef_ != 0)
    
    return penalty, train_score, test_score, coefs

#alpha_arr = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
alpha_arr = np.arange(0.000001, 0.002, 0.000001)
train_score_arr = []
test_score_arr = []
coefs_arr = []

for alpha in tqdm(alpha_arr):
    
    penalty, train_score, test_score, coefs = run_lasso(alpha, X_train, y_train, X_test, y_test)

    train_score_arr.append(train_score)
    test_score_arr.append(test_score)
    coefs_arr.append(coefs)
    
fig, ax = plt.subplots(1, 1, figsize = (24, 12))
ax.plot(alpha_arr, train_score_arr, label = 'Train Scores')
ax.plot(alpha_arr, test_score_arr, label = 'Test Scores')
ax.set_xlabel('Alpha')
ax.set_ylabel('Score')
ax.legend(loc = 'best')
ax.set_title('Scores with Varying Penalty')

fig, ax1 = plt.subplots(1, 1, figsize = (24, 12))
ax1.plot(alpha_arr, coefs_arr)