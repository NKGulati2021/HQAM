import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

df = pd.read_csv(r"D:\MFE\HQAM Project\Test_Run2.csv")
ret_df = pd.read_csv(r"D:\MFE\HQAM Project\YoYRets.csv")
ret_df['Date'] = pd.to_datetime(ret_df['Date'])
ret_df = ret_df.set_index('Date')

ticker_list = ['AAPL', 
               'AMZN', 
               'GOOG',
               'FB',
               'BRK',
               'BAC',
               'CMCSA',
               'XOM',
               'KO',
               'T',
               'CSCO',
               'ABT',
               'CVX',
               'ABBV',
               'ACN',
               'AVGO',
               'LLY',
               'COST',
               'C',
               'BMY',
               'BA',
               'AMGN',
               'CAT',
               'AXP',
               'GS']

def filter_df(df, year, ticker):
    
    copy_df = copy.deepcopy(df) 
    
    copy_df = copy_df.loc[copy_df['Ticker'] == ticker]
    copy_df = copy_df.loc[copy_df['Year'] == year]
    copy_df = copy_df.loc[copy_df['Form'] == '10K']
    
    return copy_df

def generate_log_rets(df):
    
    df_shifted = df.shift(1)
    
    ret_df = np.log(df / df_shifted)
    
    return ret_df[1:]

def get_scores(df, ticker, col_name):
    
    out_dict = {}
    
    year_list = df['Year'].unique()
    
   # val_arr = []
    
    for year in year_list:
        
        temp_filtered = filter_df(df, year, ticker)
        
        #val_arr.append(filter_df[col_name][0])
        val = list(temp_filtered[col_name].unique())
        if (len(val) > 0):
            out_dict[str(year)] = val[0]
        
    return out_dict
    

log_returns = generate_log_rets(ret_df)
normalized_returns = (log_returns - np.mean(log_returns)) / np.std(log_returns)

aapl_esg = sorted(get_scores(df, 'AAPL', 'ESG Percentage').items())
aapl_esg_x = [i[0] for i in aapl_esg]
aapl_esg_y = [i[1] for i in aapl_esg]
aapl_ret = sorted(get_scores(df, 'AAPL', 'YearReturn').items())
aapl_ret_x = [i[0] for i in aapl_ret]
aapl_ret_y = [float(i[1]) for i in aapl_ret]

fig, ax = plt.subplots(1, 1, figsize = (24, 12))
ax.plot(aapl_esg_x, aapl_esg_y)
ax.plot(aapl_ret_x, aapl_ret_y)

fig, ax = plt.subplots(25, 1, figsize = (48, 24))

for i in range(len(ticker_list)):
    
    fig, ax = plt.subplots(1, 1, figsize = (48, 24))
    
    temp_esg = sorted(get_scores(df, ticker_list[i], 'ESG Percentage').items())
    temp_esg_x = [i[0] for i in temp_esg]
    temp_esg_y = [i[1] for i in temp_esg]
    temp_ret = sorted(get_scores(df, ticker_list[i], 'YearReturn').items())
    temp_ret_x = [i[0] for i in temp_ret]
    temp_ret_y = [float(i[1]) for i in temp_ret]
    
    ax.plot(temp_esg_x, temp_esg_y, label = str(ticker_list[i]) + ' ESG Word Pct.')
    ax.plot(temp_ret_x, temp_ret_y, label = str(ticker_list[i]) + ' YoY Returns')
    ax.legend(loc = 'best')
    ax.set_title(str(ticker_list[i]))
    

    


    