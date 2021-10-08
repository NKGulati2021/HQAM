import numpy as np
import pandas as pd
import yfinance as yf


ticker_list = ['AAPL', 
               'AMZN', 
               'GOOG',
               'FB',
               'BRK-B',
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
               'GS',
               '^GSPC']

out_df = pd.DataFrame()

for ticker in ticker_list:
    
    temp_df = yf.download(tickers = ticker, start = '2015-12-31', end = '2020-12-31', interval = '1d')
    temp_df = temp_df['Close']
    temp_df = temp_df.dropna()
    
    year_ret_arr = []
    year_arr = []
    ticker_arr = []
    
    for i in range(2016, 2021):
        
        year_arr.append(i)
        ticker_arr.append(ticker)
        
        end_date = str(i) + '-12-31'
        start_date = str(i - 1) + '-12-31'
        
        if (len(temp_df[temp_df.index == start_date]) == 0):
            start_date = str(i - 1) + '-12-30'
        if (len(temp_df[temp_df.index == end_date]) == 0):
            end_date = str(i) + '-12-30'
        
        if (len(temp_df[temp_df.index == start_date]) == 0):
            start_date = str(i - 1) + '-12-29'
        if (len(temp_df[temp_df.index == end_date]) == 0):
            end_date = str(i) + '-12-29'
            
        start_val = np.array(temp_df[temp_df.index == start_date])
        end_val = np.array(temp_df[temp_df.index == end_date])
        
        ret_val = ((end_val / start_val) - 1)[0]
        
        year_ret_arr.append(ret_val)
        
    out_df[ticker] = year_ret_arr
    
out_df.index = year_arr

#Calc market premium
df_cols = out_df.columns

for i in range(len(df_cols) - 1):

    col_name = df_cols[i]
    
    out_df[col_name] = out_df[col_name] - out_df['^GSPC']
    
out_df.to_csv(r"D:\MFE\HQAM Project\YoYReturns.csv", index = True)
    
        





