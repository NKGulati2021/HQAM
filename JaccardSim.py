import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"D:\MFE\HQAM Project\TopTen.csv")

def compute_jaccard(year, df):
    
    new_df = df.loc[df['Year'] == year]
    
    ticker_list = new_df['Ticker'].unique()
    ticker_list.sort()
    
    jac_mat = np.zeros([len(ticker_list), len(ticker_list)])
    
    for i in range(len(ticker_list)):
        
        for j in range(len(ticker_list)):
            
            A = new_df.loc[new_df['Ticker'] == ticker_list[i]]
            B = new_df.loc[new_df['Ticker'] == ticker_list[j]]
            A_set = set(A['Word'])
            B_set = set(B['Word'])
            
            jac_mat[i][j] = len(A_set.intersection(B_set)) / len(A_set.union(B_set))
            
    out_dict = {}
    
    for i in range(len(ticker_list)):
        
        out_dict[ticker_list[i]] = jac_mat[i][:]
        
    out_df = pd.DataFrame(out_dict)
    out_df = out_df.set_index(ticker_list)
            
    return out_df

def compute_jaccard_all(df):
    
    unique_year = df['Year'].unique()
    
    return_arr = []
    
    for year in unique_year:
    
        new_df = df.loc[df['Year'] == year]
        
        ticker_list = new_df['Ticker'].unique()
        ticker_list.sort()
        
        jac_mat = np.zeros([len(ticker_list), len(ticker_list)])
        
        for i in range(len(ticker_list)):
            
            for j in range(len(ticker_list)):
                
                A = new_df.loc[new_df['Ticker'] == ticker_list[i]]
                B = new_df.loc[new_df['Ticker'] == ticker_list[j]]
                A_set = set(A['Word'])
                B_set = set(B['Word'])
                
                jac_mat[i][j] = len(A_set.intersection(B_set)) / len(A_set.union(B_set))
                
        out_dict = {}
        year_arr = []
        for i in range(len(ticker_list)):
            
            out_dict[ticker_list[i]] = jac_mat[i][:]
            year_arr.append(year)
        out_dict['Year'] = year_arr
        out_df = pd.DataFrame(out_dict)
        out_df = out_df.set_index(ticker_list)
        
        return_arr.append(out_df)
            
    return return_arr

def find_most_similar(jac_df):
    
    out_dict = {}
    #year = jac_df[jac_df.columns[-1]].iloc[1]
    for col in jac_df.columns[:-1]:
        
        out_dict[col] = jac_df[jac_df[col] > 0.8].index
        
    return out_dict

out_arr = compute_jaccard_all(df)



# out_arr[0].to_csv(r"D:\MFE\HQAM Project\1.csv", index = True)
# out_arr[1].to_csv(r"D:\MFE\HQAM Project\2.csv", index = True)
# out_arr[2].to_csv(r"D:\MFE\HQAM Project\3.csv", index = True)
# out_arr[3].to_csv(r"D:\MFE\HQAM Project\4.csv", index = True)
# out_arr[4].to_csv(r"D:\MFE\HQAM Project\5.csv", index = True)
# out_arr[5].to_csv(r"D:\MFE\HQAM Project\6.csv", index = True)

for year in df['Year'].unique():
    fig, ax = plt.subplots(figsize = (24, 12))
    ax.set_title(str(year), fontsize = 24)
    jac = compute_jaccard(year, df)
    ax = sns.heatmap(jac, annot=True, annot_kws={"size": 10})
    ax.set_xticklabels(labels = jac.columns, fontsize = 12)
    ax.set_yticklabels(labels = jac.columns, fontsize = 10)
    
