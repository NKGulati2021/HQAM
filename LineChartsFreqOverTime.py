import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\MFE\HQAM Project\DataFiles\ESGbyTickerOverTime.csv")

#Try without BAC
df = df[df['Ticker'] != 'BAC']

env_soc_df = df[df['Topic'] != 'Governance']
gov_df = df[df['Topic'] == 'Governance']

#Normalize Freq
#env_soc_df['Frequency'] = (env_soc_df['Frequency'] - np.mean(env_soc_df['Frequency'])) / np.std(env_soc_df['Frequency'])
gov_df['Frequency'] = (gov_df['Frequency'] - np.mean(gov_df['Frequency'])) / np.std(gov_df['Frequency'])

#Create Lines
ticker_list = df['Ticker'].unique()

#Social/Environmental
fig, ax = plt.subplots(1, 1, figsize = (24, 12))

for ticker in ticker_list:
    
    temp_env_df = env_soc_df[env_soc_df['Ticker'] == ticker]
    temp_env_df['Year'] = pd.to_datetime(temp_env_df['Year'])
    temp_env_df = temp_env_df.set_index('Year')
    
    ax.plot(temp_env_df.index, temp_env_df['Frequency'], label = ticker)
ax.set_title('Social and Environmental')
ax.legend(loc = 'upper left')

#Governance
fig, ax1 = plt.subplots(1, 1, figsize = (24, 12))

for ticker in ticker_list:
    
    temp_gov_df = gov_df[gov_df['Ticker'] == ticker]
    temp_gov_df['Year'] = pd.to_datetime(temp_gov_df['Year'])
    temp_gov_df = temp_gov_df.set_index('Year')
    
    ax1.plot(temp_gov_df.index, temp_gov_df['Frequency'], label = ticker)
ax1.set_title('Governance')
ax1.legend(loc = 'upper left')

