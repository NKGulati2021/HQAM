import pandas as pd
from sec_edgar_downloader import Downloader
from tqdm import tqdm
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
from nltk.corpus import stopwords
import pysentiment2 as ps
from tqdm import tqdm
import os


def remove_html_tags(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    
    return text


def clean_text(text):
    text = text.lower()
    text = remove_html_tags(text)
    
    return text

def lemmatize_words(words):

    return [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]



def print_ten_k_data(ten_k_data, fields, field_length_limit=50):
    indentation = '  '

    print('[')
    for ten_k in ten_k_data:
        print_statement = '{}{{'.format(indentation)
        for field in fields:
            value = str(ten_k[field])

            # Show return lines in output
            if isinstance(value, str):
                value_str = '\'{}\''.format(value.replace('\n', '\\n'))
            else:
                value_str = str(value)

            # Cut off the string if it gets too long
            if len(value_str) > field_length_limit:
                value_str = value_str[:field_length_limit] + '...'

            print_statement += '\n{}{}: {}'.format(indentation * 2, field, value_str)

        print_statement += '},'
        print(print_statement)
    print(']')
    
def CountFrequency(my_list):
  
    # Creating an empty dictionary 
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
  
    # for key, value in freq.items():
    #     print ('Word: ' + str(key) + ', Freq: ' + str(value))
        
    return freq

lm = ps.LM()
nltk.download('stopwords')
nltk.download('wordnet') 
nltk.download('omw')

dl = Downloader(r"D:\MFE\HQAM Project\APITest")

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

for ticker in tqdm(ticker_list):
    dl.get("10-K", ticker, amount = 5)
    dl.get("DEF 14A", ticker, amount = 5)
   
esg_word_list_df = pd.read_csv(r"D:\MFE\HQAM Project\Word_List.csv") 
lemma_english_stopwords = lemmatize_words(stopwords.words('english'))
wrdnet = lemmatize_words(wordnet.words())

directory = "D:\MFE\HQAM Project\APITest\sec-edgar-filings"

folder_dict = {}
score_dict = {}
total_word_dict = {}
ticker_tuple_arr = []

for filename in os.listdir(directory):
    temp_arr = []    
    
    f = os.path.join(directory, filename)
    ticker_10k = os.path.join(f, '10-K')
    ticker_14a = os.path.join(f, 'DEF 14A')
    
    for folder in os.listdir(ticker_10k):
        folder_10k = os.path.join(ticker_10k, folder)
        temp_year = folder[-9:-7]
        #temp_arr.append(os.path.join(folder_10k, 'full-submission.txt'))
        ticker_tuple_arr.append( (filename, temp_year, ' 10K', os.path.join(folder_10k, 'full-submission.txt')))
    
    for folder in os.listdir(ticker_14a):
        folder_14a = os.path.join(ticker_14a, folder)
        temp_year = folder[-9:-7]
        #temp_arr.append(os.path.join(folder_10k, 'full-submission.txt'))
        ticker_tuple_arr.append( (filename, temp_year, ' DEF 14A', os.path.join(folder_14a, 'full-submission.txt')))

for i in tqdm(range(len(ticker_tuple_arr))):
#for i in tqdm(range(71)):
    print('\nParsing: ' + str(ticker_tuple_arr[i][0] + ', Year: ' + str(ticker_tuple_arr[i][1]) + ', Form: ' + str(ticker_tuple_arr[i][2])))
    
    with open(ticker_tuple_arr[i][3]) as fp:
        soup = BeautifulSoup(fp, "html.parser")
    
    soup.head.title

    x = soup.get_text()
    x = clean_text(x)
    x = remove_html_tags(x)
    
    word_pattern = re.compile('\w+')
    x = lemmatize_words(word_pattern.findall(x))  
    x = [i for i in x if not i.isdigit()]        
    x = [word for word in x if word not in lemma_english_stopwords]
    
    num_words = len(x)    

    score = lm.get_score(x)       
    
    z_freq = CountFrequency(x)
    
    z_test = {k: v for k, v in sorted(z_freq.items(), key=lambda item: item[1], reverse = True)}
    
    esg_word_list = list(esg_word_list_df['Word'])

    esg_z = [k for k in esg_word_list if k in z_test.keys()]
    temp_word_freq_arr = []
    
    for word in esg_z:
        temp_word_freq_arr.append( (str(word), str(z_freq[word])) )
    
    folder_dict[str(ticker_tuple_arr[i][0]) + ' ' + str(ticker_tuple_arr[i][1]) + str(ticker_tuple_arr[i][2])] = temp_word_freq_arr
    score_dict[str(ticker_tuple_arr[i][0]) + ' ' + str(ticker_tuple_arr[i][1]) + str(ticker_tuple_arr[i][2])] = score
    total_word_dict[str(ticker_tuple_arr[i][0]) + ' ' + str(ticker_tuple_arr[i][1]) + str(ticker_tuple_arr[i][2])] = num_words
    
out_df = pd.DataFrame(columns=['Ticker','Year','Form','Frequency','Word','Topic','Category', 'Subcategory', 'Positive Frequency', 'Negative Frequency',
                               'Polarity Frequency', 'Subjectivity Frequency'])

ticker_arr = []
year_arr = []
form_arr = []
freq_arr = []
word_arr = []
total_arr = []
topic_arr = []
category_arr = []
subcat_arr = []
pos_arr = []
neg_arr = []
polar_arr = []
subj_arr = []

for key in folder_dict.keys():
    
    split_arr = key.split()

    for element in folder_dict[key]:    
        
        ticker_arr.append(split_arr[0])
        year_arr.append('20' + str(split_arr[1]))
        form_arr.append(split_arr[2])    
        freq_arr.append(element[1])
        word_arr.append(element[0])
        total_arr.append(total_word_dict[key])
        
        temp_df = esg_word_list_df.loc[esg_word_list_df['Word'] ==  element[0]]
        topic_arr.append(temp_df.iloc[0]['Topic'])
        category_arr.append(temp_df.iloc[0]['Category'])
        subcat_arr.append(temp_df.iloc[0]['Subcategory'])
        
        pos_arr.append(score_dict[key]['Positive'])
        neg_arr.append(score_dict[key]['Negative'])
        polar_arr.append(score_dict[key]['Polarity'])
        subj_arr.append(score_dict[key]['Subjectivity'])
        
    
out_df['Ticker'] = ticker_arr
out_df['Year'] = year_arr
out_df['Form'] = form_arr
out_df['Frequency'] = freq_arr
out_df['Word'] = word_arr
out_df['TotalWords'] = total_arr
out_df['Topic'] = topic_arr
out_df['Category'] = category_arr
out_df['Subcategory'] = subcat_arr
out_df['Positive Frequency'] = pos_arr
out_df['Negative Frequency'] = neg_arr
out_df['Polarity Frequency'] = polar_arr
out_df['Subjectivity Frequency'] = subj_arr


out_df.to_csv(r"D:\MFE\HQAM Project\TextParseData.csv", index = False)


    



        





