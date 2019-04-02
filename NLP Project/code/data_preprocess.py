# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:04:02 2019

@author: cassie
"""



import pandas as pd  
import numpy as np

# Preprocessing
from nltk.corpus import stopwords
import string,re

english_stop_words = stopwords.words('english')
def preclean(corpus):
    removed_stop_words = []
    new_punctuation=list(string.punctuation)+['“','”','‘','’','…','—']
    new_punctuation.remove('@')
    new_punctuation.remove('#')
    remove_spl_char_regex = re.compile('[%s]' % re.escape("".join(new_punctuation)))
    emoji_pattern = re.compile(u'['
                          u'\U0001F300-\U0001F64F'
                          u'\U0001F680-\U0001F6FF'
                          u'\u2600-\u2B55'
                          u'\u23cf'
                          u'\u23e9'
                          u'\u231a'
                          u'\u3030'
                          u'\ufe0f'
                          u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u'\U00010000-\U0010ffff'
                           u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
                           u'\U00002702-\U000027B0]+',     
                          re.UNICODE)
    
    for review in corpus:
        review = review.lower()#统一小写
        review=emoji_pattern.sub(" ", review)# Remove emoji
        review = remove_spl_char_regex.sub(" ", review)
        review = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',review) 
        review = re.sub('@([A-Za-z0-9_]+)', ' ',review)  # to replace @user with ''
        #text=re.sub('\d+\.\d+','deci',text)# replace decimal
        #text=re.sub('\d+','int',text)# replace int
        
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if (word=="a") | (word=="i") | \
                      (word not in english_stop_words \
                       and '@' not in word
                       and word not in "".join(new_punctuation) \
                       and len(word) > 1 \
                       and word !=" " \
                       and word != '``' \
                       and word.isdigit()!=True\
                       and word != 'rt')])
        )
    return removed_stop_words

# Stemming
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def get_stemmed_text(corpus):
    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

# Lemmatizing
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def get_lemmatized_text(corpus):
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

# Read Data
train_A=pd.read_table("..//data//Training data for Task A.txt",encoding='gbk')
print(len(train_A))
train_A=train_A.append(pd.read_table("..//data//semeval2016-task6-trialdata.txt",encoding='gbk'))
print(len(train_A))
train_A.reset_index(inplace=True)
test_A=pd.read_table("..//data//semeval2016-task6-testdata-gold//SemEval2016-Task6-subtaskA-testdata-gold.txt",encoding='gbk')
slang=pd.read_table("..//data//slang.txt",names=['slang','phrase'])
slang=dict([(s,p) for s,p in zip(slang['slang'],slang['phrase'])])

# Preprocessing
train_A['noslang_Tweet']=["" for i in range(len(train_A))]
test_A['noslang_Tweet']=["" for i in range(len(test_A))]
for i in train_A.index:
    words=train_A.loc[i,'Tweet'].split()
    for j,w in enumerate(words):
        if w in slang.keys():
            words[j]=slang.get(w)
    train_A.loc[i,'noslang_Tweet']=' '.join(words)
for i in test_A.index:
    words=test_A.loc[i,'Tweet'].split()
    for j,w in enumerate(words):
        if w in slang.keys():
            words[j]=slang.get(w)
    test_A.loc[i,'noslang_Tweet']=' '.join(words)
    
train_A['preclean_Tweet'] = preclean(train_A['Tweet'])
test_A['preclean_Tweet'] = preclean(test_A['Tweet'])

train_A['stemmed_Tweet'] = get_stemmed_text(train_A['preclean_Tweet'])
test_A['stemmed_Tweet'] = get_stemmed_text(test_A['preclean_Tweet'])

train_A['lemmatized_Tweet'] = get_lemmatized_text(train_A['preclean_Tweet'])
test_A['lemmatized_Tweet'] = get_lemmatized_text(test_A['preclean_Tweet'])

train_A['preclean_noslang_Tweet'] = preclean(train_A['noslang_Tweet'])
test_A['preclean_noslang_Tweet'] = preclean(test_A['noslang_Tweet'])

train_A['noslang_stemmed_Tweet'] = get_stemmed_text(train_A['preclean_noslang_Tweet'])
test_A['noslang_stemmed_Tweet'] = get_stemmed_text(test_A['preclean_noslang_Tweet'])

train_A['noslang_lemmatized_Tweet'] = get_lemmatized_text(train_A['preclean_noslang_Tweet'])
test_A['noslang_lemmatized_Tweet'] = get_lemmatized_text(test_A['preclean_noslang_Tweet'])

train_A.to_csv("..//preprocessed_data//train_A.csv")
test_A.to_csv("..//preprocessed_data//test_A.csv")

import json

tweets = []
count=0
count1=0
for line in open("../data/additionalTweetsStanceDetectionBig.json ",'r', encoding='utf-8'):
    print(count1)
    count1+=1
    try:
        tweets.append(json.loads(line).get('text'))
    except:
        print("exception")
        count+=1
        
x=pd.DataFrame()
x['preclean_Tweet']=preclean(tweets)
x['lemmatized_Tweet']=get_stemmed_text(tweets)
x['stemmed_Tweet']=get_lemmatized_text(tweets)

x.to_csv("..//preprocessed_data//additional_Tweets.csv")