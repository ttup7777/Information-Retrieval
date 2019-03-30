# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 21:33:12 2019

@author: cassie
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from nltk.util import ngrams

#s = s.lower()
#s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
#tokens = [token for token in s.split(" ") if token != ""]
#output = list(ngrams(tokens, 5))

# Read Data
train_A=pd.read_table("..//data//Training data for Task A.txt",encoding='gbk')
test_A=pd.read_table("..//data//semeval2016-task6-testdata-gold//SemEval2016-Task6-subtaskA-testdata-gold.txt",encoding='gbk')

# Preprocessing
from nltk.corpus import stopwords
import string,re

english_stop_words = stopwords.words('english')
def remove_stop_words(corpus):
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
        '''
        text = re.sub('@([A-Za-z0-9_]+)', '',text)  # to replace @用户 with ''
        text=re.sub('\d+\.\d+','deci',text)#小数替换
        text=re.sub('\d+','int',text)#整数替换
        '''
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if (word=="a") | (word=="i") | \
                      (word not in english_stop_words \
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

train_A['Tweet'] = remove_stop_words(train_A['Tweet'])
test_A['Tweet'] = remove_stop_words(test_A['Tweet'])
train_A['stemmed_Tweet'] = get_stemmed_text(train_A['Tweet'])
test_A['stemmed_Tweet'] = remove_stop_words(test_A['Tweet'])
train_A['lemmatized_Tweet'] = get_lemmatized_text(train_A['Tweet'])
test_A['lemmatized_Tweet'] = get_lemmatized_text(test_A['Tweet'])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# word n-gram
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(train_A['lemmatized_Tweet'])
X = ngram_vectorizer.transform(train_A['lemmatized_Tweet'])
X_test = ngram_vectorizer.transform(test_A['lemmatized_Tweet'])

# word-count (unigram)
wc_vectorizer = CountVectorizer(binary=False)
wc_vectorizer.fit(train_A['lemmatized_Tweet'])
X = wc_vectorizer.transform(train_A['lemmatized_Tweet'])
X_test = wc_vectorizer.transform(test_A['lemmatized_Tweet'])

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(train_A['lemmatized_Tweet'])
X = tfidf_vectorizer.transform(train_A['lemmatized_Tweet'])
X_test = tfidf_vectorizer.transform(test_A['lemmatized_Tweet'])

# SentiFeature
from nltk.corpus import sentiwordnet as swn

# Word Embedding
from sklearn import utils
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
def labelize_tweets_ug(tweets,label):
    result = []
    prefix = label
    for i, t in enumerate(tweets):
        result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
    return result

def twitter_embedding(tweets_text,word_size,CBOW=True):
    print("word2vec...")
    all_x =  tweets_text 
    all_x_w2v = labelize_tweets_ug(all_x, 'all')
    print("training word embedding...")
    
    x=[i.words for i in all_x_w2v]
    #cores = multiprocessing.cpu_count()
    if CBOW==True:
        #CBOW
        model_ug_cbow = Word2Vec(sg=0, size=word_size, negative=5, window=2, alpha=0.065, min_count=3, min_alpha=0.065)
    else:
        #Skip-gram
        model_ug_cbow = Word2Vec(sg=1, size=word_size, negative=5, window=2, alpha=0.065, min_count=3, min_alpha=0.065)
    model_ug_cbow.build_vocab(x)
    
    for epoch in range(15):
        print("epoch %s..."%(epoch))
        model_ug_cbow.train(utils.shuffle(x), total_examples=len(all_x_w2v), epochs=1)
        model_ug_cbow.alpha -= 0.002
        model_ug_cbow.min_alpha = model_ug_cbow.alpha
    
    print("saving word embedding...")
    model_ug_cbow.save('w2v_model_ug_cbow.word2vec')
    print("the number of words in this model: %s."%(len(model_ug_cbow.wv.vocab.keys()))) #单词数

def Glove_embedding():
    print('read GloV...')
    gloV={}
    with open("..//data//glove.6B//glove.6B.300d.txt",'rb') as f:  
        line = f.readline()
        no = 1
        while line:
            print(no)
            elements=str(line)[2:-3].split(" ")
            gloV[elements[0]]=elements[1:]
            no+=1
            line = f.readline()
    #from nltk.stem.porter import PorterStemmer
    #porter_stemmer = PorterStemmer()
    #words=[porter_stemmer.stem(i) for i in list(gloV.keys())]
    words=[i.lower() for i in list(gloV.keys())]
    embeddings=list(gloV.values())
    
# Pos Tag
    
# train and validation split
#from sklearn.model_selection import train_test_split
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20)  

# Lable Mapping
train_A['Stance_three_type']=train_A['Stance']
train_A.loc[train_A['Stance_three_type']=='NONE','Stance_three_type']=0
train_A.loc[train_A['Stance_three_type']=='AGAINST','Stance_three_type']=1
train_A.loc[train_A['Stance_three_type']=='FAVOR','Stance_three_type']=2

train_A['Stance_two_type']=train_A['Stance']
train_A.loc[train_A['Stance_two_type']!='NONE','Stance_two_type']=1
train_A.loc[train_A['Stance_two_type']=='NONE','Stance_two_type']=0

test_A['Stance_three_type']=test_A['Stance']
test_A.loc[test_A['Stance_three_type']=='NONE','Stance_three_type']=0
test_A.loc[test_A['Stance_three_type']=='AGAINST','Stance_three_type']=1
test_A.loc[test_A['Stance_three_type']=='FAVOR','Stance_three_type']=2

test_A['Stance_two_type']=test_A['Stance']
test_A.loc[test_A['Stance_two_type']!='NONE','Stance_two_type']=1
test_A.loc[test_A['Stance_two_type']=='NONE','Stance_two_type']=0

# Model Training
X_train=X
y_train=train_A['Stance_two_type']
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')#kernel='poly', degree=8 #Gaussian：kernel='rbf' #kernel='sigmoid'
svclassifier.fit(X_train, y_train)

#predict
y_pred = svclassifier.predict(X_test)  
y_test=test_A['Stance_two_type']

#Evaluation
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  

# Cross Validate
from sklearn.model_selection import cross_validate
scores = cross_validate(svclassifier, X_train, y_train, cv=5)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))          
