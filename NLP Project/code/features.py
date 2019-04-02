# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 19:41:01 2019

@author: cassie
"""
import numpy as np  
import pickle
# Pos Tag
import nltk
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
        
def extract_feature(preprocess, preprocessed_train_tweets,preprocessed_test_tweets,feature_no=0,train_save_path='',test_save_path=''):
    if feature_no==0:
        # word n-gram
        ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3))
        ngram_vectorizer.fit(preprocessed_train_tweets)
        X = ngram_vectorizer.transform(preprocessed_train_tweets)
        X_test = ngram_vectorizer.transform(preprocessed_test_tweets)
        
    elif feature_no==1:    
        # word-count (unigram)
        wc_vectorizer = CountVectorizer(binary=False)
        wc_vectorizer.fit(preprocessed_train_tweets)
        X = wc_vectorizer.transform(preprocessed_train_tweets)
        X_test = wc_vectorizer.transform(preprocessed_test_tweets)
        
    elif feature_no==2:
        # character n-gram
        all_character_ngram=[]
        for n in [2,3,4,5]:
            for tweet in preprocessed_train_tweets:
                tweet=list(tweet.replace(" ","_"))
                for index,character in enumerate(tweet):
                    if (index+n)>len(tweet):
                        continue
                    all_character_ngram.append(''.join(tweet[index:index+n]))
        all_character_ngram=list(set(all_character_ngram))
        X=[]
        for tweet in preprocessed_train_tweets:
            X.append([])
            tweet=tweet.replace(" ","_")
            for ngram in all_character_ngram:
                if ngram in tweet:
                    X[len(X)-1].append(1)
                else:
                    X[len(X)-1].append(0)
        X=np.array(X)
        X_test=[]
        for tweet in preprocessed_test_tweets:
            X_test.append([])
            tweet=tweet.replace(" ","_")
            for ngram in all_character_ngram:
                if ngram in tweet:
                    X_test[len(X_test)-1].append(1)
                else:
                    X_test[len(X_test)-1].append(0)
        X_test=np.array(X_test)                
        
    elif feature_no==3:
        #  POS tagged bigram feature
        train_tagged_texts = [nltk.pos_tag(word_tokenize(t)) for t in preprocessed_train_tweets]
        test_tagged_texts = [nltk.pos_tag(word_tokenize(t)) for t in preprocessed_test_tweets]
        all_pos_tagged_bigram=[]
        for tagg in train_tagged_texts:
            for i,(word,tag) in enumerate(tagg):
                if ('JJ' not in tag) | (i+1>=len(tagg)):
                    continue
                if 'NN' in tagg[i+1][1]:
                    all_pos_tagged_bigram.append(word+" "+tagg[i+1][0])
        X=[]
        for tweet in preprocessed_train_tweets:
            X.append([])
            for bigram in all_pos_tagged_bigram:
                if bigram in tweet:
                    X[len(X)-1].append(1)
                else:
                    X[len(X)-1].append(0)
        X=np.array(X)
        X_test=[]
        for tweet in preprocessed_test_tweets:
            X_test.append([])
            for bigram in all_pos_tagged_bigram:
                if bigram in tweet:
                    X_test[len(X_test)-1].append(1)
                else:
                    X_test[len(X_test)-1].append(0)
        X_test=np.array(X_test)
        
    elif feature_no==4:
        # TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(preprocessed_train_tweets)
        tfidf_words_list=[i.lower() for i in tfidf_vectorizer.get_feature_names()]
        X = tfidf_vectorizer.transform(preprocessed_train_tweets)
        X_test = tfidf_vectorizer.transform(preprocessed_test_tweets)
        
    elif (feature_no==5)|((feature_no>=9)&(feature_no<=19)):
        # Word Embedding
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(preprocessed_train_tweets)
        tfidf_words_list=[i.lower() for i in tfidf_vectorizer.get_feature_names()]
        
        from sklearn import utils
        from gensim.models.word2vec import Word2Vec
        from gensim.models.doc2vec import TaggedDocument
        def labelize_tweets_ug(tweets,label):
            result = []
            prefix = label
            for i, t in enumerate(tweets):
                result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
            return result
        
        def twitter_embedding(feature_no,tweets_text,word_size,CBOW=True):
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
            model_ug_cbow.save('..//features//w2v_model_ug_cbow (feature_'+feature_no+'.word2vec')
            print("the number of words in this model: %s."%(len(model_ug_cbow.wv.vocab.keys()))) #单词数
            return model_ug_cbow
        
        word_size=100
        if (feature_no==5) | (feature_no==11) | (feature_no==14) | (feature_no==17):
            word_size=100
        elif (feature_no==9) | (feature_no==12) | (feature_no==15) | (feature_no==18):
            word_size=200
        elif (feature_no==10) | (feature_no==13) | (feature_no==16) | (feature_no==19):
            word_size=300
        
        CBOW=True
        if ((feature_no>=11) & (feature_no<=13)) | ((feature_no>=17) & (feature_no<=19)):
            CBOW=False
        
        train_corpus=preprocessed_train_tweets
        if ((feature_no>=14) & (feature_no<=19)):
            import pandas as pd
            additionalTweets = pd.read_csv('..//preprocessed_data//additional_Tweets.csv')
            train_corpus=additionalTweets[preprocess] # crawled additional tweets after proprocessing
                
        model_ug_cbow=twitter_embedding(feature_no,train_corpus,word_size,CBOW)
        total = len(model_ug_cbow.wv.vocab.keys()) # words number
        vec = np.ones((total, word_size), dtype = np.float32)
        
        word2id={}
        for i,word in enumerate(model_ug_cbow.wv.vocab.keys()):
            word2id[word] = len(word2id)
            for j in range(word_size):
                vec[i][j] = model_ug_cbow.wv[word][j]
        
        X=[]
        tfidf_documents=tfidf_vectorizer.transform(preprocessed_train_tweets).toarray()
        for no,doc in enumerate(preprocessed_train_tweets):
            weighted_glove=np.zeros(300)
            doc_length=0
            for i in doc.split():
                doc_length+=1
                if (i.lower() in word2id.keys()) & (i.lower() in tfidf_words_list):
                    weighted_glove+=tfidf_documents[no][tfidf_words_list.index(i.lower())]*vec[word2id.get(i.lower())]
            if doc_length!=0:
                X.append(weighted_glove/doc_length) # pre-process: stemming
            else:
                X.append(weighted_glove)
        X=np.array(X)
        
        X_test=[]
        tfidf_documents=tfidf_vectorizer.transform(preprocessed_test_tweets).toarray()
        for no,doc in enumerate(preprocessed_test_tweets):
            weighted_glove=np.zeros(300)
            doc_length=0
            for i in doc.split():
                doc_length+=1
                if (i.lower() in word2id.keys()) & (i.lower() in tfidf_words_list):
                    weighted_glove+=tfidf_documents[no][tfidf_words_list.index(i.lower())]*vec[word2id.get(i.lower())]
            if doc_length!=0:
                X_test.append(weighted_glove/doc_length) # pre-process: stemming
            else:
                X_test.append(weighted_glove)
        X_test=np.array(X_test)
    
    elif feature_no==6:
        
        def Glove_embedding(tweets,tfidf_words_list,tfidf_documents):
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
            glove_documents=[]
            for no,doc in enumerate(tweets):
                weighted_glove=np.zeros(300)
                doc_length=0
                for i in doc.split():
                    doc_length+=1
                    if (i.lower() in words) & (i.lower() in tfidf_words_list):
                        weighted_glove+=tfidf_documents[no][tfidf_words_list.index(i.lower())]*np.array(embeddings[words.index(i.lower())],dtype=np.float)
                if doc_length!=0:
                    glove_documents.append(weighted_glove/doc_length) # pre-process: stemming
                else:
                    glove_documents.append(weighted_glove)
            return glove_documents
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(preprocessed_train_tweets)
        tfidf_words_list=[i.lower() for i in tfidf_vectorizer.get_feature_names()]
        X = np.array(Glove_embedding(preprocessed_train_tweets, tfidf_words_list, tfidf_vectorizer.transform(preprocessed_train_tweets).toarray()))
        X_test = np.array(Glove_embedding(preprocessed_test_tweets, tfidf_words_list, tfidf_vectorizer.transform(preprocessed_test_tweets).toarray()))
    elif feature_no==7:    
        # Sentiment Feature
        train_tagged_texts = [nltk.pos_tag(word_tokenize(t)) for t in preprocessed_train_tweets]
        test_tagged_texts = [nltk.pos_tag(word_tokenize(t)) for t in preprocessed_test_tweets]
        from nltk.corpus import sentiwordnet as swn
        X=[]
        for index,tagged in enumerate(train_tagged_texts):
            pos=neg=obj=0
            for (word, tag) in tagged:
                ss_set = None
                if 'NN' in tag and len(list(swn.senti_synsets(word)))>0:
                    ss_set = list(swn.senti_synsets(word))[0]
                elif 'VB' in tag and len(list(swn.senti_synsets(word)))>0:
                    ss_set = list(swn.senti_synsets(word))[0]
                elif 'JJ' in tag and len(list(swn.senti_synsets(word)))>0:
                     ss_set = list(swn.senti_synsets(word))[0]
                elif 'RB' in tag and len(list(swn.senti_synsets(word)))>0:
                     ss_set = list(swn.senti_synsets(word))[0]
                if ss_set:
                    pos=pos+ss_set.pos_score()
                    neg=neg+ss_set.neg_score()
                    obj=obj+ss_set.obj_score()
                    
            X.append(pos-neg)
        X=np.array(X).reshape(-1, 1)
        
        X_test=[]
        for index,tagged in enumerate(test_tagged_texts):
            pos=neg=obj=0
            for (word, tag) in tagged:
                ss_set = None
                if 'NN' in tag and len(list(swn.senti_synsets(word)))>0:
                    ss_set = list(swn.senti_synsets(word))[0]
                elif 'VB' in tag and len(list(swn.senti_synsets(word)))>0:
                    ss_set = list(swn.senti_synsets(word))[0]
                elif 'JJ' in tag and len(list(swn.senti_synsets(word)))>0:
                    ss_set = list(swn.senti_synsets(word))[0]
                elif 'RB' in tag and len(list(swn.senti_synsets(word)))>0:
                    ss_set = list(swn.senti_synsets(word))[0]
                if ss_set:
                    pos=pos+ss_set.pos_score()
                    neg=neg+ss_set.neg_score()
                    obj=obj+ss_set.obj_score()
                    
            X_test.append(pos-neg)
        X_test=np.array(X_test).reshape(-1, 1)
    elif feature_no==8:
        # Skip-ngram
        all_skip_ngram={}
        for n in [1,2,3]:
            if n==1:
                all_skip_ngrams=[]
                for tweet in preprocessed_train_tweets:
                    tweet=tweet.split()
                    all_skip_ngrams.extend([' '.join(tweet[index:index+n]) for index in range(len(tweet)) if (index+n)<=len(tweet)])
                all_skip_ngram[n]={0:list(set(all_skip_ngrams))}
            else:
                temp={}
                for skip in [1,2]:
                    all_skip_ngrams=[]
                    for tweet in preprocessed_train_tweets:
                        tweet=tweet.split()
                        all_skip_ngrams.extend([' '.join(tweet[index:index+(n-1)*(skip+1)+1:2]) for index in range(len(tweet)) if (index+(n-1)*(skip+1)+1)<=len(tweet)])
                    temp[skip]=list(set(all_skip_ngrams))
                all_skip_ngram[n]=temp
            
            
        X=[]
        for tweet in preprocessed_train_tweets:
            X.append([])
            tweet=tweet.split()
            for n,ngrams in all_skip_ngram.items():
                if n==1:
                    for skip,ngram_array in ngrams.items():
                        for ngram in ngram_array:
                            if ngram in tweet:
                                X[len(X)-1].append(1)
                            else:
                                X[len(X)-1].append(0)
                else:
                    for skip,ngram_array in ngrams.items():
                        tweet_skip=[' '.join(tweet[index:index+(n-1)*(skip+1)+1:2]) for index in range(len(tweet)) if (index+(n-1)*(skip+1)+1)<=len(tweet)]
                        for ngram in ngram_array:
                            if ngram in tweet_skip:
                                X[len(X)-1].append(1)
                            else:
                                X[len(X)-1].append(0)
        X=np.array(X)
        X_test=[]
        for tweet in preprocessed_test_tweets:
            X_test.append([])
            tweet=tweet.split()
            for n, ngrams in all_skip_ngram.items():
                if n==1:
                    for skip,ngram_array in ngrams.items():
                        for ngram in ngram_array:
                            if ngram in tweet:
                                X_test[len(X_test)-1].append(1)
                            else:
                                X_test[len(X_test)-1].append(0)
                else:
                    for skip,ngram_array in ngrams.items():
                        tweet_skip=[' '.join(tweet[index:index+(n-1)*(skip+1)+1:2]) for index in range(len(tweet)) if (index+(n-1)*(skip+1)+1)<=len(tweet)]
                        for ngram in ngram_array:
                            if ngram in tweet_skip:
                                X_test[len(X_test)-1].append(1)
                            else:
                                X_test[len(X_test)-1].append(0)
        X_test=np.array(X_test)            
    # Syntactic n-grams
    pickle.dump(X,open(train_save_path,'wb'))
    pickle.dump(X_test,open(test_save_path,'wb'))
    return X, X_test