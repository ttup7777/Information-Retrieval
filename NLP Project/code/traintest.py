# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 21:33:12 2019

@author: cassie
"""

import pandas as pd  
import numpy as np  
from features import extract_feature
import pickle,os
from sklearn.svm import SVC  
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
        
# Read Data
train_A=pd.read_csv("..//preprocessed_data//train_A.csv")
test_A=pd.read_csv("..//preprocessed_data//test_A.csv")
# Lable Mapping
print("labeling")
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

preprocess_design=['preclean_Tweet','lemmatized_Tweet','stemmed_Tweet',
                   'preclean_noslang_Tweet', 'noslang_lemmatized_Tweet',
                   'noslang_stemmed_Tweet']
#'preclean_Tweet','lemmatized_Tweet','stemmed_Tweet',

parameters = [
        {
                'C': [0.1, 1, 3, 5, 7, 9, 11], #0.01,
                'gamma': ['scale', 0.0001, 0.001, 0.1, 1/2, 1/3, 1, 10],
                'kernel': ['rbf']
                },
        {
                'C': [0.1, 1, 3, 5, 7, 9, 11], #0.01,
                'kernel': ['linear']
                },
        {
                'degree': [1, 3, 5], # 7, 9, 11
                'kernel': ['poly']
                }
        ]

model_path="..//model" #G://NLP project//model
result_path='..//result//'
for preprocess in preprocess_design:
    print(preprocess)
    preprocessed_train_tweets=train_A[preprocess]
    preprocessed_test_tweets=test_A[preprocess]
    for feature_no in range(22):#0,1,3,4 #2,8 #6, 5,9,10,11,12,13,14,15,16,17,18,19 #7,20,21
        '''
        if ((feature_no<14) & (preprocess=='preclean_Tweet')):
            continue
        '''
        train_save_path='..//features//train_feature_'+str(feature_no)+"_"+preprocess+".pkl"
        test_save_path='..//features//test_feature_'+str(feature_no)+"_"+preprocess+".pkl"
        if (os.path.isfile(train_save_path) & os.path.isfile(test_save_path)):
            print('read feature')
            X=pickle.load(open(train_save_path,'rb'))
            X_test=pickle.load(open(test_save_path,'rb'))
        else:
            print('extract feature')
            X, X_test=extract_feature(preprocess, preprocessed_train_tweets,preprocessed_test_tweets,feature_no,train_save_path,test_save_path)
        print(X.shape)
        print(X_test.shape)
        '''
        X1, X_test1=extract_feature(preprocessed_train_tweets,preprocessed_test_tweets,0,'','')
        X2, X_test2=extract_feature(preprocessed_train_tweets,preprocessed_test_tweets,2,'','')
        X=np.concatenate((X1.toarray(), X2), axis=1)
        X_test=np.concatenate((X_test1.toarray(),X_test2),axis=1)
        '''
        # Model Training
        # Single Stage
        # Combine
        if (not os.path.isfile(result_path+'svm_OneStage_combine_'+str(feature_no)+"_"+preprocess+".txt")) & (not os.path.isfile('..//result//svm_linear_OneStage_combine_'+str(feature_no)+"_"+preprocess+".txt")): 
            f = open(result_path+'svm_OneStage_combine_'+str(feature_no)+"_"+preprocess+".txt", 'a+')
            print("One stage and Combine:")
            print("One stage and Combine:", file=f)
            X_train=X
            y_train=train_A['Stance_three_type']
            # GridSearchCV
            if X.shape[1]<=60000:
                print('GridSearchCV')
                svclassifier = GridSearchCV(SVC(), parameters, cv=5, n_jobs=2)
                svclassifier.fit(X_train, y_train)
                print("SVM best score:",file=f)
                print(svclassifier.best_score_,file=f)
                print("SVM best parameters:",file=f)
                print(svclassifier.best_params_,file=f)
                #params=svclassifier.best_params_
            else:
                svclassifier=SVC(kernel='linear')
                svclassifier.fit(X_train, y_train)
            # Predict
            y_pred = svclassifier.predict(X_test)  
            y_test=test_A['Stance_three_type']
            pickle.dump(svclassifier,open(model_path+'//svm_OneStage_combine_'+str(feature_no)+"_"+preprocess+".pkl",'wb'))
            
            print(confusion_matrix(y_test,y_pred), file=f)  
            print(classification_report(y_test,y_pred), file=f)
            
        # Seperate
        if (not os.path.isfile(result_path+'svm_OneStage_seperate_'+str(feature_no)+"_"+preprocess+".txt")) & (not os.path.isfile('..//result//svm_linear_OneStage_seperate_'+str(feature_no)+"_"+preprocess+".txt")):
            f = open(result_path+'svm_OneStage_seperate_'+str(feature_no)+"_"+preprocess+".txt", 'a+')
            print("One stage and Seperate:")
            print("One stage and Seperate:", file=f)
            X_train=X
            y_train=train_A['Stance_three_type']
            seperate=train_A.groupby("Target")
            train_key_index=seperate.indices
            test_seperate=test_A.groupby("Target")
            test_key_index=test_seperate.indices
            
            SVMs={}
            seperate_y_pred=np.zeros(len(test_A))
            for (key,index) in train_key_index.items():
                print(key, file=f)
                X_train_=X_train[index]
                y_train_=y_train[index]
                if X.shape[1]<=60000:
                    print("GridSearchCV")
                    svclassifier = GridSearchCV(SVC(), parameters, cv=5, n_jobs=2)
                    svclassifier.fit(X_train_, y_train_)
                    print("SVM1 best score:",file=f)
                    print(svclassifier.best_score_,file=f)
                    print("SVM1 best parameters:",file=f)
                    print(svclassifier.best_params_,file=f)
                else:
                    svclassifier=SVC(kernel='linear')
                    svclassifier.fit(X_train_, y_train_)
                    
                test_index=test_key_index.get(key)
                y_pred = svclassifier.predict(X_test[test_index])
                y_test=test_A.loc[test_index,'Stance_three_type']
                
                seperate_y_pred[test_index]=y_pred
                pickle.dump(svclassifier,open(model_path+'//'+key+'_svm_OneStage_seperate_'+str(feature_no)+"_"+preprocess+".pkl",'wb'))
    
                print(confusion_matrix(y_test,y_pred), file=f)  
                print(classification_report(y_test,y_pred), file=f)
            
            print("All:", file=f)
            print(confusion_matrix(test_A['Stance_three_type'],seperate_y_pred), file=f)    
            print(classification_report(test_A['Stance_three_type'],seperate_y_pred), file=f)  
            
        # Two Stage
        # Combine
        if (not os.path.isfile(result_path+'svm_TwoStage_combine_'+str(feature_no)+"_"+preprocess+".txt")) & (not os.path.isfile('..//result//svm_linear_TwoStage_combine_'+str(feature_no)+"_"+preprocess+".txt")):
            print("Two stage and Combine:")
            f = open(result_path+'svm_TwoStage_combine_'+str(feature_no)+"_"+preprocess+".txt", 'a+')
            print("Two stage and Combine:", file=f)
            X_train=X
            y_train=train_A['Stance_two_type']
            svclassifier = SVC()#kernel='poly', degree=8 #Gaussian：kernel='rbf' #kernel='sigmoid'
            if X.shape[1]<=60000:
                print('GridSearchCV1')
                svclassifier = GridSearchCV(svclassifier, parameters, cv=5, n_jobs=2)
                svclassifier.fit(X_train, y_train)
                print("SVM1 best score:",file=f)
                print(svclassifier.best_score_,file=f)
                print("SVM1 best parameters:",file=f)
                print(svclassifier.best_params_,file=f)
                #params1=svclassifier.best_params_
            else:
                svclassifier=SVC(kernel='linear')
                svclassifier.fit(X_train, y_train)
            
            
            notNone=train_A[train_A['Stance']!='NONE'].index
            X_train_notNone=X[notNone]
            y_train_notNone=train_A.loc[notNone,'Stance_three_type']
            if X.shape[1]<=60000:
                print('GridSearchCV2')
                svclassifier2 = GridSearchCV(SVC(), parameters, cv=5,  n_jobs=2)
                svclassifier2.fit(X_train_notNone, y_train_notNone)
                print("SVM2 best score:",file=f)
                print(svclassifier2.best_score_,file=f)
                print("SVM2 best parameters:",file=f)
                print(svclassifier2.best_params_,file=f)
                #params2=svclassifier2.best_params_
            else:
                svclassifier2=SVC(kernel='linear')
                svclassifier2.fit(X_train_notNone, y_train_notNone)
            
            #predict
            #stage1
            y_pred = svclassifier.predict(X_test)  
            y_test=test_A['Stance_two_type']
            #stage2
            notNone=np.where(y_pred>0)[0]
            X_test_notNone=X_test[notNone]
            #y_test_notNone=test_A.loc[notNone,'Stance_three_type']
            y_pred2 = svclassifier2.predict(X_test_notNone)
            
            #combine predict
            y_pred3=y_pred.copy()
            for no,i in enumerate(notNone):
                y_pred3[i]=y_pred2[no]
            
            pickle.dump(svclassifier,open(model_path+'//svm1_TwoStage_combine_'+str(feature_no)+"_"+preprocess+".pkl",'wb'))
            pickle.dump(svclassifier2,open(model_path+'//svm2_TwoStage_combine_'+str(feature_no)+"_"+preprocess+".pkl",'wb'))
            
            print(confusion_matrix(y_test,y_pred), file=f)  
            print(classification_report(y_test,y_pred), file=f)
            #print(confusion_matrix(y_test_notNone,y_pred2))  
            #print(classification_report(y_test_notNone,y_pred2))  
            print(confusion_matrix(test_A['Stance_three_type'],y_pred3), file=f) 
            print(classification_report(test_A['Stance_three_type'],y_pred3), file=f)    
            
        # seperate
        if (not os.path.isfile(result_path+'result//svm_TwoStage_seperate_'+str(feature_no)+"_"+preprocess+".txt")) & (not os.path.isfile('..//result//svm_linear_TwoStage_seperate_'+str(feature_no)+"_"+preprocess+".txt")):
            print("Two stage and Seperate:")
            f = open(result_path+'svm_TwoStage_seperate_'+str(feature_no)+"_"+preprocess+".txt", 'a+')
            print("Two stage and Seperate:", file=f)
            X_train=X
            y_train=train_A['Stance_two_type']
            
            seperate=train_A.groupby("Target")
            train_key_index=seperate.indices
            test_seperate=test_A.groupby("Target")
            test_key_index=test_seperate.indices
            
            SVMs={}
            seperate_y_pred=np.zeros(len(test_A))
            for (key,index) in train_key_index.items():
                print(key,file=f)
                X_train_=X_train[index]
                y_train_=y_train[index]
                
                if X.shape[1]<=60000:
                    svclassifier = GridSearchCV(SVC(), parameters, cv=5, n_jobs=2)
                    svclassifier.fit(X_train_, y_train_)
                    print("SVM1 best score:",file=f)
                    print(svclassifier.best_score_,file=f)
                    print("SVM1 best parameters:",file=f)
                    print(svclassifier.best_params_,file=f)
                else:
                    svclassifier=SVC(kernel='linear')
                    svclassifier.fit(X_train_, y_train_)
            
                notNone=train_A.loc[list(index),:].loc[train_A['Stance']!='NONE',:].index
                X_train_notNone_=X_train[notNone]
                y_train_notNone_=train_A.loc[list(index),:].loc[notNone,'Stance_three_type']
                if X.shape[1]<=60000:
                    svclassifier2 = GridSearchCV(SVC(), parameters, cv=5,  n_jobs=2)
                    svclassifier2.fit(X_train_notNone_, y_train_notNone_)
                    print("SVM2 best score:",file=f)
                    print(svclassifier2.best_score_,file=f)
                    print("SVM2 best parameters:",file=f)
                    print(svclassifier2.best_params_,file=f)
                else:
                    svclassifier2=SVC(kernel='linear')
                    svclassifier2.fit(X_train_notNone_, y_train_notNone_)
                
                SVMs[key]=(svclassifier,svclassifier2)
                    
                #predict
                test_index=test_key_index.get(key)
                #stage1
                y_pred = svclassifier.predict(X_test[test_index])  
                y_test=test_A.loc[test_index,'Stance_two_type']
                #stage2
                notNone=np.where(y_pred>0)[0]
                #notNone=test_A.loc[test_index,:].loc[test_A['Stance']!='NONE',:].index
                X_test_notNone=X_test[test_index][notNone]
                #y_test_notNone=test_A.loc[notNone,'Stance_three_type']
                y_pred2 = svclassifier2.predict(X_test_notNone)
                #combine predict
                y_pred3=y_pred.copy()
                for no,i in enumerate(notNone):
                    y_pred3[i]=y_pred2[no]
                
                seperate_y_pred[test_index]=y_pred3
                pickle.dump(svclassifier,open(model_path+'//'+key+'_svm1_TwoStage_seperate_'+str(feature_no)+"_"+preprocess+".pkl",'wb'))
                pickle.dump(svclassifier2,open(model_path+'//'+key+'_svm2_TwoStage_seperate_'+str(feature_no)+"_"+preprocess+".pkl",'wb'))
    
                #Evaluation
                print(confusion_matrix(y_test,y_pred), file=f)
                print(classification_report(y_test,y_pred), file=f)  
                #print(confusion_matrix(y_test_notNone,y_pred2))  
                #print(classification_report(y_test_notNone,y_pred2))  
                print(confusion_matrix(test_A.loc[test_index,'Stance_three_type'],y_pred3), file=f)
                print(classification_report(test_A.loc[test_index,'Stance_three_type'],y_pred3), file=f)  
            
            print("All:", file=f)
            print(confusion_matrix(test_A['Stance_three_type'],seperate_y_pred), file=f) 
            print(classification_report(test_A['Stance_three_type'],seperate_y_pred), file=f)

