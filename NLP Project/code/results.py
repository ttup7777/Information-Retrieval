# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 21:24:52 2019

@author: cassie
"""

import os
path = "F://大学生活//Delft//IR//NLP//project//result" #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称
s = []
feature_map={0:1, 1:2, 2:3, 3:4, 8:5, 4:6, 7:7, 20:8, 21:9, 5:10, 9:11, 10:12, 11:13, 12:14, 13:15, 14:16,
             15:17, 16:18, 17:19, 18:20, 19:21, 6:22}
preprocess_design={'preclean_Tweet':"Necessary",'lemmatized_Tweet':"Necessary + Lemmatization",
                   'stemmed_Tweet':"Necessary + Stemming",'preclean_noslang_Tweet':"Necessary + Slang Removal",
                   'noslang_lemmatized_Tweet':"Necessary + Slang Removal + Lemmatization",
                   'noslang_stemmed_Tweet':"Necessary + Slang Removal + Stemming"}

wrong=[]
for file in files: #遍历文件夹
    if ('svm_' in file) : 
        if ("noslang" in file) :
            model_train=' + '.join(file.split(".")[0].split("_")[-6:-4])
            feature_no='Feature %s' %feature_map[int(file.split(".")[0].split("_")[-4])]
            preprocess=preprocess_design['_'.join(file.split(".")[0].split("_")[-3:])]
        else:
            model_train=' + '.join(file.split(".")[0].split("_")[-5:-3])
            feature_no='Feature %s' % feature_map[int(file.split(".")[0].split("_")[-3])]
            preprocess=preprocess_design['_'.join(file.split(".")[0].split("_")[-2:])]
        model_train=model_train.replace("seperate","separate")
    
        f = open(path+"/"+file); #打开文件
        #iter_f = iter(f); #创建迭代器
        str = ""
        lines=f.readlines()
        p_f=0
        n_f=0
        for no, line in enumerate(lines): #遍历文件，一行行遍历，读取文本
            try:
                if no==len(lines)-7:
                    print(line)
                    p_f=float(list(filter(None,line.split(" ")))[3])
                if no==len(lines)-6:
                    print(line)
                    n_f=float(list(filter(None,line.split(" ")))[3])
                if no==len(lines)-3:
                    print(line)
                    if 'macro avg' not in line:
                        wrong.append(file)
                    x = list(filter(None,line.split(" ")))
                    ss = [model_train, preprocess,feature_no, x[2], x[3], x[4], "%.2f"%((p_f+n_f)/2)]
                    if ss not in s:
                        s.append(ss) #每个文件的文本存到list中
                    break
            except:
                if no==len(lines)-8:
                    print(line)
                    p_f=float(list(filter(None,line.split(" ")))[3])
                if no==len(lines)-7:
                    print(line)
                    n_f=float(list(filter(None,line.split(" ")))[3])
                if no==len(lines)-3:
                    print(line)
                    if 'macro avg' not in line:
                        wrong.append(file)
                    x = list(filter(None,line.split(" ")))
                    
                    ss = [model_train, preprocess,feature_no, x[2], x[3], x[4], "%.2f"%((p_f+n_f)/2)] #每个文件的文本存到list中
                    if ss not in s:
                        s.append(ss) #每个文件的文本存到list中
                    break
s.sort(key=lambda x: (x[0], x[1], int(x[2].split(" ")[1])))
import pandas as pd
ss=pd.DataFrame(s,columns=['model','preprocess','feature','p','r','f','of'])
f = open('..//result//results.txt', 'w')
temp=ss.groupby(['model','preprocess'])
p_max=dict(temp.max()['p'])
r_max=dict(temp.max()['r'])
f_max=dict(temp.max()['f'])
of_max=dict(temp.max()['of'])

result=[]
for i in s:
    i3=float(i[3])
    i4=float(i[4])
    i5=float(i[5])
    i6=float(i[6])
    
    if 'separate' in i[0]:
        if i3<=0.46:
            i[3]='\\textbf{{\color{blue}{'+i[3]+'}}}'
        if i4<=0.46:
            i[4]='\\textbf{{\color{blue}{'+i[4]+'}}}'
        if i5<=0.43:
            i[5]='\\textbf{{\color{blue}{'+i[5]+'}}}'
        if i6<=1:
            i[6]='\\textbf{{\color{blue}{'+i[6]+'}}}'
    else:
        if i3<=0.19:
            i[3]='\\textbf{{\color{blue}{'+i[3]+'}}}'
        if i4<=0.33:
            i[4]='\\textbf{{\color{blue}{'+i[4]+'}}}'
        if i5<=0.24:
            i[5]='\\textbf{{\color{blue}{'+i[5]+'}}}'
        if i6<=0.36:
            i[6]='\\textbf{{\color{blue}{'+i[6]+'}}}'
            
    if i3==float(p_max.get((i[0],i[1]))):
        i[3]='\\textbf{{\color{red}{'+i[3]+'}}}'
    if i4==float(r_max.get((i[0],i[1]))):
        i[4]='\\textbf{{\color{red}{'+i[4]+'}}}'
    if i5==float(f_max.get((i[0],i[1]))):
        i[5]='\\textbf{{\color{red}{'+i[5]+'}}}'
    if i6==float(of_max.get((i[0],i[1]))):
        i[6]='\\textbf{{\color{red}{'+i[6]+'}}}'
        
    result.append(' & '.join(i)+" \\\\") 

for no, line in enumerate(result):
    f.write(line+'\n')
    if (no==len(result)-1) :
        f.write('\\hline \n')
    elif (line.split(" & ")[0:2]!=result[no+1].split(" & ")[0:2]):
        f.write('\\hline \n\n')
     
f.close()
#print(s) #打印结果

