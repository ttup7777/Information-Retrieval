# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:05:49 2019

@author: cassie
"""

import pandas as pd
import random
from trectools import TrecQrel
import trec_car.read_data
import numpy as np
import pickle,re,string
from nltk.corpus import stopwords

count=0
paragraphs=[]
for para in trec_car.read_data.iter_paragraphs(open('H://dataset//data//test200-train//train.pages.cbor-paragraphs.cbor', 'rb')):
    paragraphs.append(para)
    count+=1
print('number of paragraphs: %s'%count)
paragraph_ids=[i.para_id for i in paragraphs]
pickle.dump(paragraphs,open('processed_data\paragraphs.pkl','wb'))
pickle.dump(paragraph_ids,open('processed_data\paragraph_ids.pkl','wb'))
# preprocess paragraph
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
regex = re.compile('[%s]' % re.escape(string.punctuation))
stopwords=list(set(stopwords.words('english')))
#[line.strip() for line in open("ENstopwords891.txt", 'r', encoding='utf-8').readlines()]
new_punctuation=list(string.punctuation)

documents = [i.get_text() for i in paragraphs]
for no,doc in enumerate(documents):
    documents[no]=" ".join([porter_stemmer.stem(i) for i in regex.sub(' ', doc).split() if (i!=" ") & (i not in stopwords) & (not i.isdigit()) & (i not in new_punctuation)]) # pre-process: stemming
pickle.dump(documents,open('processed_data\processed_paragraph.pkl','wb'))
        
A = TrecQrel("H://dataset//data//test200-train//train.pages.cbor-article.qrels")
article=A.qrels_data
H = TrecQrel("H://dataset//data//test200-train//train.pages.cbor-hierarchical.qrels")
hierarchical=H.qrels_data
T=TrecQrel("H://dataset//data//test200-train//train.pages.cbor-toplevel.qrels")
toplevel=T.qrels_data

#Combine the query files abour paragraph retrieval
combine=list(toplevel['query'])
combine.extend(list(hierarchical['query']))
combine.extend(list(article['query']))
combine=np.unique(combine)
flat_query=[]
for query in combine:
    query=query[7:].replace("/"," ")
    query=query.replace("%20"," ")
    # preprocess query
    flat_query.append(" ".join([porter_stemmer.stem(i) for i in regex.sub(' ', query).split() if (i!=" ") & (i not in stopwords) & (not i.isdigit()) & (i not in new_punctuation)]))
pickle.dump(flat_query,open('processed_data\processed_query.pkl','wb'))
pickle.dump(combine,open('processed_data\combined_qid.pkl','wb'))

count=0
pages=[]
for para in trec_car.read_data.iter_annotations(open('H://dataset//data//test200-train//train.pages.cbor', 'rb')):
    pages.append(para)
    count+=1
print('number of pages: %s'%count)
page_ids=[i.page_id for i in pages]

########################################## page structure #########################################
article_paragraphs={}
article_sections={}
section_paragraphs={}
paragraph_article_section={}
for page in pages:
    paragraphs=[]
    sections=[]
    for skeleton in page.skeleton:
        if (type(skeleton)==trec_car.read_data.Para):
            paragraphs.append(skeleton.paragraph.para_id)
            paragraph_article_section[skeleton.paragraph.para_id]=page.page_id
        elif (type(skeleton)==trec_car.read_data.List):
            paragraphs.append(skeleton.body.para_id)
            paragraph_article_section[skeleton.body.para_id]=page.page_id
            #print(skeleton.body.para_id)
        elif (type(skeleton)==trec_car.read_data.Section):
            sections.append(page.page_id+"/"+skeleton.headingId)
            sparagraphs=[]
            for section_p in skeleton.children:
                if (type(section_p)==trec_car.read_data.Para):
                    paragraphs.append(section_p.paragraph.para_id)
                    sparagraphs.append(section_p.paragraph.para_id)
                    paragraph_article_section[section_p.paragraph.para_id]=page.page_id+"/"+skeleton.headingId
                elif (type(section_p)==trec_car.read_data.List):
                    paragraphs.append(section_p.body.para_id)
                    sparagraphs.append(section_p.body.para_id)
                    paragraph_article_section[section_p.body.para_id]=page.page_id+"/"+skeleton.headingId
                else:
                    sections.append(page.page_id+"/"+skeleton.headingId+"/"+section_p.headingId)
                    sparagraphs_c=[]
                    for section_pp in section_p.children:
                        if (type(section_pp)==trec_car.read_data.Para):
                            paragraphs.append(section_pp.paragraph.para_id)
                            sparagraphs_c.append(section_pp.paragraph.para_id)
                            paragraph_article_section[section_pp.paragraph.para_id]=page.page_id+"/"+skeleton.headingId+"/"+section_p.headingId
                        elif (type(section_pp)==trec_car.read_data.List):
                            paragraphs.append(section_pp.body.para_id)
                            sparagraphs_c.append(section_pp.body.para_id)
                            paragraph_article_section[section_pp.body.para_id]=page.page_id+"/"+skeleton.headingId+"/"+section_p.headingId
                        else:
                            sections.append(page.page_id+"/"+skeleton.headingId+"/"+section_p.headingId+"/"+section_pp.headingId)
                            sparagraphs_cc=[]
                            for section_ppp in section_pp.children:
                                if (type(section_ppp)==trec_car.read_data.Para):
                                    paragraphs.append(section_ppp.paragraph.para_id)
                                    sparagraphs_cc.append(section_ppp.paragraph.para_id)
                                    paragraph_article_section[section_ppp.paragraph.para_id]=page.page_id+"/"+skeleton.headingId+"/"+section_p.headingId+"/"+section_pp.headingId
                                elif (type(section_ppp)==trec_car.read_data.List):
                                    paragraphs.append(section_ppp.body.para_id)
                                    sparagraphs_cc.append(section_ppp.body.para_id)
                                    paragraph_article_section[section_ppp.body.para_id]=page.page_id+"/"+skeleton.headingId+"/"+section_p.headingId+"/"+section_pp.headingId
                                else:
                                    sections.append(page.page_id+"/"+skeleton.headingId+"/"+section_p.headingId+"/"+section_pp.headingId+"/"+section_ppp.headingId)
                                    sparagraphs_ccc=[]
                                    for section_pppp in section_ppp.children:
                                        if (type(section_pppp)==trec_car.read_data.Para):
                                            paragraphs.append(section_pppp.paragraph.para_id)
                                            sparagraphs_ccc.append(section_pppp.paragraph.para_id)
                                            paragraph_article_section[section_pppp.paragraph.para_id]=page.page_id+"/"+skeleton.headingId+"/"+section_p.headingId+"/"+section_pp.headingId+"/"+section_ppp.headingId
                                
                                        elif (type(section_pppp)==trec_car.read_data.List):
                                            paragraphs.append(section_pppp.body.para_id)
                                            sparagraphs_ccc.append(section_pppp.body.para_id)
                                            paragraph_article_section[section_pppp.body.para_id]=page.page_id+"/"+skeleton.headingId+"/"+section_p.headingId+"/"+section_pp.headingId+"/"+section_ppp.headingId
                                
                                        else:
                                            print('section')
                                    section_paragraphs[page.page_id+"/"+skeleton.headingId+"/"+section_p.headingId+"/"+section_pp.headingId+"/"+section_ppp.headingId]=sparagraphs_ccc
                            section_paragraphs[page.page_id+"/"+skeleton.headingId+"/"+section_p.headingId+"/"+section_pp.headingId]=sparagraphs_cc
                    section_paragraphs[page.page_id+"/"+skeleton.headingId+"/"+section_p.headingId]=sparagraphs_c
            section_paragraphs[page.page_id+"/"+skeleton.headingId]=sparagraphs
        else:
            print("None Type")
    article_paragraphs[page.page_id]=paragraphs
    article_sections[page.page_id]=sections
    
pickle.dump(article_paragraphs,open('processed_data/article_paragraphs.pkl','wb'))
pickle.dump(article_sections,open('processed_data/article_sections.pkl','wb'))
pickle.dump(section_paragraphs,open('processed_data\section_paragraphs.pkl','wb'))
pickle.dump(paragraph_article_section,open('processed_data\paragraph_article_section.pkl','wb'))
############################################### page strucure ##########################################

# split train set and test set
Y_true=toplevel.append(hierarchical).append(article)
train=pd.DataFrame(columns=Y_true.columns)
test=pd.DataFrame(columns=Y_true.columns)
for query_index,qid in enumerate(combine):
    #qid=combine[query_index]
    print("%s/%s..."%(query_index,len(combine)))
    train=train.append(Y_true[Y_true['query']==qid])
    test=test.append(Y_true[Y_true['query']==qid])
    true_paragraphs=list(Y_true[Y_true['query']==qid]['docid'])
    # add noisy
    for docid in true_paragraphs:
        if "/" not in paragraph_article_section[docid]:
            exclude_true_index=page_ids.index(paragraph_article_section[docid].split("/")[0])
            randselect_index=[i for i in range(exclude_true_index)]
            randselect_index.extend([i for i in range(exclude_true_index+1,len(article_paragraphs))])
            noise_paragraph=[]
            while len(noise_paragraph)<10:
                num=10-len(noise_paragraph)
                select_index=random.sample(randselect_index,1)[0]
                random_other_article=article_paragraphs.get(list(article_paragraphs.keys())[select_index])
                if(len(random_other_article)<10):
                    noisy_sample=set(random.sample(random_other_article,len(random_other_article)))
                    inter = list(set(true_paragraphs).intersection(noisy_sample))
                    if len(inter)>0:
                        #print('inter')
                        for j in inter:
                            noisy_sample.remove(j)
                    for j in list(noisy_sample):
                        if j not in paragraph_ids:
                            noisy_sample.remove(j)
                    noise_paragraph.extend(noisy_sample)
                else:
                    noisy_sample=set(random.sample(random_other_article,10))
                    inter = list(set(true_paragraphs).intersection(noisy_sample))
                    if len(inter)>0:
                        #print('inter')
                        for j in inter:
                            noisy_sample.remove(j)
                    for j in list(noisy_sample):
                        if j not in paragraph_ids:
                            noisy_sample.remove(j)
                    noise_paragraph.extend(noisy_sample)
            
        else:
            section_in_article=paragraph_article_section[docid].split("/")[0]
            sections=article_sections.get(section_in_article)
            exclude_section_index=sections.index(paragraph_article_section[docid])
            randselect=sections[0:exclude_section_index]
            randselect.extend(sections[exclude_section_index+1:])
            random.shuffle(randselect)
            noise_paragraph=[]
            i=0
            while (len(noise_paragraph)<5) & (i <len(randselect)):
                
                if len(section_paragraphs.get(randselect[i]))>5-len(noise_paragraph):
                    num=5-len(noise_paragraph)
                else:
                    num=len(section_paragraphs.get(randselect[i]))
                    
                noisy_sample=set(random.sample(section_paragraphs.get(randselect[i]),num))
                inter = list(set(true_paragraphs).intersection(noisy_sample))
                if len(inter)>0:
                    #print('inter')
                    for j in inter:
                        noisy_sample.remove(j)
                for j in list(noisy_sample):
                    if j not in paragraph_ids:
                        noisy_sample.remove(j)
                noise_paragraph.extend(noisy_sample)
                i+=1
                
            exclude_true_index=page_ids.index(paragraph_article_section[docid].split("/")[0])
            randselect_index=[i for i in range(exclude_true_index)]
            randselect_index.extend([i for i in range(exclude_true_index+1,len(article_paragraphs))])
            noise_paragraph2=[]
            while len(noise_paragraph2)<10-len(noise_paragraph):
                num=10-len(noise_paragraph)-len(noise_paragraph2)
                select_index=random.sample(randselect_index,1)[0]
                random_other_article=article_paragraphs.get(list(article_paragraphs.keys())[select_index])
                if(len(random_other_article)<10):
                    noisy_sample=set(random.sample(random_other_article,len(random_other_article)))
                    inter = list(set(true_paragraphs).intersection(noisy_sample))
                    if len(inter)>0:
                        #print('inter')
                        for j in inter:
                            noisy_sample.remove(j)
                    for j in list(noisy_sample):
                        if j not in paragraph_ids:
                            noisy_sample.remove(j)
                    noise_paragraph2.extend(noisy_sample)
                else:
                    noisy_sample=set(random.sample(random_other_article,10))
                    inter = list(set(true_paragraphs).intersection(noisy_sample))
                    if len(inter)>0:
                        #print('inter')
                        for j in inter:
                            noisy_sample.remove(j)
                    for j in list(noisy_sample):
                        if j not in paragraph_ids:
                            noisy_sample.remove(j)
                    noise_paragraph2.extend(noisy_sample)
            
            noise_paragraph.extend(noise_paragraph2)
            
        test_in_this_article=list(set(article_paragraphs.get(page_ids[exclude_true_index])))
        inter = list(set(true_paragraphs).intersection(test_in_this_article))
        if len(inter)>0:
            #print('inter')
            for j in inter:
                test_in_this_article.remove(j)
        for j in list(test_in_this_article):
           if j not in paragraph_ids:
               test_in_this_article.remove(j)
               
        test_in_other_article=[]
        i=0
        random.shuffle(randselect_index)
        while len(test_in_other_article)<len(test_in_this_article):
            if len(article_paragraphs.get(list(article_paragraphs.keys())[randselect_index[i]]))>len(test_in_this_article)-len(test_in_other_article):
               num=len(test_in_this_article)-len(test_in_other_article)
            else:
               num=len(article_paragraphs.get(list(article_paragraphs.keys())[randselect_index[i]]))
            noisy_sample=set(random.sample(article_paragraphs.get(list(article_paragraphs.keys())[randselect_index[i]]),num))
            inter = list(set(true_paragraphs).intersection(noisy_sample))
            if len(inter)>0:
                #print('inter')
                for j in inter:
                    noisy_sample.remove(j)
            for j in list(noisy_sample):
               if j not in paragraph_ids:
                   noisy_sample.remove(j)
            test_in_other_article.extend(noisy_sample)
            i+=1
            
        test_in_this_article.extend(test_in_other_article)
        
        train=train.append(pd.DataFrame([[qid,0,i,0] for i in noise_paragraph],columns=train.columns))
        test=test.append(pd.DataFrame([[qid,0,i,0] for i in test_in_this_article],columns=test.columns))

train.drop_duplicates(inplace=True)
test.drop_duplicates(inplace=True)
test.sort_values(by="query", inplace=True) 
train.sort_values(by="query", inplace=True) 
pickle.dump(train,open('processed_data\simulated_train.pkl','wb'))        
pickle.dump(test,open('processed_data\simulated_test.pkl','wb'))
