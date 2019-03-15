import numpy as np
import BM25,pickle,tagme,os,collections
from RankMetrics import MAPScorer,mean_reciprocal_rank,r_precision

re_train_flag=True
tagme.GCUBE_TOKEN = "d387b94b-bb1b-40fe-af3f-ca30f5f3138e-843339462"

def QE_LM_Dirichlet(queries, paragraphs, en=False):
    # pseudo relevance matrix
    result=np.zeros((len(paragraphs),len(queries)),dtype=np.float)
    mu=1500
    
    collection=[w for (pid,para) in paragraphs for w in para.split(" ")]
    count_w_C={}
    sum_w_D={}
    for q_index, query in enumerate(queries):
        print("query: %s with %s words.."%(q_index,len(query.split(" "))))
        cwD=np.zeros((len(paragraphs),len(query.split(" "))),dtype=np.float)
        for no,q in enumerate(query.split(" ")):
            count_w_D={}
            for p_index, (pid,para) in enumerate(paragraphs):
                count_w_D[pid]=para.split(" ").count(q)
                cwD[p_index,no]=count_w_D.get(pid)
            if q in count_w_C.keys():
                continue
            count_w_C[q]=count_w_D
            sum_w_D[q]=sum(count_w_D.values())
        # Smoothing & i.i.d Language Model
        print(((cwD+mu*collection.count(q))/(sum(count_w_D.values())+mu)).shape)
        result[:,q_index]=np.prod((cwD+mu*collection.count(q))/(sum(count_w_D.values())+mu),axis=1)
       
    if en==False:
        for q_index in range(result.shape[1]):
            print("expanding:%s"%q_index)
            pseudo_relevance=result[:,q_index]
            top_document_index=list(pseudo_relevance.argsort()[0:10])
            for index in top_document_index:
                print("expanding:%s; top document: %s"%(q_index,index))
                words=dict(collections.Counter(paragraphs[index][1].split(" ")))
                words=sorted(words.items(), key=lambda item:item[1],reverse=True)
                queries[q_index]=queries[q_index]+" "+words[0][0]
    else:
        for q_index in range(result.shape[1]):
            pseudo_relevance=result[:,q_index]
            top_document_index=list(pseudo_relevance.argsort())
            count=0
            for index in top_document_index:
                if count>=10:
                    break
                annotations = tagme.annotate(paragraphs[index][1])
                # annotations with a score higher than 0.1
                for ann in annotations.get_annotations(0.1):
                    queries[q_index]=queries[q_index]+" "+ann.entity_title
                    count+=1
    return result,queries

def Paragraph_Rocchio(section_heading, article_paragraphs):
    heading_pid={}
    for h in section_heading:
        if "/" not in h:
            continue
        index_heading=h.split("/")[len(h.split("/"))-1]
        candidate_paragraph=[]
        for other_h in section_heading:
            if "/" not in other_h:
                continue
            if (index_heading in [i for i in other_h.split("/")]) & (other_h!=h):
                page_id=other_h.split("/")[0]
                candidate_paragraph.extend(article_paragraphs[page_id])
        if len(candidate_paragraph)>0:
            heading_pid[h]=candidate_paragraph
    return heading_pid

# paragraph    
documents=pickle.load(open('processed_data\processed_paragraph.pkl','rb'))
paragraphs=pickle.load(open('processed_data\paragraphs.pkl','rb'))
paragraph_ids=pickle.load(open('processed_data\paragraph_ids.pkl','rb'))
# query
combine=list(pickle.load(open('processed_data\combined_qid.pkl','rb')))
flat_query=pickle.load(open('processed_data\processed_query.pkl','rb'))
# train&test
train=pickle.load(open('processed_data\simulated_train.pkl','rb'))        
test=pickle.load(open('processed_data\simulated_test.pkl','rb')) #safe_environment.pkl
unique=list(np.unique(test['query']))
test_qid= np.array([unique.index(i) for i in test['query']])
# BM25
#corpus=[i.split(" ") for i in documents] # whole paragraphs
# on test set
corpus_id=np.unique(test['docid'])
corpus=[(i,documents[paragraph_ids.index(i)].split(" ")) for i in corpus_id] #if i in paragraph_ids
document_structure={}
for (no,para) in corpus:
    count=[para.count(i) for i in para]
    max_count=max(count)
    # dictionary  consisting of document id mapping to the ranked dict of words
    document_structure[no]=dict([(i,0.5+0.5*para.count(i)/max_count) for i in para])  

bm25=BM25.BM25(document_structure)

if (re_train_flag==False) & (os.path.isfile('results\BM25scores.pkl')):
    scores=pickle.load(open('results\BM25scores.pkl','rb'))
else:
    scores={}
    for query_index,query in enumerate(flat_query):
        print("BM25: %s"%query_index)
        temp=[]
        for document_id in document_structure.keys():
            s=bm25.score(query,document_id)
            if s==None:
                s=0
            temp.append(s)
        scores[query]=temp
            
    for query,score in scores.items():
        temp={}
        for i,j in zip(score,document_structure.keys()):
            temp[j]=i 
        temp=sorted(temp.items(), key=lambda item:item[1],reverse=True)
        scores[query]=dict(temp)
    
pred=[]
for qid,docid in zip(test['query'],test['docid']):
    '''
    if (qid not in combine):
        pred.append(0)
        continue
    elif (docid not in scores.get(flat_query[combine.index(qid)])):
        pred.append(0)
        continue
    '''
    pred.append(scores.get(flat_query[combine.index(qid)]).get(docid))
print('MAP value under BM25: %s' % MAPScorer(np.array(test['rel']), np.array(pred),test_qid))    
print('MRR value under BM25: %s' % mean_reciprocal_rank(np.array(test['rel']), np.array(pred),test_qid))
print('R-PRECISION value under BM25: %s' % r_precision(np.array(test['rel']), np.array(pred),test_qid))
pickle.dump(scores,open('results\BM25scores.pkl','wb'))

# Cosine Similarity
# CS + TF_IDF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

test_documents=[documents[paragraph_ids.index(i)] for i in corpus_id]
vectorizer = TfidfVectorizer(norm='l2')
tfidf_documents = vectorizer.fit_transform(test_documents)
tfidf_query=vectorizer.transform(flat_query)

if (re_train_flag==False) & (os.path.isfile('results\CS_TFIDF_scores.pkl')):
    scores=pickle.load(open('results\CS_TFIDF_scores.pkl','rb'))
else:
    scores={}
    for query_index,query in enumerate(flat_query):
        print("CS + TF_IDF: %s"%query_index)
        temp=[]
        for tfidf_document_index in range(tfidf_documents.shape[0]):
            temp.append(float(cosine_similarity(tfidf_query[query_index,:],tfidf_documents[tfidf_document_index,:])))
        scores[query]=temp
    for query,score in scores.items():
        temp={}
        for i,j in zip(score,corpus_id):
            temp[j]=i 
        temp=sorted(temp.items(), key=lambda item:item[1],reverse=True)
        scores[query]=dict(temp)
        
pred=[]
for qid,docid in zip(test['query'],test['docid']):
    pred.append(scores.get(flat_query[combine.index(qid)]).get(docid))
print('MAP value under CS_TFIDF: %s' % MAPScorer(np.array(test['rel']), np.array(pred),test_qid))    
print('MRR value under CS_TFIDF: %s' % mean_reciprocal_rank(np.array(test['rel']), np.array(pred),test_qid))
print('R-PRECISION value under CS_TFIDF: %s' % r_precision(np.array(test['rel']), np.array(pred),test_qid))
pickle.dump(scores,open('results\CS_TFIDF_scores.pkl','wb'))

# CS + TF-IDF + RM1
if (re_train_flag==False) & (os.path.isfile('processed_data\expand_query.pkl')):
    expand_query=pickle.load(open('processed_data\expand_query.pkl','rb'))
else:
    print("RM1 expanding...")
    expand_query=flat_query.copy()
    result,expand_query=QE_LM_Dirichlet(expand_query, [(i,j) for i,j in zip(corpus_id,test_documents)], en=False)
    pickle.dump(expand_query,open('processed_data/expand_query.pkl','wb'))

tfidf_query_expand=vectorizer.transform(expand_query)

if (re_train_flag==False) & (os.path.isfile('results\CS_TFIDF_RM1_scores.pkl')):
    scores=pickle.load(open('results\CS_TFIDF_RM1_scores.pkl','rb'))
else:
    scores={}
    for query_index,query in enumerate(expand_query):
        print("CS + TF_IDF + RM1: %s"%query_index)
        temp=[]
        for tfidf_document_index in range(tfidf_documents.shape[0]):
            temp.append(float(cosine_similarity(tfidf_query_expand[query_index,:],tfidf_documents[tfidf_document_index,:])))
        scores[flat_query[query_index]]=temp
    for query,score in scores.items():
        temp={}
        for i,j in zip(score,corpus_id):
            temp[j]=i 
        temp=sorted(temp.items(), key=lambda item:item[1],reverse=True)
        scores[query]=dict(temp)
        
pred=[]
for qid,docid in zip(test['query'],test['docid']):
    pred.append(scores.get(flat_query[combine.index(qid)]).get(docid))
print('MAP value under CS_TFIDF_RM1: %s' % MAPScorer(np.array(test['rel']), np.array(pred),test_qid))    
print('MRR value under CS_TFIDF_RM1: %s' % mean_reciprocal_rank(np.array(test['rel']), np.array(pred),test_qid))
print('R-PRECISION value under CS_TFIDF_RM1: %s' % r_precision(np.array(test['rel']), np.array(pred),test_qid))
pickle.dump(scores,open('results\CS_TFIDF_RM1_scores.pkl','wb'))

# CS + TF-IDF + ent-RM
if (re_train_flag==False) & (os.path.isfile('processed_data\ent_expand_query.pkl')):
    ent_expand_query=pickle.load(open('processed_data\ent_expand_query.pkl','rb'))
else:
    ent_expand_query=flat_query.copy()
    result,ent_expand_query=QE_LM_Dirichlet(ent_expand_query, [(i,j) for i,j in zip(corpus_id,test_documents)], en=True)
    pickle.dump(ent_expand_query,open('processed_data\ent_expand_query.pkl','wb'))

tfidf_query_ent_expand=vectorizer.transform(ent_expand_query)

if (re_train_flag==False) & (os.path.isfile('results\CS_TFIDF_ent-RM1_scores.pkl')):
    print("ent-RM1 expanding...")
    scores=pickle.load(open('results\CS_TFIDF_ent-RM1_scores.pkl','rb'))
else:
    scores={}
    for query_index,query in enumerate(tfidf_query_ent_expand):
        print("CS + TF_IDF + ent-RM1: %s"%query_index)
        temp=[]
        for tfidf_document_index in range(tfidf_documents.shape[0]):
            temp.append(float(cosine_similarity(tfidf_query_ent_expand[query_index,:],tfidf_documents[tfidf_document_index,:])))
        scores[flat_query[query_index]]=temp
    for query,score in scores.items():
        temp={}
        for i,j in zip(score,corpus_id):
            temp[j]=i 
        temp=sorted(temp.items(), key=lambda item:item[1],reverse=True)
        scores[query]=dict(temp)
        
pred=[]
for qid,docid in zip(test['query'],test['docid']):
    pred.append(scores.get(flat_query[combine.index(qid)]).get(docid))
print('MAP value under CS_TFIDF_ent-RM1: %s' % MAPScorer(np.array(test['rel']), np.array(pred),test_qid))    
print('MRR value under CS_TFIDF_ent-RM1: %s' % mean_reciprocal_rank(np.array(test['rel']), np.array(pred),test_qid))
print('R-PRECISION value under CS_TFIDF_ent-RM1: %s' % r_precision(np.array(test['rel']), np.array(pred),test_qid))

pickle.dump(scores,open('results/CS_TFIDF_ent-RM1_scores.pkl','wb'))

# CS_TFIDF_Rocchio
import random
if (re_train_flag==False) & (os.path.isfile('processed_data/rocchio_expand_query.pkl')):
    rocchio_expand_query=pickle.load(open('processed_data/rocchio_expand_query.pkl','rb'))
else:
    article_paragraphs=pickle.load(open('processed_data/article_paragraphs.pkl','rb'))
    section_heading=list(pickle.load(open('processed_data/section_paragraphs.pkl','rb')).keys())
    heading_pid=Paragraph_Rocchio(section_heading,article_paragraphs)
    rocchio_expand_query=flat_query.copy()
    for qid in unique:
        if qid in heading_pid.keys():
            condidata_paragraphs=heading_pid.get(qid)
            chosen=[]
            while len(chosen)<5:
                chosen.extend(random.sample(condidata_paragraphs,5-len(chosen)))
                for j in chosen:
                    if j not in paragraph_ids:
                        chosen.remove(j)
            for i in chosen:
               rocchio_expand_query[combine.index(qid)]=rocchio_expand_query[combine.index(qid)]+" "+documents[paragraph_ids.index(i)]
    pickle.dump(rocchio_expand_query, open('processed_data/rocchio_expand_query.pkl','wb'))

#flat_query=pickle.load(open('processed_data\processed_query.pkl','rb'))
tfidf_query_expand_rocchio=vectorizer.transform(rocchio_expand_query)

if (re_train_flag==False) & (os.path.isfile('results\CS_TFIDF_Rocchio_scores.pkl')):
    scores=pickle.load(open('results\CS_TFIDF_Rocchio_scores.pkl','rb'))
else:
    scores={}
    for query_index,query in enumerate(rocchio_expand_query):
        print("CS + TF_IDF + Rocchio: %s"%query_index)
        temp=[]
        for tfidf_document_index in range(tfidf_documents.shape[0]):
            temp.append(float(cosine_similarity(tfidf_query_expand_rocchio[query_index,:],tfidf_documents[tfidf_document_index,:])))
        scores[flat_query[query_index]]=temp
    for query,score in scores.items():
        temp={}
        for i,j in zip(score,corpus_id):
            temp[j]=i 
        temp=sorted(temp.items(), key=lambda item:item[1],reverse=True)
        scores[query]=dict(temp)
        
pred=[]
for qid,docid in zip(test['query'],test['docid']):
    pred.append(scores.get(flat_query[combine.index(qid)]).get(docid))
print('MAP value under CS_TFIDF_Rocchio: %s' % MAPScorer(np.array(test['rel']), np.array(pred),test_qid))    
print('MRR value under CS_TFIDF_Rocchio: %s' % mean_reciprocal_rank(np.array(test['rel']), np.array(pred),test_qid))
print('R-PRECISION value under CS_TFIDF_Rocchio: %s' % r_precision(np.array(test['rel']), np.array(pred),test_qid))
pickle.dump(scores,open('results/CS_TFIDF_Rocchio_scores.pkl','wb'))

# GloV Representation
tfidf_documents = tfidf_documents.toarray() 
tfidf_query = tfidf_query.toarray() 
tfidf_words_list=[i.lower() for i in vectorizer.get_feature_names()]

print('read GloV...')
f=open("H:\dataset\glove.6B\glove.6B.300d.txt",'rb')
lines=f.readlines()
gloV={}
#print(len(lines))
for no,line in enumerate(lines):
    print(no)
    elements=str(line)[2:-3].split(" ")
    gloV[elements[0]]=elements[1:]
#from nltk.stem.porter import PorterStemmer
#porter_stemmer = PorterStemmer()
#words=[porter_stemmer.stem(i) for i in list(gloV.keys())]
words=[i.lower() for i in list(gloV.keys())]
embeddings=list(gloV.values())
del lines

glove_documents=[]
for no,doc in enumerate(test_documents):
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
glove_query=[]    
for no,query in enumerate(flat_query):
    weighted_glove=np.zeros(300)
    query_length=0
    for i in query.split():
        query_length+=1
        if (i.lower() in words) & (i.lower() in tfidf_words_list):
            weighted_glove+=tfidf_query[no][tfidf_words_list.index(i.lower())]*np.array(embeddings[words.index(i.lower())],dtype=np.float)
    glove_query.append(weighted_glove/query_length) # pre-process: stemming

if (re_train_flag==False) & (os.path.isfile('results\CS_GLOVE_scores.pkl')):
    scores=pickle.load(open('results\CS_GLOVE_scores.pkl','rb'))
else:
    scores={}
    for query_index,query in enumerate(flat_query):
        print("CS_GLOVE: %s"%query_index)
        temp=[]
        for index in range(len(glove_documents)):
            aa = glove_query[query_index].reshape(1,300)
            ba = glove_documents[index].reshape(1,300)
            temp.append(float(cosine_similarity(aa,ba)))
        scores[query]=temp
            
    for query,score in scores.items():
        temp={}
        for i,j in zip(score,corpus_id):
            temp[j]=i 
        temp=sorted(temp.items(), key=lambda item:item[1],reverse=True)
        scores[query]=dict(temp)
        
pred=[]
for qid,docid in zip(test['query'],test['docid']):
    pred.append(scores.get(flat_query[combine.index(qid)]).get(docid))
print('MAP value under CS_GLOVE: %s' % MAPScorer(np.array(test['rel']), np.array(pred),test_qid))    
print('MRR value under CS_GLOVE: %s' % mean_reciprocal_rank(np.array(test['rel']), np.array(pred),test_qid))
print('R-PRECISION value under CS_GLOVE: %s' % r_precision(np.array(test['rel']), np.array(pred),test_qid))
pickle.dump(scores,open('results\CS_GLOVE_scores.pkl','wb'))

# CS + GloV + RM1
tfidf_query_expand = tfidf_query_expand.toarray() 

glove_query_expand=[]    
for no,query in enumerate(expand_query):
    weighted_glove=np.zeros(300)
    query_length=0
    for i in query.split():
        query_length+=1
        if (i.lower() in words) & (i.lower() in tfidf_words_list):
            weighted_glove+=tfidf_query_expand[no][tfidf_words_list.index(i.lower())]*np.array(embeddings[words.index(i.lower())],dtype=np.float)
    glove_query_expand.append(weighted_glove/query_length) # pre-process: stemming

if (re_train_flag==False) & (os.path.isfile('results\CS_GLOVE_RM1_scores.pkl')):
    scores=pickle.load(open('results\CS_GLOVE_RM1_scores.pkl','rb'))
else:
    scores={}
    for query_index,query in enumerate(expand_query):
        print("CS_GLOVE_RM1: %s"%query_index)
        temp=[]
        for index in range(len(glove_documents)):
            aa = glove_query_expand[query_index].reshape(1,300)
            ba = glove_documents[index].reshape(1,300)
            temp.append(float(cosine_similarity(aa,ba)))
        scores[flat_query[query_index]]=temp
    
    for query,score in scores.items():
        temp={}
        for i,j in zip(score,corpus_id):
            temp[j]=i 
        temp=sorted(temp.items(), key=lambda item:item[1],reverse=True)
        scores[query]=dict(temp)

pred=[]
for qid,docid in zip(test['query'],test['docid']):
    pred.append(scores.get(flat_query[combine.index(qid)]).get(docid))
print('MAP value under CS_GLOVE_RM1: %s' % MAPScorer(np.array(test['rel']), np.array(pred),test_qid))    
print('MRR value under CS_GLOVE_RM1: %s' % mean_reciprocal_rank(np.array(test['rel']), np.array(pred),test_qid))
print('R-PRECISION value under CS_GLOVE_RM1: %s' % r_precision(np.array(test['rel']), np.array(pred),test_qid))

pickle.dump(scores,open('results\CS_GLOVE_RM1_scores.pkl','wb'))

# CS_GLOVE_ent-RM
tfidf_query_ent_expand = tfidf_query_ent_expand.toarray() 

glove_query_ent_expand=[]    
for no,query in enumerate(ent_expand_query):
    weighted_glove=np.zeros(300)
    query_length=0
    for i in query.split():
        query_length+=1
        if (i.lower() in words) & (i.lower() in tfidf_words_list):
            weighted_glove+=tfidf_query_ent_expand[no][tfidf_words_list.index(i.lower())]*np.array(embeddings[words.index(i.lower())],dtype=np.float)
    glove_query_ent_expand.append(weighted_glove/query_length) # pre-process: stemming

if (re_train_flag==False) & (os.path.isfile('results\CS_GLOVE_ent-RM1_scores.pkl')):
    scores=pickle.load(open('results\CS_GLOVE_ent-RM1_scores.pkl','rb'))
else:
    scores={}
    for query_index,query in enumerate(ent_expand_query):
        print("CS_GLOVE_ent-RM1: %s"%query_index)
        temp=[]
        for index in range(len(glove_documents)):
            aa = glove_query_ent_expand[query_index].reshape(1,300)
            ba = glove_documents[index].reshape(1,300)
            temp.append(float(cosine_similarity(aa,ba)))
        scores[flat_query[query_index]]=temp
    
    for query,score in scores.items():
        temp={}
        for i,j in zip(score,corpus_id):
            temp[j]=i 
        temp=sorted(temp.items(), key=lambda item:item[1],reverse=True)
        scores[query]=dict(temp)

pred=[]
for qid,docid in zip(test['query'],test['docid']):
    pred.append(scores.get(flat_query[combine.index(qid)]).get(docid))
print('MAP value under CS_GLOVE_ent-RM1: %s' % MAPScorer(np.array(test['rel']), np.array(pred),test_qid))    
print('MRR value under CS_GLOVE_ent-RM1: %s' % mean_reciprocal_rank(np.array(test['rel']), np.array(pred),test_qid))
print('R-PRECISION value under CS_GLOVE_ent-RM1: %s' % r_precision(np.array(test['rel']), np.array(pred),test_qid))

pickle.dump(scores,open('results\CS_GLOVE_ent-RM1_scores.pkl','wb'))

# CS_GLOVE_Rocchio
tfidf_query_expand_rocchio = tfidf_query_expand_rocchio.toarray() 

glove_query_expand_rocchio=[]    
for no,query in enumerate(rocchio_expand_query):
    weighted_glove=np.zeros(300)
    query_length=0
    for i in query.split():
        query_length+=1
        if (i.lower() in words) & (i.lower() in tfidf_words_list):
            weighted_glove+=tfidf_query_expand_rocchio[no][tfidf_words_list.index(i.lower())]*np.array(embeddings[words.index(i.lower())],dtype=np.float)
    glove_query_expand_rocchio.append(weighted_glove/query_length) # pre-process: stemming

if (re_train_flag==False) & (os.path.isfile('results\CS_GLOVE_Rocchio_scores.pkl')):
    scores=pickle.load(open('results\CS_GLOVE_Rocchio_scores.pkl','rb'))
else:
    scores={}
    for query_index,query in enumerate(rocchio_expand_query):
        print("CS_GLOVE_Rocchio: %s"%query_index)
        temp=[]
        for index in range(len(glove_documents)):
            aa = glove_query_expand_rocchio[query_index].reshape(1,300)
            ba = glove_documents[index].reshape(1,300)
            temp.append(float(cosine_similarity(aa,ba)))
        scores[flat_query[query_index]]=temp
            
    for query,score in scores.items():
        temp={}
        for i,j in zip(score,corpus_id):
            temp[j]=i 
        temp=sorted(temp.items(), key=lambda item:item[1],reverse=True)
        scores[query]=dict(temp)
        
pred=[]
for qid,docid in zip(test['query'],test['docid']):
    pred.append(scores.get(flat_query[combine.index(qid)]).get(docid))
print('MAP value under CS_GLOVE_Rocchio: %s' % MAPScorer(np.array(test['rel']), np.array(pred),test_qid))    
print('MRR value under CS_GLOVE_Rocchio: %s' % mean_reciprocal_rank(np.array(test['rel']), np.array(pred),test_qid))
print('R-PRECISION value under CS_GLOVE_Rocchio: %s' % r_precision(np.array(test['rel']), np.array(pred),test_qid))

pickle.dump(scores,open('results\CS_GLOVE_Rocchio_scores.pkl','wb'))

# Learning to Rank
from CoordinateAscent.coordinate_ascent import CoordinateAscent

def LR(X, y, qid,X_test, y_test, qid_test):

    model = CoordinateAscent(n_restarts=1,
                             max_iter=25,
                             scorer=None)
    #X_valid, y_valid, qid_valid = load_svmlight_file(args.valid_file, query_id=True)
    #model.fit(X, y, qid, X_valid, y_valid, qid_valid)
    model.fit(X, y, qid)
    pred = model.predict(X_test, qid_test)

    print('MAP Score: %s' % MAPScorer(y_test, pred, qid_test))
    print('MRR Score: %s' % mean_reciprocal_rank(y_test, pred, qid_test))
    print('R-PRECISION Score: %s' % r_precision(y_test, pred, qid_test))

CS_TFIDF_scores=pickle.load(open('results\CS_TFIDF_scores.pkl','rb'))
CS_GLOVE_scores=pickle.load(open("results\CS_GLOVE_scores.pkl",'rb'))
CS_TFIDF_Rocchio_scores=pickle.load(open('results\CS_TFIDF_Rocchio_scores.pkl','rb'))
CS_GLOVE_Rocchio_scores=pickle.load(open("results\CS_GLOVE_Rocchio_scores.pkl",'rb'))
CS_TFIDF_RM1_scores=pickle.load(open('results\CS_TFIDF_RM1_scores.pkl','rb'))
CS_GLOVE_RM1_scores=pickle.load(open("results\CS_GLOVE_RM1_scores.pkl",'rb'))
CS_TFIDF_ent_RM1_scores=pickle.load(open('results\CS_TFIDF_ent-RM1_scores.pkl','rb'))
CS_GLOVE_ent_RM1_scores=pickle.load(open("results\CS_GLOVE_ent-RM1_scores.pkl",'rb'))

X_train=np.zeros((len(train),8),dtype=np.float)
X_test=np.zeros((len(test),8),dtype=np.float)
train.sort_values(by=["query",'docid'], inplace=True)
test.sort_values(by=["query",'docid'], inplace=True)
y_train=train['rel']
y_test=test['rel']
unique=list(np.unique(train['query']))
train_qid= np.array([unique.index(i) for i in train['query']])
unique=list(np.unique(test['query']))
test_qid= np.array([unique.index(i) for i in test['query']])
no=0
for qid, docid in zip(train['query'],train['docid']):
    index=flat_query[combine.index(qid)]
    X_train[no,0]=CS_TFIDF_scores.get(index).get(docid)
    X_train[no,1]=CS_GLOVE_scores.get(index).get(docid)
    X_train[no,2]=CS_TFIDF_Rocchio_scores.get(index).get(docid)
    X_train[no,3]=CS_GLOVE_Rocchio_scores.get(index).get(docid)
    X_train[no,4]=CS_TFIDF_RM1_scores.get(index).get(docid)
    X_train[no,5]=CS_GLOVE_RM1_scores.get(index).get(docid)
    X_train[no,6]=CS_TFIDF_ent_RM1_scores.get(index).get(docid)
    X_train[no,7]=CS_GLOVE_ent_RM1_scores.get(index).get(docid)
    
    no+=1
    
no=0
for qid, docid in zip(test['query'],test['docid']):
    index=flat_query[combine.index(qid)]
    X_test[no,0]=CS_TFIDF_scores.get(index).get(docid)
    X_test[no,1]=CS_GLOVE_scores.get(index).get(docid)
    X_test[no,2]=CS_TFIDF_Rocchio_scores.get(index).get(docid)
    X_test[no,3]=CS_GLOVE_Rocchio_scores.get(index).get(docid)
    X_test[no,4]=CS_TFIDF_RM1_scores.get(index).get(docid)
    X_test[no,5]=CS_GLOVE_RM1_scores.get(index).get(docid)
    X_test[no,6]=CS_TFIDF_ent_RM1_scores.get(index).get(docid)
    X_test[no,7]=CS_GLOVE_ent_RM1_scores.get(index).get(docid)
    
    no+=1
 
LR(X_train, np.array(y_train), train_qid, X_test, np.array(y_test), test_qid)