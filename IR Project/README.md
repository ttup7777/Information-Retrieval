# Core IR project
The dataset and python code here are utilized by us for reproducing the paper of Nanni et al. [1] in Core IR project.

### Introduction of The Project
Answering complex questions is a challenging task which requires retrieval approaches to show multiple relevance facets of a complex question from many different Internet pages. To solve this problem, one of idea in information retrieval is to organize the comprehensive answers for a complex question into a single page, like Wikipedia articles. The new TREC Complex Answer Retrieval (TREC CAR) track provides several comprehensive dataset that target at retrieving paragraphs to populate such articles about complex topics. Inspired by Nanni et al. [1], we reproduce part of their work and rely on a small-size dataset Test200 in TREC CAR to evaluate a variety of retrieval approaches that can be applied in this task, from standard ones (e.g., BM25, vector model with different embeddings) to complex ones that using learning to rank or query expansion with knowledge bases. The goal of our project is to provide an overview of some promising methods in tackling this problem to future participants of this track.

### Introduction of The Code
- **Test Script: code.py**. This file read the preprocessed data and test the performance of the following retrieval approaches. All scores are dumped into pkl files under the folder "results" (ps: those files are too large to be uploaded!).
      
      - Baseline: query only + Okapi BM25
      - TF-IDF vector representation + query only + Cosine Similarity.
      - TF-IDF vector representation + query +RM1 + Cosine Similarity.
      - TF-IDF vector representation + query +ent-RM1 + Cosine Similarity.
      - TF-IDF vector representation + query + Rocchio + Cosine Similarity.
      - Word embeddings representation (GloVE, with TF-IDF taken into account) + query only + Cosine Similarity.
      - Word embeddings representation (GloVE, with TF-IDF taken into account) + query + RM1 + Cosine Similarity.
      - Word embeddings representation (GloVE, with TF-IDF taken into account) + query + ent-RM1 + Cosine Similarity.
      - Word embeddings representation (GloVE, with TF-IDF taken into account) + query + Rocchio + Cosine Similarity.
      - Cosine Similarity Results + Learning to Rank (Linear Regression)
- **Retrieval Models: BM25.py and the files under the folder CoordinateAscent**
- **Evaluation Matrices: RankMetrics.PY**. This file includes the code for calculating MAP, R-PREC and MRR.
- **Data Preprocess: data_preprocess.py**. This file deals with pre-processing dataset, including removing stopwords, digits and performing stemming. In addition, it also serves for constructing the the Wikipage structure and the simulated train set and test set with noise. All pre-processed data are dumped into pkl files under the folder "processed_data".

### Contact Us
If you have any problem with our scripts or dataset, please contact us by email.

Hong Lin: H.LIN-11@student.tudelft.nl

Kang Lang: K.Lang@student.tudelft.nl

Tian Tian: T.Tian-1@student.tudelft.nl

## Reference
[1] Nanni F., Mitra B., Magnusson M., Dietz L. (2017) Benchmark for complex answer retrieval. In: ICTIR 2017.
