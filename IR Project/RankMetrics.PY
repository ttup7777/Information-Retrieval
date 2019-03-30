import numpy as np
from CoordinateAscent.utils import group_offsets

def mrr(y_true, y_pred):
    order = np.argsort(-y_pred)
    p_k = 0
    _y_true = np.take(y_true, order)
    for k in range(1, len(np.where(y_true > 0)[0])):
        if _y_true[k - 1] <= 0:
            continue
        else:
            p_k = 1 / k
            break
    return p_k


def mean_reciprocal_rank(y_true, y_pred, qid):
    # MAP (Mean Average Precision)
    return np.mean(np.array([mrr(y_true[a:b], y_pred[a:b]) for a, b in group_offsets(qid)]))
        

def rp(y_true, y_pred):
    order = np.argsort(-y_pred)
    _y_true = np.take(y_true, order)
    length=len(np.where(y_true > 0)[0])
    p_k = np.sum(_y_true[:length] > 0) / length
    return p_k


def r_precision(y_true, y_pred, qid):
    return np.mean(np.array([rp(y_true[a:b], y_pred[a:b]) for a, b in group_offsets(qid)]))


# MAP (Mean Average Precision)
def _map_score(y_true, y_pred):
    #print(y_pred)
    order = np.argsort(-y_pred)
    p_ks=0
    _y_true = np.take(y_true, order)
    for k in range(1,len(np.where(y_true > 0)[0])):
        if _y_true[k-1]<=0:
            continue
        p_k=np.sum(_y_true[:k] > 0) / k
        p_ks+=p_k
    return p_ks/len(np.where(y_true > 0)[0]) if len(np.where(y_true > 0)[0])>0 else 0

def MAPScorer(y_true, y_pred, qid):
    return np.mean(np.array([_map_score(y_true[a:b], y_pred[a:b]) for a, b in group_offsets(qid)]))