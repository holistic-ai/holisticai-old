import random
import pandas as pd

def generate_rankings(M, k:int, p):
    """
    Generates M rankings of n elements using Yang-Stoyanovich process
    :param M:           how many rankings to generate
    :param k:           how many elements should each ranking have
    :param p:           what is the probability that a candidate is protected
    :return:            the generated rankings (list of lists of FairScoreDoc))
    """
    rankings = []
    for m in range(M):
        for i in range(k):
            is_protected = (random.random() <= p)
            gender = 'Male' if is_protected else 'Female'
            rankings.append({'query_id':m, 'doc_id':k-i, 'score':k-i, 'gender': gender})
    return pd.DataFrame(rankings)