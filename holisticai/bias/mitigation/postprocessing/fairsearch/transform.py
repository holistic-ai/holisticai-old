import pandas as pd
import warnings
import numpy as np

from .algorithm_utils import mtable_generator
from .algorithm_utils import fail_prob
from .algorithm_utils import re_ranker

class ResultInfo:
    def __init__(self, query_id, doc_id, group_id, score):
        self.query_id = query_id
        self.doc_id = doc_id
        self.group_id = group_id
        self.score = score
        

class FairSearch:
    def __init__(self, top_n: int, p: float, alpha: float, query_col='query_id', doc_col='doc_id', group_col='group_id', score_col='score'):
        # check the parameters first
        _validate_basic_parameters(top_n, p, alpha)
        self.query_col = query_col
        self.doc_col = doc_col
        self.group_col = group_col
        self.score_col = score_col
        # assign the parameters
        self.top_n = top_n # the total number of elements
        self.p = p # the proportion of protected candidates in the top-k ranking
        self.alpha = alpha # the significance level

        self. _cache = {}  # stores generated mtables in memory
        
    def transform(self, rankings):
        results = []
        for query_id, gdf in rankings.groupby(self.query_col):
            ranking = []
            for _,row in gdf.iterrows():
                ranking.append(ResultInfo(query_id=query_id, doc_id=row[self.doc_col], group_id=row[self.group_col], score=row[self.score_col]))
            results.append(ranking)
        re_rankings = []
        for ranking in results:
            if not self.is_fair(ranking):
                ranking = self.re_rank(ranking)
            for result in ranking:
                re_rankings.append({self.query_col:result.query_id, 
                                    self.doc_col:result.doc_id, 
                                    self.group_col: result.group_id, 
                                    self.score_col:result.score})
        return pd.DataFrame(re_rankings)
        
    def create_unadjusted_mtable(self):
        """
        Creates an mtable using alpha unadjusted
        :return:
        """
        return self._create_mtable(self.alpha, False)

    def create_adjusted_mtable(self):
        """
        Creates an mtable using alpha adjusted
        :return:
        """
        return self._create_mtable(self.alpha, True)

    def _create_mtable(self, alpha, adjust_alpha):
        """
        Creates an mtable by using the passed alpha value
        :param alpha:           The significance level
        :param adjust_alpha:    Boolean indicating whether the alpha be adjusted or not
        :return:
        """

        if not (self.top_n, self.p, self.alpha, adjust_alpha) in self._cache:
            # check if passed alpha is ok
            _validate_alpha(alpha)

            # create the mtable
            fc = mtable_generator.MTableGenerator(self.top_n, self.p, alpha, adjust_alpha)

            # store as list
            self._cache[(self.top_n, self.p, self.alpha, adjust_alpha)] = fc.mtable_as_list()

        # return from cache
        return self._cache[(self.top_n, self.p, self.alpha, adjust_alpha)]

    def adjust_alpha(self) :
        """
        Computes the alpha adjusted for the given set of parameters
        :return:
        """
        rnfpc = fail_prob.RecursiveNumericFailProbabilityCalculator(self.top_n, self.p, self.alpha)
        fpp = rnfpc.adjust_alpha()
        return fpp.alpha

    def compute_fail_probability(self, mtable):
        """
        Computes analytically the probability that a ranking created with the simulator will fail to pass the mtable
        :return:
        """
        if len(mtable) != self.top_n:
            raise ValueError("Number of elements k and mtable length must be equal!")

        rnfpc = fail_prob.RecursiveNumericFailProbabilityCalculator(self.top_n, self.p, self.alpha)

        mtable_df = pd.DataFrame(columns=["m"])

        # transform the list into an pd.DataFrame
        for i in range(1, len(mtable) + 1):
            mtable_df.loc[i] = [mtable[i-1]]

        return rnfpc.calculate_fail_probability(mtable_df)

    def is_fair(self, ranking):
        """
        Checks if the ranking is fair for the given parameters
        :param ranking:     The ranking to be checked (list of FairScoreDoc)
        :return:
        """
        return check_ranking(ranking, self.create_adjusted_mtable())

    def re_rank(self, ranking):
        """
        Applies FA*IR re-ranking to the input ranking with an adjusted mtable
        :param ranking:     The ranking to be re-ranked (list of FairScoreDoc)
        :return:
        """
        return self._re_rank(ranking, True)

    def _re_rank_unadjusted(self, ranking):
        """
        Applies FA*IR re-ranking to the input ranking with an unadjusted mtable
        :param ranking:     The ranking to be re-ranked (list of FairScoreDoc)
        :return:
        """
        return self._re_rank(ranking, False)

    def _re_rank(self, ranking, adjust):
        """
        Applies FA*IR re-ranking to the input ranking and boolean whether to use an adjusted mtable
        :param ranking:     The ranking to be re-ranked (list of FairScoreDoc)
        :return:
        """
        protected = []
        non_protected = []
        for item in ranking:
            if item.is_protected:
                protected.append(item)
            else:
                non_protected.append(item)
        mtable = self.create_adjusted_mtable() if adjust else self.create_unadjusted_mtable()
        return re_ranker.fair_top_k(self.top_n, protected, non_protected, mtable)


def check_ranking(ranking, mtable):
    """
    Checks if the ranking is fair in respect to the mtable
    :param ranking:     The ranking to be checked (list of FairScoreDoc)
    :param mtable:      The mtable against to check (list of int)
    :return:            Returns whether the rankings satisfies the mtable
    """
    count_protected = 0

    # if the mtable has a different number elements than there are in the top docs return false
    if len(ranking) != len(mtable):
        raise ValueError("Number of documents in ranking and mtable length must be equal!")

    # check number of protected element at each rank
    for i, element in enumerate(ranking):
        count_protected += 1 if element.group_id else 0
        if count_protected < mtable[i]:
            return False
    return True


def _validate_basic_parameters(k, p, alpha):
    """
    Validates if k, p and alpha are in the required ranges
    :param k:           Total number of elements (above or equal to 10)
    :param p:           The proportion of protected candidates in the top-k ranking (between 0.02 and 0.98)
    :param alpha:       The significance level (between 0.01 and 0.15)
    """
    if k < 10 or k > 400:
        if k < 2:
            raise ValueError("Total number of elements `k` should be between 10 and 400")
        else:
            warnings.warn("Library has not been tested with values outside this range")

    if p < 0.02 or p > 0.98:
        if p < 0 or p > 1:
            raise ValueError("The proportion of protected candidates `p` in the top-k ranking should be between "
                             "0.02 and 0.98")
        else:
            warnings.warn("Library has not been tested with values outside this range")

    _validate_alpha(alpha)


def _validate_alpha(alpha):
    if alpha < 0.01 or alpha > 0.15:
        if alpha < 0.001 or alpha > 0.5:
            raise ValueError("The significance level `alpha` must be between 0.01 and 0.15")
        else:
            warnings.warn("Library has not been tested with values outside this range")

