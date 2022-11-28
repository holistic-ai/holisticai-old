def fair_top_k(k, protected_candidates, non_protected_candidates, mtable):
    """    
    Parameters:
    ----------
    k : int
        the expected length of the ranking
    protected_candidates : [FairScoreDoc]
        array of protected class:`candidates <fairsearhcore.models.FairScoreDoc>`, assumed to be
        sorted by item score in descending order
    non_protected_candidates : [FairScoreDoc]
        array of non-protected class:`candidates <fairsearhcore.models.FairScoreDoc>`, assumed to be
        sorted by item score in descending order
        significance level for the binomial cumulative distribution function -> minimum probability at
        which a fair ranking contains the minProp amount of protected candidates
    Return:
    ------
    an array of elements in the form of `dict` that maximizes ordering and
    selection fairness
    the left-over candidates that were not selected into the ranking, sorted color-blindly
    """

    result = []
    countProtected = 0

    idxProtected = 0
    idxNonProtected = 0

    for i in range(k):
        if idxProtected >= len(protected_candidates) and idxNonProtected >= len(non_protected_candidates):
            # no more candidates available, return list shorter than k
            return result, []
        if idxProtected >= len(protected_candidates):
            # no more protected candidates available, take non-protected instead
            result.append(non_protected_candidates[idxNonProtected])
            idxNonProtected += 1

        elif idxNonProtected >= len(non_protected_candidates):
            # no more non-protected candidates available, take protected instead
            result.append(protected_candidates[idxProtected])
            idxProtected += 1
            countProtected += 1
        elif countProtected < mtable[i]:
            # add a protected candidate
            result.append(protected_candidates[idxProtected])
            idxProtected += 1
            countProtected += 1
        else:
            # find the best candidate available
            if protected_candidates[idxProtected].score >= non_protected_candidates[idxNonProtected].score:
                # the best is a protected one
                result.append(protected_candidates[idxProtected])
                idxProtected += 1
                countProtected += 1
            else:
                # the best is a non-protected one
                result.append(non_protected_candidates[idxNonProtected])
                idxNonProtected += 1

    return result # , __mergeTwoRankings(protected_candidates[idxProtected:], non_protected_candidates[idxNonProtected:])


def __mergeTwoRankings(ranking1, ranking2):
    result = ranking1 + ranking2
    result.sort(key=lambda candidate: candidate, reverse=True)
    return result