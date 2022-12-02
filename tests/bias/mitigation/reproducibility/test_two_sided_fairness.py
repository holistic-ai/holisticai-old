import os
import sys

import numpy as np

sys.path.append(os.getcwd())
np.random.seed(42)


def test_two_sided_fairness():
    from holisticai.datasets import load_last_fm
    from holisticai.utils import recommender_formatter

    bunch = load_last_fm()
    lastfm = bunch["frame"]
    lastfm["score"] = 1
    lastfm = lastfm.iloc[:500]
    df_pivot, p_attr = recommender_formatter(
        lastfm,
        users_col="user",
        groups_col="sex",
        items_col="artist",
        scores_col="score",
        aggfunc="mean",
    )
    data_matrix = df_pivot.fillna(0).to_numpy()
    numUsers, _ = data_matrix.shape

    from holisticai.bias.mitigation import FairRec

    # size of recommendation
    rec_size = 10

    for alpha in np.arange(0,1,0.1):
        recommender = FairRec(rec_size, alpha)
        res = recommender.fit(data_matrix)
        assert len(res.keys()) == numUsers

        for key in res.keys():
            assert len(res[key]) == rec_size
