from collections import OrderedDict
from itertools import takewhile

from more_itertools import ilen
import numpy as np
import pandas as pd
from potoo.numpy import np_sample
from potoo.pandas import df_reorder_cols
import xgboost as xgb

from util import enumerate_with_n, model_stats
import xgb_sklearn_hack


@model_stats.register(xgb.sklearn.XGBClassifier)
@model_stats.register(xgb_sklearn_hack.XGBClassifier)
def _(gbm, **kwargs) -> pd.DataFrame:
    return model_stats(gbm._Booster, **kwargs)


@model_stats.register(xgb.Booster)
def _(gbm, sample_tree_n=None, random_state=None) -> pd.DataFrame:
    tree_strs = gbm.get_dump(
        with_stats=False,  # Unused, avoid the extra cpu time
    )
    tree_node_strs = [[node for node in tree.rstrip('\n').split('\n')] for tree in tree_strs]
    if sample_tree_n:
        tree_node_strs = np_sample(tree_node_strs, n=sample_tree_n, replace=False, random_state=random_state)
    return (
        pd.DataFrame(
            OrderedDict(
                type='xgb',
                n_trees=n_trees,
                tree_i=tree_i,
                depth=max(ilen(takewhile(lambda c: c == '\t', node_str)) for node_str in node_strs),
                node_count=len(node_strs),
                leaf_count=sum(':leaf=' in node_str for node_str in node_strs),
            )
            for tree_i, n_trees, node_strs in enumerate_with_n(tree_node_strs)
        )
        .assign(
            fork_count=lambda df: df.node_count - df.leaf_count,
        )
        .pipe(df_reorder_cols,
            first=['n_trees', 'tree_i', 'depth', 'node_count', 'leaf_count', 'fork_count'],
        )
    )
