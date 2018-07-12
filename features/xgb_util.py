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

        # XXX
        # pd.DataFrame(
        #     OrderedDict(
        #         type='xgb',
        #         n_trees=n_trees,
        #         tree_i=tree_i,
        #         node_count=node_count,
        #         node_i=node_i,
        #         is_leaf=':leaf=' in node_str,
        #         node_depth=ilen(takewhile(lambda x: x == '\t', node_str)),
        #         node_str=node_str,
        #     )
        #     for tree_i, n_trees, node_strs in enumerate_with_n(tree_node_strs)
        #     for node_i, node_count, node_str in enumerate_with_n(node_strs)
        # )
        # # Two versions of sql agg-over-partition-by, keeping both for reference:
        # #
        # # Version 1: simple code, a little slower:
        # # .assign(
        # #     depth=lambda df: df.groupby('tree_i').node_depth.transform(np.max),
        # #     leaf_count=lambda df: df.groupby('tree_i').is_leaf.transform(lambda x: x.sum()),
        # #     fork_count=lambda df: df.groupby('tree_i').is_leaf.transform(lambda x: (~x).sum()),
        # # )
        # #
        # # Version 2: not-simple code, a little faster:
        # .pipe(lambda df: df.merge(on='tree_i', how='left', right=(df
        #     .groupby('tree_i')[['node_depth', 'is_leaf']].agg({
        #         'node_depth': {
        #             'depth': np.max,
        #         },
        #         'is_leaf': {
        #             'leaf_count': lambda x: x.sum(),
        #             'fork_count': lambda x: (~x).sum(),
        #         },
        #     })
        #     .reset_index()
        # )))
        # .pipe(df_reorder_cols,
        #     first=['n_trees', 'tree_i', 'node_count', 'leaf_count', 'fork_count', 'depth'],
        #     last=['node_str'],
        # )

    )
