from notebooks import *
print_sys_info()


def train_final   (**kwargs): return train_cv(**{**dict(
    logreg_Cs=[
        # TODO(train_us): Run train_final
        #   - TODO(train_us): Run Thu (~1-2h): losing preempt instances too fast, and don't want to leave a non-preemt running overnight
        .001,
    ],
    n_species_n_recs=[(1.0, 1.0)],
    test_size='n_species',
), **kwargs})
def train_cv      (**kwargs): return _train_cv(**{**dict(
    # NOTE For feat, tune override_scheduler in Projection.transform
    # NOTE For mem safety, tune n_jobs in Search._make_classifier -> OneVsRestClassifier
    logreg_Cs=[  # Order fast to slow
        # .0001,
        # .0003,
        # .001,
        # .003,
        # .01,
        # .03,
        # .1,
    ],
    n_jobs=1,  # As per "Use n_jobs=1" below
), **kwargs})
def load_for_eval (**kwargs): return _train_cv(**{**dict(
    logreg_Cs=[  # Order fast to slow
        .0001,
        .0003,
        .001,
        .003,
        .01,
        .03,
        .1,
        .3,
        1,
    ],
    n_jobs=1,  # Way faster for 100% loads
    skip_compute_if_missing=True,  # Partial report: load only the models that are done training
), **kwargs})
def _train_cv(
    logreg_Cs,
    n_jobs,
    n_species_n_recs=[
        (0.014, 1.00),  # sp[ 10], recs[1.0]
        (0.25,  1.00),  # sp[186], recs[1.0]
        (1.00,  1.00),  # sp[743], recs[1.0]
    ],
    skip_compute_if_missing=False,
    test_size=.2,
):

    ##
    # Params
    experiment_id_prefix = 'train-na-us-v0'
    # countries_k, com_names_k = 'am', 'cr_ebird'  # 9.4k/400k -> 4.0k/204k -> 754/35k
    # countries_k, com_names_k = 'na', 'us_ebird'  # 9.4k/400k -> 1.1k/ 60k -> 762/53k
    countries_k, com_names_k = 'na', 'us'        # 9.4k/400k -> 1.1k/ 60k -> 774/53k
    # countries_k, com_names_k = 'na', 'ca_ebird'  # 9.4k/400k -> 1.1k/ 60k -> 551/46k
    # countries_k, com_names_k = 'na', 'ca'        # 9.4k/400k -> 1.1k/ 60k -> 334/35k
    # countries_k, com_names_k = 'na', 'dan170'    # 9.4k/400k -> 1.1k/ 60k -> 170/3.4k
    # countries_k, com_names_k = 'na', 'dan4'      # 9.4k/400k -> 1.1k/ 60k ->   4/2.2k

    ##
    # Subset for faster dev
    inf = np.inf
    recs_at_least, num_species, num_recs =   0, inf, inf
    # recs_at_least, num_species, num_recs = 100, 100, 100  # CA[334/35k -> 127/25k -> 100/21k -> 100/10k   -> 100/10k]
    # recs_at_least, num_species, num_recs =  50, 100, 100  # CA[334/35k -> 224/32k -> 100/16k -> 100/ 9.0k -> 100/ 9.0k]
    # recs_at_least, num_species, num_recs =   0, 100, 100  # CA[334/35k -> 334/35k -> 100/12k -> 100/ 6.8k ->  99/ 6.8k]
    # recs_at_least, num_species, num_recs =   0, 168, 100  # CA[334/34k -> 334/34k -> 168/18k -> 168/11k   -> 167/11k]   # (old 'all')
    # recs_at_least, num_species, num_recs =   0, 168,  20  # CA[334/34k -> 334/35k -> 168/18k -> 168/ 3.2k -> 167/ 3.2k] # (old 'recs')
    # recs_at_least, num_species, num_recs =  20, 168,  20  # CA[334/35k -> 296/35k -> 168/20k -> 168/ 3.4k -> 168/ 3.4k] # Familiar set
    # recs_at_least, num_species, num_recs =  20, 336,  20  # US[774/53k -> 528/51k -> 336/33k -> 336/ 6.7k -> 336/ 6.7k] # Scale species
    # recs_at_least, num_species, num_recs =  10, 168,  20  # CA[334/35k ->                                             ] # Cls imbalance
    # recs_at_least, num_species, num_recs =   0, 168,  20  # CA[334/35k ->                                             ] # Cls imbalance
    # recs_at_least, num_species, num_recs =   0, inf,  20  # dan170 for app_brainstorm_1
    # recs_at_least, num_species, num_recs =  20, 50,  100  # Faster dev
    # recs_at_least, num_species, num_recs =  20, 50,   50  # Faster dev
    # recs_at_least, num_species, num_recs =  20, 50,   20  # Faster dev
    # recs_at_least, num_species, num_recs =  20, 50,   10  # Faster dev
    # recs_at_least, num_species, num_recs =  10, 10,   10  # Faster dev

    ##
    # Load models
    load = Load()
    projection = Projection.load('peterson-v0-26bae1c', features=Features(load=load))

    ##
    # Select recs
    #   1. countries: Filter recs to these countries
    #   2. species: Filter recs to these species
    #   3. recs_at_least: Filter species to those with at least this many recs
    #   4. num_species: Sample this many of the species
    #   5. num_recs: Sample this many recs per species
    get_recs_stats = lambda df: dict(sp=df.species.nunique(), recs=len(df))
    puts_stats = lambda desc: partial(tap, f=lambda df: print('%-15s %12s (sp/recs)' % (desc, '%(sp)s/%(recs)s' % get_recs_stats(df))))
    xcs = (xc.metadata
        .pipe(puts_stats('all'))
        # 1. countries: Filter recs to these countries
        [lambda df: df.country.isin(constants.countries[countries_k])]
        .pipe(puts_stats('countries'))
        # 2. species: Filter recs to these species
        [lambda df: df.species.isin(com_names_to_species(*com_names[com_names_k]))]
        .pipe(puts_stats('species'))
        # Omit not-downloaded recs (should be few within the selected countries)
        [lambda df: df.downloaded]
        .pipe(puts_stats('(downloaded)'))
        # Remove empty cats for perf
        .pipe(df_remove_unused_categories)
        # 3. recs_at_least: Filter species to those with at least this many recs
        [lambda df: df.species.isin(df.species.value_counts()[lambda s: s >= recs_at_least].index)]
        .pipe(puts_stats('recs_at_least'))
        # 4. num_species: Sample this many of the species
        [lambda df: df.species.isin(df.species.drop_duplicates().pipe(lambda s: s.sample(n=min(len(s), num_species), random_state=0)))]
        .pipe(puts_stats('num_species'))
        # 5. num_recs: Sample this many recs per species
        #   - Remove empty cats else .groupby fails on empty groups
        .pipe(df_remove_unused_categories)
        .groupby('species').apply(lambda g: g.sample(n=min(len(g), num_recs), random_state=0))
        .pipe(puts_stats('num_recs'))
        # Drop species with <2 recs, else StratifiedShuffleSplit complains (e.g. 'TUVU')
        [lambda df: df.species.isin(df.species.value_counts()[lambda s: s >= 2].index)]
        .pipe(puts_stats('recs ≥ 2'))
        # Clean up for downstream
        .pipe(df_remove_unused_categories)
    )
    _recs_stats = get_recs_stats(xcs)
    recs_stats = ', '.join(['%s[%s]' % (k, v) for k, v in _recs_stats.items()])
    display(
        recs_stats,
        df_summary(xcs).T,
        xcs.sample(n=10, random_state=0).sort_values('species'),
    )

    ##
    xcs_paths = [
        ('xc', f'{data_dir}/xc/data/{row.species}/{row.id}/audio.mp3')
        for row in df_rows(xcs)
    ]
    display(
        f"{len(xcs_paths)}/{len(xcs)}",
        # xcs_paths[:2],
    )

    ##
    recs = load.recs(paths=xcs_paths)
    display(
        df_summary(recs).T,
        recs[:5],
    )

    ##
    # Fast-and-cheap version (<1s) of the plots below (~7s)
    display(recs
        .species_longhand.value_counts().sort_index()
        .reset_index().rename(columns={'index': 'species_longhand', 'species_longhand': 'num_recs'})
        .assign(num_recs=lambda df: df.num_recs.map(lambda n: '%s /%s' % ('•' * int(n / df.num_recs.max() * 60), df.num_recs.max())))
    )

    ## {skip: true}
    # # Num recs loaded for training + total num available [slow: ~7s]
    # display(recs
    #     .pipe(df_reverse_cat, 'species_longhand')
    #     .assign(recs_n=1).groupby(['species', 'species_longhand'])['recs_n'].sum().reset_index()
    #     .set_index('species')
    #     # [:100]
    #     .join(how='left', other=(xc.metadata
    #         .assign(total_n=1).groupby('species')['total_n'].sum().reset_index()
    #         .set_index('species')
    #     ))
    #     .reset_index()
    #     .pipe(ggplot)
    #     + aes(x='species_longhand')
    #     + geom_col(aes(y='total_n'), fill='darkgray')
    #     + geom_col(aes(y='recs_n'), fill=scale_color_cmap(mpl.cm.tab10).palette(0))
    #     # + geom_point(aes(y='total_n'), color='darkgray')
    #     # + geom_point(aes(y='recs_n'), color=scale_color_cmap(mpl.cm.tab10).palette(0))
    #     + expand_limits(y=0)
    #     + coord_flip()
    #     + theme(axis_text_y=element_text(size=4))
    #     + theme_figsize(width=18, aspect_ratio=3/2)
    #     + ggtitle('Num recs loaded for training + total num available')
    # )

    ## {skip: true}
    # Plot recs with duration [slow: ~7s]
    # display(recs
    #     .assign(species=lambda df: df.species_longhand)
    #     .assign(count=1)
    #     # 0-fill all (species, dataset) combinations to create empty placeholders for missing bars
    #     .pipe(lambda df: df.append(
    #         pd.DataFrame([
    #             dict(species=species, dataset=dataset, duration_s=0)
    #             for species in df.species.unique()
    #             for dataset in df.dataset.unique()
    #         ])
    #         .astype({'species': df.species.dtype})
    #     ))
    #     .groupby(['dataset', 'species'])[['count', 'duration_s']].sum().reset_index()
    #     # Order by species by taxo
    #     .pipe(df_reverse_cat, 'species')
    #     # Order by species by count
    #     # .pipe(df_ordered_cat, species=lambda df: df.sort_values('count').species)
    #     .pipe(pd.melt, id_vars=['dataset', 'species'], value_vars=['count', 'duration_s'])
    #     .pipe(df_remove_unused_categories)
    #     .pipe(ggplot, aes(x='species', y='value', fill='dataset', color='dataset'))
    #     + coord_flip()
    #     + geom_bar(stat='identity', position=position_dodge(), width=.8)
    #     + facet_wrap('variable', nrow=1, scales='free')
    #     + xlab('')
    #     + ylab('')
    #     + scale_fill_cmap_d(mpl_cmap_concat('tab20', 'tab20b', 'tab20c'))
    #     + scale_color_cmap_d(mpl_cmap_concat('tab20', 'tab20b', 'tab20c'))
    #     + theme(panel_spacing=2.5)
    #     + theme_figsize(width=18, aspect_ratio=4/1)
    #     + ggtitle(f'recs: Total (count, duration_s) per (species, dataset)')
    # )

    ##
    # Add .feat
    recs = projection.transform(recs)

    ##
    # GridSearchCV many models / model params
    #   - Order these roughly from most to least expensive, so that training runs fail fast (e.g. oom at the start, not the end)
    param_grid = list(unique_everseen(tqdm([
        dict(
            classifier=[classifier],
            n_species=[n_species if isinstance(n_species, int) else int(n_species * _recs_stats['sp'])],
            n_recs=[
                # n_recs if isinstance(n_recs, int) else int(n_recs * _recs_stats['recs']),
                # TODO Figure out how to represent this as an n instead of a frac again [TODO interaction with test_size=.2 ...]
                #   - TODO Also make clear in the plot descs when e.g. (recs[3360], n_species[33], n_recs[1.0]) -> n_recs[33*20 < 3360]
                n_recs,
            ],
        )
        for (n_species, n_recs) in [
            # Subset for learning curves
            #   - Biggest first, to fail fast
            *n_species_n_recs,
        ]
        for logreg_cls, logreg_solver in [
            # ('logreg_ovr', 'liblinear'),  # 1 core
            ('ovr-logreg_ovr', 'liblinear'),  # Multicore [HACK but careful with mem: >1gb per 100% cpu -- tune in model.py]
        ]
        for logreg_max_iter in (
            [None] if logreg_solver in ['liblinear'] else [
                # 800,
                3200,
                # 6400, 12800,
            ]
        )
        for logreg_C in [
            # 1/reg_strength: C=inf is no regularization, smaller C is more regularization (default: 1)
            *logreg_Cs,
        ]
        for logreg_tol in one([tols for solvers, tols in [
            # Default: 1e-4
            # (['liblinear'], ['.0000000000000001']),  # liblinear keeps going, but only at large C -> unstable coefs
            # (['sag'], [
            #     # '.01',  # Faster than .001 (~5x) and even more suboptimal. Slower than liblinear, which has ~optimal acc.
            #     # '.001',  # Faster than .0001 (~5x) and a little suboptimal. Maybe useful for quick prototyping.
            #     None,  # Default: .0001, optimal-ish acc
            # ]),
        ] if logreg_solver in solvers] or [[None]])
        for logreg_class_weight in [
            # None,  # TODO Disabled to reduce noise. Re-enable to ponder the benefits of balance (slight but looks positive)
            'balanced',  # No (visually) significant effect with class sizes 0-20 (see class_imbalance notebooks)
        ]
        # for sgdlog_cls in [
        #     # 'sgdlog',  # Acc is less good and more noisy than with std
        #     'std-sgdlog',
        # ]
        # for sgdlog_alpha in [
        #     # Multiplier for regularization, and influences learning_rate='optimal' (the default)
        #     # Default: .0001
        #     .0001, .001, .01,
        #     # '.00001', .0001, .001, .01, .1, 1, 10,
        #     # '.0000001', '.000001', '.00001', .0001, .001, .01, .1, 1, 10, 100, 1000,
        # ]
        # for sgdlog_tol in [
        #     None,  # Default: .001
        #     # .0001, '.00001',  # TODO No observed improvement (at small scale)
        # ]
        # for sgdlog_average in [
        #     None,
        #     # 'true',  # TODO No observed improvement (at small scale)
        # ]
        # for sgdlog_class_weight in [logreg_class_weight]
        for classifier in [

            # Logistic regression
            ','.join(x for x in [
                f'cls: {logreg_cls}',
                f'solver: {logreg_solver}',
                '' if logreg_max_iter is None else f'max_iter: {logreg_max_iter}',
                '' if logreg_tol is None else f'tol: {logreg_tol}',
                f'C: {logreg_C}',
                '' if logreg_class_weight is None else f'class_weight: {logreg_class_weight}',
            ] if x),

            # # Logistic regression via SGD(loss=log)
            # ','.join(x for x in [
            #     f'cls: {sgdlog_cls}',
            #     f'alpha: {sgdlog_alpha}',
            #     '' if not sgdlog_tol else f'tol: {sgdlog_tol}',
            #     '' if not sgdlog_average else f'average: {sgdlog_average}',
            #     '' if sgdlog_class_weight is None else f'class_weight: {sgdlog_class_weight}',
            # ] if x),

        ]
    ])))

    # TODO Increase cv to decrease variance in eval metrics (this made Alex extremely squirmy)
    cv = GridSearchCVCached(
        estimator=Search(projection=projection),
        param_grid=param_grid,
        refit=False,  # Don't spend time fitting cv.best_estimator_ at the end (default: True)
        # cv=3,  # [SP14] uses two-/three-fold CV [why?]
        # Stratified ensures that all classes have >0 instances in each split, which is statistically maybe fishy but avoids
        # breaking various code that merges the split results back together and assumes all splits' classes are the same
        cv=sk.model_selection.StratifiedShuffleSplit(
            n_splits=1,  # [for ~18/20 miss: ~19m, ~16g disk cache]
            # n_splits=2,
            # n_splits=3,
            # n_splits=5,
            # n_splits=10,
            # n_splits=20,  # Known good [>51m uncached, >25g disk cache]
            # n_splits=100,  # [?m runtime, ?g disk cache]
            # Set test_size, e.g. for train_cv vs. train_final
            #   - HACK test_size must be ≥n_classes, so interpret 'n_species' for train_final
            test_size=(
                test_size if test_size != 'n_species' else
                (_recs_stats['sp'] + .1) / _recs_stats['recs']  # (+.1 to avoid underflow)
            ),
            random_state=0,
        ),
        return_train_score=True,
        # return_estimator=True,  # Verrrry heavy, use extra_metrics instead
        # recompute_extra_metrics=True,  # Loads estimator.pkl (fast) but doesn't return it (huge)
        extra_metrics=dict(
            # [How to specify SearchEvals here without creating caching headaches?]
            #   - e.g. defs don't bust cache on code edit
            #   - And avoid thrashing cache every time we refactor SearchEvals
            classes='estimator.classes_',
            train_i='train',
            train_y='y_train',
            train_predict_proba='estimator.classifier_.predict_proba(X_train)',
            test_i='test',
            test_y='y_test',
            test_predict_proba='estimator.classifier_.predict_proba(X_test)',
            model_size='len(joblib_dumps(estimator))',
            model_stats='model_stats(estimator)',
            proc_stats='proc_stats',
        ),
        # verbose=10,  # O(models * n_splits) lines of outputs
        verbose=1,  # O(1) lines of outputs
        # Use n_jobs=1
        #   - All classifiers can parallelize .fit/.predict
        #   - Avoid memory contention
        n_jobs=n_jobs,  # Take as param
        # n_jobs=-1,
        # n_jobs=1,  # For %prun, and way faster for 100% loads
        # n_jobs=6,  # For exactly 6 models
        # n_jobs=8,  # For slow ovr serdes [why?]
        # n_jobs=16,  # For recomputing model_stats
        artifacts=dict(
            dir=f'{data_dir}/artifacts',  # TODO Iron out syncing across local/gs/remote
            save=True,
            reuse=f'{experiment_id_prefix}-{countries_k}-{com_names_k}',
            skip_compute_if_missing=skip_compute_if_missing,
        ),
    )
    with contextlib.ExitStack() as stack:
        # stack.enter_context(cache_control(refresh=True))  # Disk unsafe...
        stack.enter_context(cache_control(enabled=False))  # Disk safe
        # stack.enter_context(joblib.parallel_backend('threading'))  # Default: 'multiprocessing'
        # stack.enter_context(joblib.parallel_backend('sequential'))  # For %prun [FIXME Has no effect; why?]
        # stack.enter_context(log.context(level='info'))  # FIXME log.context no longer exists [wat]
        # stack.enter_context(log.context(level='debug'))
        X, y = Search.Xy(recs)
        cv.fit(X, y)

    return locals()
