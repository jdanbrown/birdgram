from ggplot import *
ggplot(aes(x='date', y='beef'), data=meat) + geom_point()

ggplot(pd.DataFrame([{'bias':bias, 'layer':layer} for layer in ['conv%s' % i for i in range(1,6)] for bias in net.params[layer][1].data]), aes(x='layer',y='bias')) + geom_boxplot()

@singleton
class _:
    from ggplot import *
    print \
    (ggplot(diamonds, aes(x='carat')) +
        geom_histogram() +
        #facet_wrap(x='cut') +               # Good: ylim applies to all facets
        facet_wrap(x='cut', scales='free') + # Bad: ylim only applies to last facet
        geom_blank()
    )
