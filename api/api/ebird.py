import pandas as pd
import re
import requests
import structlog

from api.request import cached_request
from api.util import pp


log = structlog.get_logger(__name__)


def barchart():

    # Cache key:
    #   - (lat,lon) -> k nearest hotspots -> stable cache key per outing
    #   - TODO Add day to cache key
    rep = cached_request('GET', 'http://ebird.org/ebird/BarChart', params = dict(
        cmd          = 'getChart',
        displayType  = 'download',
        getLocations = 'hotspots',
        hotspots     = 'L389606,L3186718,L1333065,L1020070',
        bYear        = '1900',
        eYear        = '2016',
        bMonth       = '1',
        eMonth       = '12',
        reportType   = 'location',
        parentState  = 'US-CA',
    ))
    lines = rep.text.splitlines()

    # Very weird format; warn if it doesn't look right but try to parse anyway
    for line_i, expected_line_pattern in [
        (0, ''),
        (1, ''),
        (2, ''),
        (3, ''),
        (4, ''),
        (5, ''),
        (6, ''),
        (7, ''),
        (8, ''),
        (9, 'Frequency of observations in the selected location\(s\).:.*'),
        (10, 'Number of taxa: .*'),
        (11, ''),
        (12, '\tJan\t\t\t\tFeb\t\t\t\tMar\t\t\t\tApr\t\t\t\tMay\t\t\t\tJun\t\t\t\tJul\t\t\t\tAug\t\t\t\tSep\t\t\t\tOct\t\t\t\tNov\t\t\t\tDec\t\t\t\t'),
        (13, 'Sample Size:.*'),
        (-1, ''),
    ]:
        if not re.match('^%s$' % expected_line_pattern, lines[line_i]):
            log.warn('barchart_parse_fail',
                line_i                = line_i,
                line                  = lines[line_i],
                expected_line_pattern = expected_line_pattern,
            )

    rows = [x.rstrip('\t').split('\t') for x in lines]

    # http://help.ebird.org/customer/portal/articles/1010553-understanding-the-ebird-bar-charts
    #   "Since most months do not divide equally into 7-day periods, any remaining days are added to
    #   the last period. Thus, while the first, second, and third periods are all seven days, the
    #   final period ranges from seven to ten days, depending on if the month has 28, 29, 30, or 31
    #   days. The fact that the final period is consistently longer does not seem to bias the
    #   results strongly, but please do keep this in mind as you explore data using this tool."
    pseudo_weeks = [
        x
        for xs in [
            [x+'/1', x+'/7', x+'/14', x+'/21']
            for x in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ]
        for x in xs
    ]

    sample_sizes = pd.DataFrame(
        columns = pseudo_weeks,
        data    = [[float(x) for x in rows[13][1:]]],
    )

    species = pd.DataFrame(
        columns  = ['species_cn'] + pseudo_weeks,
        data     = [[row[0]] + [float(x) for x in row[1:]] for row in rows[15:-1]],
    )

    return dict(
        sample_sizes = sample_sizes,
        species      = species,
    )


if __name__ == '__main__':
    import jdanbrown.pandas; jdanbrown.pandas.set_display()
    pp(barchart())
