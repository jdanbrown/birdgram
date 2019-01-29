import pytest

import metadata


def test_species_getitem_query():
    for query in ['SOSP', 'sosp', 'song sparrow', 'melospiza melodia']:
        assert metadata.species[query] == metadata.species['sonspa']
    for query in ['WIWA', 'wiwa', "wilson's warbler"]:
        assert metadata.species[query] == metadata.species['wlswar']
    for query in ['does-not-exist', '']:
        assert metadata.species[query] is None


def test_species_getitem_query_key():
    metadata.species['sonspa', 'species_code'] == 'sonspa'
    metadata.species['sonspa', 'shorthand'] == 'SOSP'
    metadata.species['sonspa', 'com_name'] == 'Song Sparrow'
    metadata.species['sonspa', 'sci_name'] == 'Melospiza Melodia'
    metadata.species['does-not-exist', 'sci_name'] is None
