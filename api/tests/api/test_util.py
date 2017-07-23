from api.util import snakecase


def test_snakecase():
    tests = {
        'sciName': 'sci_name',
        'SciName': 'sci_name',
        'sci_name': 'sci_name',
        'subnational1Code': 'subnational1_code',
        'Sci-Name': 'sci_name',  # Not 'sci__name'
        'locID': 'loc_id',  # Not 'loc_i_d'
        'HTTPResponse': 'http_response',  # Not 'h_t_t_p_response'
        'SomeHTTPResponse': 'some_http_response',
    }
    assert {k: snakecase(k) for k, v in tests.items()} == tests
