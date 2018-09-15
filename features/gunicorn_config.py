# gunicorn settings
#   - http://docs.gunicorn.org/en/stable/settings.html

# logging
from api.app import load_logging_dict
logconfig_dict = load_logging_dict()
