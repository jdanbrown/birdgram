# TODO Is this helpful?
# """
# Warm various caches at build time to speed up app startup at runtime
# """
#
# from log import log
#
# log.info('Running: buildtime_warm_caches')
#
# for stmt in [
#
#     # All the imports are slow the first time (e.g. ~48s cold vs. ~5s warm)
#     #   - TODO Figure out whether this creates file state within the docker image, or just warms some caches within the vm
#     #     running docker
#     # 'from notebooks import *',  # TODO Requires data/**/metadata
#     'import util'  # XXX After we can import notebooks
#
#     # Populate disk caches for metadata
#     # 'metadata.species.df',  # TODO Requires data/metadata/
#     # 'xc.metadata',          # TODO Requires data/xc/metadata/
#
# ]:
#     log.info(f'Running stmt[{stmt}]')
#     exec(stmt)
#
# log.info('Done: buildtime_warm_caches')
