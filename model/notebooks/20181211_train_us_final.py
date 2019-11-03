#!/usr/bin/env python

from notebooks._20181211_train_us import *
train_final()

# FIXME ^C during audio_metadata causes lots of empty outputs to be cached (b/c it makes ffmpeg fail as if it had a junk input file)
#   - HACK To clean up manually: https://gist.github.com/jdanbrown/11f082549c9cb6858b64977e88e7fde9
