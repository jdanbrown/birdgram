import pytest

from attrdict import AttrDict
from load import *
from util import *


def test_transcode_audio():
    # Currently have to test with _transcode_audio(dry_run=True, unsafe_fs=True) since we don't yet have a mock fs
    load_defaults = AttrDict(
        sample_rate=22050, channels=1, sample_width_bit=16,
    )
    audio_defaults = AttrDict(
        frame_rate=load_defaults.sample_rate,
        channels=load_defaults.channels,
        sample_width=load_defaults.sample_width_bit // 8,
    )
    for x in [

        # Would noop: .wav -> .wav with same (hz,ch,bit)
        dict(
            would_noop=True,
            audio=dict(name='recs/foo.wav'),
            load=dict(format='wav'),
        ),

        # Wouldn't noop: .wav -> .wav with different (hz,ch,bit)
        dict(
            would_noop=False,
            audio=dict(name='recs/foo.wav', frame_rate=44100),
            load=dict(format='wav'),
        ),

        # Wouldn't noop: .mp3 -> .wav
        dict(
            would_noop=False,
            audio=dict(name='recs/foo.mp3'),
            load=dict(format='wav'),
        ),

        # Wouldn't noop: .wav -> .mp3
        dict(
            would_noop=False,
            audio=dict(name='recs/foo.wav'),
            load=dict(format='mp3', bitrate='64k'),
        ),

        # Wouldn't noop: .mp3 -> .mp3
        #   - Because we don't attempt to detect input bitrate/codec/vbr/etc.
        dict(
            would_noop=False,
            audio=dict(name='recs/foo.mp3'),
            load=dict(format='mp3', bitrate='64k'),
        ),

    ]:
        load = Load(**{**load_defaults, **x['load']})
        audio = AttrDict(**{**audio_defaults, **x['audio']})
        (would_noop, audio_id) = load._transcode_audio(audio, dry_run=True, unsafe_fs=True)
        assert x['would_noop'] == would_noop
