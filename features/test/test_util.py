import audiosegment
import numpy as np
import pydub
import pytest

from util import *


@pytest.fixture
def audio(
    random_state=0,
    duration_s=10.8,
    framerate=44100,
):
    return audio_replace(
        audiosegment.from_numpy_array(
            np.random.RandomState(random_state).randint(low=2**15, size=int(framerate * duration_s), dtype=np.int16),
            framerate=framerate,
        ),
        name=f'test-audio-{random_state}',
    )


def test_audio_eq():
    a = audio(random_state=0)
    b = audio(random_state=1)
    c = audio(random_state=0); c.name = a.name + '-different'
    d = audio(random_state=0); d.seg._data = a.seg._data * 2
    assert audio_eq(a, a)
    assert audio_eq(b, b)
    assert audio_eq(c, c)
    assert audio_eq(d, d)
    assert not audio_eq(a, b)
    assert not audio_eq(a, c)
    assert not audio_eq(a, d)


def test_audio_copy():
    a = audio(random_state=0)
    a.name = 'a-name'
    a.seg.frame_rate = 44100
    a.nonstandard_attr = 'a-nonstandard_attr'
    b = audio_copy(a)
    # Copies are the same, structurally
    assert audio_eq(a, b)
    # Copies isolate mutation
    b.name = 'b-name'
    b.seg.frame_rate = 42
    assert a.name == 'a-name'
    assert a.seg.frame_rate == 44100
    assert not audio_eq(a, b)
    # Copies preserve nonstandard attrs
    assert b.nonstandard_attr == 'a-nonstandard_attr'


def test_audio_replace():
    a = audio(random_state=0)
    a.name = 'a-name'
    a.path = 'a-path'
    b = audio_replace(a,
        name='b-name',  # Standard attr
        path='b-path',  # Nonstandard attr
    )
    assert a.seg == b.seg
    assert (a.name, a.path) == ('a-name', 'a-path')
    assert (b.name, b.path) == ('b-name', 'b-path')


def test_audio_concat_doesnt_mutate():
    a0 = audio(random_state=0)
    a = audio(random_state=0)
    b = audio(random_state=1)
    _ = audio_concat([a, b])
    assert audio_eq(a0, a)
    _ = audio_concat([a])
    assert audio_eq(a0, a)


def test_audio_concat_preserves_metadata():
    a = audio(random_state=0)
    b = audio(random_state=1)
    c = audio_concat([a, b])
    assert c.name == f'{a.name} + {b.name}'
    assert c.sample_width == a.sample_width
    assert c.frame_rate == a.frame_rate
    assert c.channels == a.channels
    assert c.seg._data == a.seg._data + b.seg._data


def test_audio_concat_is_associative():
    a = audio(random_state=0)
    b = audio(random_state=1)
    c = audio(random_state=2)
    x = audio_concat([a, b, c])
    y = audio_concat([audio_concat([a, b]), c])
    z = audio_concat([a, audio_concat([b, c])])
    assert audio_eq(x, y)
    assert audio_eq(x, z)
    assert audio_eq(y, z)


def test_audio_concat_has_zero():
    a = audio(random_state=0)
    z = a[:0]
    assert audio_eq(a, audio_concat([a, z]))
    assert audio_eq(a, audio_concat([z, a]))


def test_audio_hash(audio):
    expects = {
        '8d2ca5441c370a40f4d3197539b8ddbeec568a11': [
            audio,
            audio_copy(audio),
            audio_replace(audio, name=audio.name),
            audio_replace(audio, seg=audio.seg),
            audio_replace(audio, name=audio.name, seg=audio.seg),
            audio_concat([audio]),
            audio[:],
            audio[0:],
            audio[:len(audio)],
            audio[0:len(audio)],
        ],
        'a2b1f01f0b112acbb77507fb212761991821f41b': [
            audio[1000:3000],
            audio_concat([audio[1000:2000], audio[2000:3000]], name=audio.name),
        ],
        '1c746fc531070be8e68fb777a3309b864292df14': [
            audio_replace(name='', audio=audio[1000:3000]),
            audio_concat(name='', audios=[audio[1000:2000], audio[2000:3000]]),
        ],
        '609a55da1cb7fc65cc6593dcbcfe6c843d76e502': [
            audio_replace(audio, name='foo'),
        ],
        '87c054f336a7dad0aef0bdfb7ca23b70037878b5': [
            audio_replace(name='', audio=audio_pad(audio, 3)),
            audio_replace(name='', audio=audio_pad(audio_pad(audio, 3), 0)),
            audio_replace(name='', audio=audio_pad(audio_pad(audio, 0), 3)),
            audio_replace(name='', audio=audio_pad(audio_pad(audio, 2), 1)),
            audio_replace(name='', audio=audio_pad(audio_pad(audio, 1), 2)),
        ],
    }
    assert [
        (k, k)
        for k, audios in expects.items()
        for a in audios
    ] == [
        (k, audio_hash(a))
        for k, audios in expects.items()
        for a in audios
    ]
