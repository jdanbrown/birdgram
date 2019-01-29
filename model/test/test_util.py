import audiosegment
import numpy as np
import pydub
import pytest
from pytest import raises

from util import *
from util import _audio_id_simplify, _audio_id_simplify_ops


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
    b = audio_copy(a)
    # Copies are the same, structurally
    assert audio_eq(a, b)
    # Copies isolate mutation
    b.name = 'b-name'
    b.seg.frame_rate = 42
    assert a.name == 'a-name'
    assert a.seg.frame_rate == 44100
    assert not audio_eq(a, b)


def test_audio_replace():
    a = audio(random_state=0)
    a.name = 'a-name'
    b = audio_replace(a, name='b-name')
    assert a.seg == b.seg
    assert a.name == 'a-name'
    assert b.name == 'b-name'


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


def test_audio_id_to_mimetype():
    for id, mimetype in [
        ('cache/audio/recs/foo.wav',                  'audio/wav'),
        ('recs/foo.wav',                              'audio/wav'),
        ('recs/foo.mp3',                              'audio/mpeg'),
        ('recs/foo.mp4',                              'audio/mp4'),
        ('recs/foo.xxx',                              raises(Exception)),
        ('recs/foo.wav.resample(22050,1,16)',         raises(Exception)),
        ('recs/foo.wav.slice(1,2).enc(wav)',          'audio/wav'),
        ('recs/foo.mp3.slice(1,2).enc(wav)',          'audio/wav'),
        ('recs/foo.wav.slice(1,2).enc(mp3,64k)',      'audio/mpeg'),
        ('recs/foo.mp3.slice(1,2).enc(mp3,64k)',      'audio/mpeg'),
        ('recs/foo.mp3.slice(1,2).enc(mp4,aac,64k)',  'audio/mp4'),
    ]:
        if type(mimetype).__name__ == 'RaisesContext':
            with mimetype:
                audio_id_to_mimetype(id)
        else:
            assert audio_id_to_mimetype(id) == mimetype


def test_audio_id_add_ops():
    for [id, *ops], out_id in [

        # No simplifications
        (['path.wav'],                           'path.wav'),
        (['path.wav',             'f()'],        'cache/audio/path.wav.f()'),
        (['path.wav',             'f()', 'g()'], 'cache/audio/path.wav.f().g()'),
        (['cache/audio/path.wav'],               'cache/audio/path.wav'),
        (['cache/audio/path.wav', 'f()'],        'cache/audio/path.wav.f()'),
        (['cache/audio/path.wav', 'f()', 'g()'], 'cache/audio/path.wav.f().g()'),

        # Simplifications
        (['cache/audio/path.wav', 'enc(wav)'],             'cache/audio/path.wav'),
        (['cache/audio/path.wav', 'enc(wav)', 'enc(wav)'], 'cache/audio/path.wav'),

    ]:
        assert audio_id_add_ops(id, *ops) == out_id


def test_audio_id_split_ops():
    for id, ops in [
        ('path.wav',                     ['path', 'wav']),
        ('cache/audio/path.wav',         ['cache/audio/path', 'wav']),
        ('cache/audio/path.wav.f()',     ['cache/audio/path', 'wav', 'f()']),
        ('cache/audio/path.wav.f().g()', ['cache/audio/path', 'wav', 'f()', 'g()']),
    ]:
        assert audio_id_split_ops(id) == ops
        if path_is_contained_by(id, 'cache/audio'):
            assert audio_id_add_ops(*audio_id_split_ops(id)) == id  # TODO Add pytest-quickcheck and do more of this


def test_audio_id_simplify():
    for x in [

        # No simplifications
        'recs/foo.wav',
        'recs/foo.wav.enc(mp3)',
        'recs/foo.mp3.enc(wav)',
        'recs/foo.mp3.enc(mp3)',
        'recs/foo.mp3.enc(mp3,64k)',
        'recs/foo.wav.enc(mp3,64k).enc(mp3,128k)',
        'recs/foo.wav.enc(mp3).enc(mp4)',
        'recs/foo.wav.enc(mp4,aac,32k).enc(mp4,libfdk_aac,32k)',
        'recs/foo.wav.resample(22050,1,16).enc(wav)',
        'recs/foo.wav.slice(0,10000).spectro_denoise().slice(1000,5000)',

        # Single simplification
        ('recs/foo.wav.enc(wav)',                                          'recs/foo.wav'),
        ('recs/foo.mp3.enc(wav).enc(wav)',                                 'recs/foo.mp3.enc(wav)'),
        ('recs/foo.wav.enc(mp3,64k).enc(mp3,64k)',                         'recs/foo.wav.enc(mp3,64k)'),
        ('recs/foo.wav.enc(mp4,aac,32k).enc(mp4,aac,32k)',                 'recs/foo.wav.enc(mp4,aac,32k)'),
        ('recs/foo.wav.resample(22050,1,16).resample(22050,1,16)',         'recs/foo.wav.resample(22050,1,16)'),
        ('recs/foo.wav.slice(1000,5000).slice(1000,2000)',                 'recs/foo.wav.slice(2000,3000)'),
        ('recs/foo.wav.slice(1000,5000).slice(1000,9000)',                 'recs/foo.wav.slice(2000,5000)'),
        ('recs/foo.wav.slice(1000,5000).slice(9000,9000)',                 'recs/foo.wav.slice(5000,5000)'),
        ('recs/foo.wav.slice(0,10000).slice(1000,5000).spectro_denoise()', 'recs/foo.wav.slice(1000,5000).spectro_denoise()'),

        # Multiple simplifications: toy cases
        ('f().enc(x).enc(x).enc(x).g()', 'f().enc(x).g()'),  # In the middle
        ('f().enc(x).enc(x).enc(x)',     'f().enc(x)'),      # At the end
        ('enc(x).enc(x).enc(x).g()',     'enc(x).g()'),      # At the beginning (not a real-world case)
        ('enc(x).enc(x).enc(x)',         'enc(x)'),          # At the beginning and end (not a real-world case)

        # Multiple simplifications: real cases
        ('recs/foo.wav.enc(wav).enc(wav)', 'recs/foo.wav'),

    ]:
        (a, b) = (x, x) if isinstance(x, str) else x
        assert _audio_id_simplify_ops(audio_id_split_ops(a)) == audio_id_split_ops(b)
        assert _audio_id_simplify(a) == b
