import os

from attrdict import AttrDict

role = os.environ.get('BUBO_ROLE')

# Global mutable config (handle with care)
config = AttrDict(

    role=role,

    hosts=dict(
        prod='35.230.68.91',
        local='192.168.0.195:8000',
    ),

    audio_to_url = dict(

        # TODO TODO WIP tuning (notebooks/audio_codecs_data_volume)
        audio_kwargs=dict(
            # format='wav',
            # format='mp3', bitrate='32k',
            # format='mp4', bitrate='32k', codec='aac',
            format='mp4', bitrate='32k', codec='libfdk_aac',
        ),

        url_type={
            # Tradeoffs:
            #   - notebook: Files are way faster (~instant) and more lightweight (~0 mem) than inline data urls for #
            #     displaying many audios at once (>>10)
            #   - api: Data urls don't require serving the resource
            'api':      'data',
            'notebook': 'file',
        }[role or 'notebook'],

    ),

)
