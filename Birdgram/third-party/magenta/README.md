# @magenta/music
- https://github.com/tensorflow/magenta-js/tree/master/music

# Why do we have to vendor?
- Errors on mobile: "No backend found in registry"
  - Looks like tflite is the intended way i/o tfjs...
- All we're after is audio_utils.ts, so we vendor that module in isolation from the
