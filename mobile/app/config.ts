import env from 'react-native-config'; // Loaded from .env.* file [https://github.com/luggit/react-native-config]

// TODO config -> Settings.state [how to share globally?]
export const config = {

  // Add keys from .env (file)
  //  - .env keys should all be uppercase by convention (like env vars)
  env,

  host: 'local',
  // host: 'remote',

  baseUris: {
    'local':  'http://192.168.0.195:8000',
    'remote': 'http://35.230.68.91',
  },

  // Animated/Gesture
  useNativeDriver: true,
  // useNativeDriver: false, // FIXME Buggy / exposes bugs in our gross gestures/animated logic (ugh)

};
