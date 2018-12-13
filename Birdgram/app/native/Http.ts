// (See native/Spectro.ts)

import { EmitterSubscription, NativeEventEmitter, NativeModules } from 'react-native';

const {RNHttp} = NativeModules;

const _emitter = new NativeEventEmitter(RNHttp);

export const NativeHttp = {

  _emitter,

  httpFetch: async (
    url: string,
  ): Promise<string> => RNHttp.httpFetch(
    url,
  ),

};
