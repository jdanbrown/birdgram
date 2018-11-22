// Based on:
//  - https://github.com/goodatlas/react-native-audio-record
//  - https://github.com/chadsmith/react-native-microphone-stream

import { EmitterSubscription, NativeEventEmitter, NativeModules } from 'react-native';

const {RNSpectro} = NativeModules;

export interface SpectroOpts {
  outputFile: string;
  sampleRate?: number;
  bitsPerChannel?: number;
  channelsPerFrame?: number;
  bufferSizeBytes?: number;
}

const _emitter = new NativeEventEmitter(RNSpectro);

export const Spectro = {

  _emitter,

  setup: async (opts: SpectroOpts): Promise<void> => RNSpectro.setup(opts),
  start: async (): Promise<void>                  => RNSpectro.start(),
  stop:  async (): Promise<string | null>         => RNSpectro.stop(),
  stats: async (): Promise<object>                => RNSpectro.stats(),

  addListener: (f: (...args: any[]) => any): EmitterSubscription => _emitter.addListener('audioChunk', f),

  // XXX Dev
  // hello: async (x: string, y: string, z: number): Promise<string> => RNSpectro.hello(x, y, z),

};
