// To expose a method from swift to js:
//  - app/native/Spectro.ts      - add js Spectro.f() calling objc NativeModules.RNSpectro.f()
//  - ios/Birdgram/Spectro.m     - add objc extern for swift RNSpectro.f()
//  - ios/Birdgram/Spectro.swift - add swift RNSpectro.f() calling Spectro.f()
//  - ios/Birdgram/Spectro.swift - add swift Spectro.f()

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
  refreshRate?: number;
  bufferSize?: number;
}

const _emitter = new NativeEventEmitter(RNSpectro);

export const Spectro = {

  _emitter,

  setup: async (opts: SpectroOpts): Promise<void> => RNSpectro.setup(opts),
  start: async (): Promise<void>                  => RNSpectro.start(),
  stop:  async (): Promise<string | null>         => RNSpectro.stop(),
  stats: async (): Promise<object>                => RNSpectro.stats(),

  onAudioChunk:      (f: (...args: any[]) => any): EmitterSubscription => _emitter.addListener('audioChunk', f),
  onSpectroFilePath: (f: (...args: any[]) => any): EmitterSubscription => _emitter.addListener('spectroFilePath', f),

};
