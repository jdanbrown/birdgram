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

const _emitter = new NativeEventEmitter(RNSpectro);

export interface NativeSpectroStats {
  nPathsSent: number;
}

export interface ImageFile {
  path: string;
  width: number;
  height: number;
}

export const NativeSpectro = {

  // XXX Debug
  debugPrintNative: (msg: string): void => RNSpectro.debugPrintNative(msg),

  // constantsToExport
  //  - e.g. `foo: RNSpectro.foo as number,`

  _emitter,

  create: async (
    opts: {
      outputPath: string;
      f_bins: number;
      sampleRate?: number;
      bitsPerChannel?: number;
      channelsPerFrame?: number;
      refreshRate?: number;
      bufferSize?: number;
    },
  ): Promise<void> => RNSpectro.create(
    opts,
  ),

  start: async (): Promise<void>               => RNSpectro.start(),
  stop:  async (): Promise<null | string>      => RNSpectro.stop(),
  stats: async (): Promise<NativeSpectroStats> => RNSpectro.stats(),

  onAudioChunk:      (f: (...args: any[]) => any): EmitterSubscription => _emitter.addListener('audioChunk', f),
  onSpectroFilePath: (f: (...args: any[]) => any): EmitterSubscription => _emitter.addListener('spectroFilePath', f),

  renderAudioPathToSpectroPath: async (
    audioPath: string,
    spectroPathBase: string,
    opts: {
      f_bins: number,
      denoise?: boolean,
    },
  ): Promise<null | ImageFile> => RNSpectro.renderAudioPathToSpectroPath(
    audioPath,
    spectroPathBase,
    opts,
  ),

  // TODO Move out of NativeSpectro
  chunkImageFile: async (
    path: string,
    chunkWidth: number
  ): Promise<Array<ImageFile>> => RNSpectro.chunkImageFile(
    path,
    chunkWidth
  ),

};
