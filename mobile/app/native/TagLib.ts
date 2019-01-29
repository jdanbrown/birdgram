// (See native/Spectro.ts)

import { EmitterSubscription, NativeEventEmitter, NativeModules } from 'react-native';

const {RNTagLib} = NativeModules;

const _emitter = new NativeEventEmitter(RNTagLib);

export const NativeTagLib = {

  _emitter,

  readComment: async (
    audioPath: string,
  ): Promise<null | string> => RNTagLib.readComment(
    audioPath,
  ),

  writeComment: async (
    audioPath: string,
    value: string,
  ): Promise<void> => RNTagLib.writeComment(
    audioPath,
    value,
  ),

  audioFiletype: async (
    audioPath: string,
  ): Promise<string> => RNTagLib.audioFiletype(
    audioPath,
  ),

};
