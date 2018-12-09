// (See native/Spectro.ts)

import { EmitterSubscription, NativeEventEmitter, NativeModules } from 'react-native';

import { ModelsSearch } from '../datatypes';

const {RNSearch} = NativeModules;

const _emitter = new NativeEventEmitter(RNSearch);

export const NativeSearch = {

  _emitter,

  create: async (
    modelsSearch: ModelsSearch,
  ): Promise<void> => RNSearch.create(
    modelsSearch,
  ),

  f_preds: async (
    audioPath: string,
  ): Promise<null | Array<number>> => RNSearch.f_preds(
    audioPath,
  ),

};
