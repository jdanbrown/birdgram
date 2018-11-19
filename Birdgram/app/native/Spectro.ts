import { NativeModules } from 'react-native';

const {RNSpectro} = NativeModules;

export default {
  foo: async (x: string, y: string, z: number): Promise<string> => RNSpectro.foo(x, y, z),
};
