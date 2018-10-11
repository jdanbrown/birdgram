declare module 'react-native-microphone-stream' {

  import { EmitterSubscription } from 'react-native';

  export interface Options {
    bufferSize?:       number; // ios | android
    sampleRate?:       number; // ios | android
    bitsPerChannel?:   number; // ios | android
    channelsPerFrame?: number; // ios | android
    framesPerPacket?:  number; // ios
    bytesPerFrame?:    number; // ios
    bytesPerPacket?:   number; // ios
  }

  export function init(options: Options): void;
  export function start(): void;
  export function pause(): void;
  export function stop(): void;
  export function addListener(listener: (...args: any[]) => any, context?: any): EmitterSubscription;

}
