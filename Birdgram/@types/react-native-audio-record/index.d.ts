declare module 'react-native-audio-record' {

  export interface Options {
    sampleRate?: number;
    bitsPerSample?: number;
    channels?: number;
    wavFile?: string;
  }

  export function init(options: Options): void;
  export function start(): void;
  export function stop(): Promise<string>;
  export function on(event: string, callback: (data: string) => void): void;

}
