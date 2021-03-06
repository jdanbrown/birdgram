import _Sound from 'react-native-sound';

import { finallyAsync } from 'app/utils';

export default class Sound extends _Sound {

  // Expose some useful private props as public
  getFilename = (): string => {
    // @ts-ignore (Not exposed)
    return this._filename;
  }

  // async variant of `new Sound`
  static newAsync = (filename: string, basePath?: string): Promise<Sound> => {
    return new Promise((resolve, reject) => {
      const sound: Sound = new Sound(
        filename,
        basePath || '',
        error => error ? reject(error) : resolve(sound),
      );
    });
  }

  // async variants
  //  - (These never reject() because the underlying api provides no error signal)
  playAsync  = (): Promise<void> => new Promise((resolve, reject) => this.play  (() => resolve()))
  pauseAsync = (): Promise<void> => new Promise((resolve, reject) => this.pause (() => resolve()))
  stopAsync  = (): Promise<void> => new Promise((resolve, reject) => this.stop  (() => resolve()))
  getCurrentTimeAsync = (): Promise<{seconds: number, isPlaying: boolean}> => new Promise((resolve, reject) => {
    this.getCurrentTime((seconds, isPlaying) => resolve({seconds, isPlaying}));
  });

  // Allocate and release a Sound for scoped usage
  static scoped = <X>(filename: string, basePath?: string) => async (f: (sound: Sound) => Promise<X>): Promise<X> => {
    const sound = await Sound.newAsync(filename, basePath);
    const xAsync = (async () => f(sound))(); // Map sync exceptions to promise failure
    return await finallyAsync(xAsync, async () => {
      sound.release();
    });
  }

}
