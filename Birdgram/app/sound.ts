import _Sound from 'react-native-sound';

export default class Sound extends _Sound {

  // async variant of `new Sound`
  static newAsync = (filename: string, basePath: string): Promise<Sound> => {
    return new Promise((resolve, reject) => {
      const sound: Sound = new Sound(
        filename,
        basePath,
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

}
