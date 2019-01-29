// https://github.com/fractaltech/react-native-timer
declare module 'react-native-timer' {

  interface Timer {

    // Without context

    setTimeout(name: string, fn: () => void, interval: number): void;
    clearTimeout(name: string): void;
    timeoutExists(name: string): void;

    setInterval(name: string, fn: () => void, interval: number): void;
    clearInterval(name: string): void;
    intervalExists(name: string): void;

    setImmediate(name: string, fn: () => void): void;
    clearImmediate(name: string): void;
    immediateExists(name: string): void;

    requestAnimationFrame(name: string, fn: () => void): void;
    cancelAnimationFrame(name: string): void;
    animationFrameExists(name: string): void;

    // With context

    setTimeout(context: any, name: string, fn: () => void, interval: number): void;
    clearTimeout(context: any, name: string): void;
    clearTimeout(context: any): void;
    timeoutExists(context: any, name: string): void;

    setInterval(context: any, name: string, fn: () => void, interval: number): void;
    clearInterval(context: any, name: string): void;
    clearInterval(context: any): void;
    intervalExists(context: any, name: string): void;

    setImmediate(context: any, name: string, fn: () => void): void;
    clearImmediate(context: any, name: string): void;
    clearImmediate(context: any): void;
    immediateExists(context: any, name: string): void;

    requestAnimationFrame(context: any, name: string, fn: () => void): void;
    cancelAnimationFrame(context: any, name: string): void;
    cancelAnimationFrame(context: any): void;
    animationFrameExists(context: any, name: string): void;

  }

  const timer: Timer;
  export default timer;

}
