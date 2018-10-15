import Chance from 'chance';

// Export global:any, which would have otherwise come from DOM but we disable DOM for react-native (tsconfig -> "lib")
// @ts-ignore
export const global: any = window.global;

// Instantiate a global Chance
export const chance = new Chance();
