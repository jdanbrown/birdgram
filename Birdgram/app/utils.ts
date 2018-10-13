// Export global:any, which would have otherwise come from DOM but we disable DOM for react-native (tsconfig -> "lib")
// @ts-ignore
export const global: any = window.global;
