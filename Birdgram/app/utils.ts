import Chance from 'chance';

// Export global:any, which would have otherwise come from DOM but we disable DOM for react-native (tsconfig -> "lib")
// @ts-ignore
export const global: any = window.global;

// Instantiate a global Chance
export const chance = new Chance();

//
// Utils
//

// `X0 extends X` so that x0 can't (quietly) generalize the type of the case patterns (e.g. to include null)
//  - e.g. fail on `match(X | null, ...)` if the case patterns don't include null
export const match = <X, X0 extends X, Y>(x0: X0, ...cases: [X, Y][]): Y => {
  for (let [x, y] of cases) {
    if (x0 === x) return y;
  }
  throw new Error(`No cases matched: ${x0} not in [${cases.map(([x, y]) => x)}]`);
};
