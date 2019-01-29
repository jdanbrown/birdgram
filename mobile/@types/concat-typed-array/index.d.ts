declare module 'concat-typed-array' {

  export default function concatTypedArray<X>(
    resultConstructor: new (n: number) => X,
    ...arrays: X[],
  ): X;

}
