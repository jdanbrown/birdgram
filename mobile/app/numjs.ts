// http://nicolaspanel.github.io/numjs/

import nj from '../third-party/numjs/dist/numjs.min';
export { nj };

// Mimic np.linalg.norm
//  - https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
export function nj_norm(
  x: nj.NjArray<number>,
  n: number = 2,
): number {
  return nj.array(x).pow(n).sum() ** (1/n);
}
