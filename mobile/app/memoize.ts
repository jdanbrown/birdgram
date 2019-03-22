// https://github.com/alexreardon/memoize-one

import _ from 'lodash';
import memoizeOne from 'memoize-one';

export { memoizeOne };

// memoizeOne uses shallow equality (===) by default
export function memoizeOneDeep<X extends (...args: any[]) => any>(f: X): X {
  return memoizeOne(f, (x, y, i) => _.isEqual(x, y));
}
