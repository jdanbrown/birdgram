// http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12
export const Paired = {
  [Symbol.iterator]: function*() { for (let v of Object.values(this)) yield v; }, // TODO Factor out [how to type?]
  lightBlue:   '#a6cee3',
  darkBlue:    '#1f78b4',
  lightGreen:  '#b2df8a',
  darkGreen:   '#33a02c',
  lightRed:    '#fb9a99',
  darkRed:     '#e31a1c',
  lightOrange: '#fdbf6f',
  darkOrange:  '#ff7f00',
  lightPurple: '#cab2d6',
  darkPurple:  '#6a3d9a',
  lightYellow: '#ffff99',
  darkYellow:  '#b15928',
};

// http://colorbrewer2.org/#type=qualitative&scheme=Set1&n=9
export const Set1 = {
  [Symbol.iterator]: function*() { for (let v of Object.values(this)) yield v; }, // TODO Factor out [how to type?]
  red:    '#e41a1c',
  blue:   '#377eb8',
  green:  '#4daf4a',
  purple: '#984ea3',
  orange: '#ff7f00',
  yellow: '#ffff33',
  brown:  '#a65628',
  pink:   '#f781bf',
  gray:   '#999999',
};

// TODO More (e.g. tab10, tab20)
//  - https://matplotlib.org/examples/color/colormaps_reference.html
