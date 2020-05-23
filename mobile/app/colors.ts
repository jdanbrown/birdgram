// Pastel1
`
[print(f'  {name}: {hex!r},') for hex, name in zip_longest(
    mpl_cmap_colors_to_hex('Pastel1'),
    [stringcase.camelcase('_'.join(xs)).replace('_', '') for xs in product(
        ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'brown', 'pink', 'gray'],
    )],
)];
`
export const Pastel1 = {
  [Symbol.iterator]: function*() { for (let v of Object.values(this)) yield v; }, // TODO Factor out [how to type?]
  red: '#fbb4ae',
  blue: '#b3cde3',
  green: '#ccebc5',
  purple: '#decbe4',
  orange: '#fed9a6',
  yellow: '#ffffcc',
  brown: '#e5d8bd',
  pink: '#fddaec',
  gray: '#f2f2f2',
};

// Pastel2
`
[print(f'  {name}: {hex!r},') for hex, name in zip_longest(
    mpl_cmap_colors_to_hex('Pastel2'),
    [stringcase.camelcase('_'.join(xs)).replace('_', '') for xs in product(
        ['green', 'orange', 'blue', 'purple', 'lime', 'yellow', 'brown', 'gray'],
    )],
)];
`
export const Pastel2 = {
  [Symbol.iterator]: function*() { for (let v of Object.values(this)) yield v; }, // TODO Factor out [how to type?]
  green: '#b3e2cd',
  orange: '#fdcdac',
  blue: '#cbd5e8',
  purple: '#f4cae4',
  lime: '#e6f5c9',
  yellow: '#fff2ae',
  brown: '#f1e2cc',
  gray: '#cccccc',
};

// Paired
`
[print(f'  {name}: {hex!r},') for hex, name in zip_longest(
    mpl_cmap_colors_to_hex('Paired'),
    [stringcase.camelcase('_'.join(reversed(xs))) for xs in product(
        ['blue', 'green', 'red', 'orange', 'purple', 'brown'],
        ['light', 'dark'],
    )],
)];
`
export const Paired = {
  [Symbol.iterator]: function*() { for (let v of Object.values(this)) yield v; }, // TODO Factor out [how to type?]
  lightBlue: '#a6cee3',
  darkBlue: '#1f78b4',
  lightGreen: '#b2df8a',
  darkGreen: '#33a02c',
  lightRed: '#fb9a99',
  darkRed: '#e31a1c',
  lightOrange: '#fdbf6f',
  darkOrange: '#ff7f00',
  lightPurple: '#cab2d6',
  darkPurple: '#6a3d9a',
  lightBrown: '#ffff99',
  darkBrown: '#b15928',
};

// Accent
`
[print(f'  {name}: {hex!r},') for hex, name in zip_longest(
    mpl_cmap_colors_to_hex('Accent'),
    [stringcase.camelcase('_'.join(xs)).replace('_', '') for xs in product(
        ['green', 'purple', 'orange', 'yellow', 'blue', 'red', 'brown', 'gray'],
    )],
)];
`
export const Accent = {
  [Symbol.iterator]: function*() { for (let v of Object.values(this)) yield v; }, // TODO Factor out [how to type?]
  green: '#7fc97f',
  purple: '#beaed4',
  orange: '#fdc086',
  yellow: '#ffff99',
  blue: '#386cb0',
  red: '#f0027f',
  brown: '#bf5b17',
  gray: '#666666',
};

// Dark2
`
[print(f'  {name}: {hex!r},') for hex, name in zip_longest(
    mpl_cmap_colors_to_hex('Dark2'),
    [stringcase.camelcase('_'.join(xs)).replace('_', '') for xs in product(
        ['green1', 'brown', 'purple', 'pink', 'green2', 'yellow', 'brown', 'gray'],
    )],
)];
`
export const Dark2 = {
  [Symbol.iterator]: function*() { for (let v of Object.values(this)) yield v; }, // TODO Factor out [how to type?]
  green1: '#1b9e77',
  orange: '#d95f02',
  purple: '#7570b3',
  pink: '#e7298a',
  green2: '#66a61e',
  yellow: '#e6ab02',
  brown: '#a6761d',
  gray: '#666666',
};

// Set1
`
[print(f'  {name}: {hex!r},') for hex, name in zip_longest(
    mpl_cmap_colors_to_hex('Set1'),
    [stringcase.camelcase('_'.join(xs)).replace('_', '') for xs in product(
        ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'brown', 'pink', 'gray'],
    )],
)];
`
export const Set1 = {
  [Symbol.iterator]: function*() { for (let v of Object.values(this)) yield v; }, // TODO Factor out [how to type?]
  red: '#e41a1c',
  blue: '#377eb8',
  green: '#4daf4a',
  purple: '#984ea3',
  orange: '#ff7f00',
  yellow: '#ffff33',
  brown: '#a65628',
  pink: '#f781bf',
  gray: '#999999',
};

// Set2
`
[print(f'  {name}: {hex!r},') for hex, name in zip_longest(
    mpl_cmap_colors_to_hex('Set2'),
    [stringcase.camelcase('_'.join(xs)).replace('_', '') for xs in product(
        ['teal', 'salmon', 'blue', 'purple', 'green', 'yellow', 'brown', 'gray'],
    )],
)];
`
export const Set2 = {
  [Symbol.iterator]: function*() { for (let v of Object.values(this)) yield v; }, // TODO Factor out [how to type?]
  teal: '#66c2a5',
  salmon: '#fc8d62',
  blue: '#8da0cb',
  purple: '#e78ac3',
  green: '#a6d854',
  yellow: '#ffd92f',
  brown: '#e5c494',
  gray: '#b3b3b3',
};

// Set3
`
[print(f'  {name}: {hex!r},') for hex, name in zip_longest(
    mpl_cmap_colors_to_hex('Set3'),
    [stringcase.camelcase('_'.join(xs)).replace('_', '') for xs in product(
        ['teal', 'yellow', 'mauve', 'salmon', 'blue', 'orange', 'green', 'pink', 'gray', 'purple', 'sea', 'lemon'],
    )],
)];
`
export const Set3 = {
  [Symbol.iterator]: function*() { for (let v of Object.values(this)) yield v; }, // TODO Factor out [how to type?]
  teal: '#8dd3c7',
  yellow: '#ffffb3',
  mauve: '#bebada',
  salmon: '#fb8072',
  blue: '#80b1d3',
  orange: '#fdb462',
  green: '#b3de69',
  pink: '#fccde5',
  gray: '#d9d9d9',
  purple: '#bc80bd',
  sea: '#ccebc5',
  lemon: '#ffed6f',
};

// tab10
`
[print(f'  {name}: {hex!r},') for hex, name in zip_longest(
    mpl_cmap_colors_to_hex('tab10'),
    [stringcase.camelcase('_'.join(xs)).replace('_', '') for xs in product(
        ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'],
    )],
)];
`
export const tab10 = {
  [Symbol.iterator]: function*() { for (let v of Object.values(this)) yield v; }, // TODO Factor out [how to type?]
  blue: '#1f77b4',
  orange: '#ff7f0e',
  green: '#2ca02c',
  red: '#d62728',
  purple: '#9467bd',
  brown: '#8c564b',
  pink: '#e377c2',
  gray: '#7f7f7f',
  olive: '#bcbd22',
  cyan: '#17becf',
};

// tab20
`
[print(f'  {name}: {hex!r},') for hex, name in zip_longest(
    mpl_cmap_colors_to_hex('tab20'),
    [stringcase.camelcase('_'.join(xs)).replace('_', '') for xs in product(
        ['dark', 'light'],
        ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'],
    )],
)];
`
export const tab20 = {
  [Symbol.iterator]: function*() { for (let v of Object.values(this)) yield v; }, // TODO Factor out [how to type?]
  darkBlue: '#1f77b4',
  darkOrange: '#aec7e8',
  darkGreen: '#ff7f0e',
  darkRed: '#ffbb78',
  darkPurple: '#2ca02c',
  darkBrown: '#98df8a',
  darkPink: '#d62728',
  darkGray: '#ff9896',
  darkOlive: '#9467bd',
  darkCyan: '#c5b0d5',
  lightBlue: '#8c564b',
  lightOrange: '#c49c94',
  lightGreen: '#e377c2',
  lightRed: '#f7b6d2',
  lightPurple: '#7f7f7f',
  lightBrown: '#c7c7c7',
  lightPink: '#bcbd22',
  lightGray: '#dbdb8d',
  lightOlive: '#17becf',
  lightCyan: '#9edae5',
};

// tab20b
`
[print(f'  {name}: {hex!r},') for hex, name in zip_longest(
    mpl_cmap_colors_to_hex('tab20b'),
    [stringcase.camelcase('_'.join(xs)).replace('_', '') for xs in product(
        ['blue', 'green', 'yellow', 'red', 'purple'],
        ['0', '1', '2', '3'],
    )],
)];
`
export const tab20b = {
  [Symbol.iterator]: function*() { for (let v of Object.values(this)) yield v; }, // TODO Factor out [how to type?]
  blue0: '#393b79',
  blue1: '#5254a3',
  blue2: '#6b6ecf',
  blue3: '#9c9ede',
  green0: '#637939',
  green1: '#8ca252',
  green2: '#b5cf6b',
  green3: '#cedb9c',
  yellow0: '#8c6d31',
  yellow1: '#bd9e39',
  yellow2: '#e7ba52',
  yellow3: '#e7cb94',
  red0: '#843c39',
  red1: '#ad494a',
  red2: '#d6616b',
  red3: '#e7969c',
  purple0: '#7b4173',
  purple1: '#a55194',
  purple2: '#ce6dbd',
  purple3: '#de9ed6',
};

// tab20c
`
[print(f'  {name}: {hex!r},') for hex, name in zip_longest(
    mpl_cmap_colors_to_hex('tab20c'),
    [stringcase.camelcase('_'.join(xs)).replace('_', '') for xs in product(
        ['blue', 'orange', 'green', 'purple', 'gray'],
        ['0', '1', '2', '3'],
    )],
)];
`
export const tab20c = {
  [Symbol.iterator]: function*() { for (let v of Object.values(this)) yield v; }, // TODO Factor out [how to type?]
  blue0: '#3182bd',
  blue1: '#6baed6',
  blue2: '#9ecae1',
  blue3: '#c6dbef',
  orange0: '#e6550d',
  orange1: '#fd8d3c',
  orange2: '#fdae6b',
  orange3: '#fdd0a2',
  green0: '#31a354',
  green1: '#74c476',
  green2: '#a1d99b',
  green3: '#c7e9c0',
  purple0: '#756bb1',
  purple1: '#9e9ac8',
  purple2: '#bcbddc',
  purple3: '#dadaeb',
  gray0: '#636363',
  gray1: '#969696',
  gray2: '#bdbdbd',
  gray3: '#d9d9d9',
};
