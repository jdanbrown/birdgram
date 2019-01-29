const blacklist = require('metro-config/src/defaults/blacklist');

module.exports = {

  // As per https://github.com/ds300/react-native-typescript-transformer
  getTransformModulePath() {
    return require.resolve('react-native-typescript-transformer');
  },
  getSourceExts() {
    return ['ts', 'tsx'];
  },

  // As per
  //  - https://github.com/parshap/node-libs-react-native
  //  - https://facebook.github.io/metro/docs/en/configuration
  resolver: {
    extraNodeModules: require('node-libs-react-native'),
    // Blacklist ignores paths for auto-refresh [https://stackoverflow.com/a/41963217/397334]
    blacklistRE: blacklist([
      /.*\.playground\/.*/,
    ])
  },

};
