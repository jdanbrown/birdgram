module.exports = {

  // As per https://github.com/ds300/react-native-typescript-transformer
  getTransformModulePath() {
    return require.resolve('react-native-typescript-transformer');
  },
  getSourceExts() {
    return ['ts', 'tsx'];
  },

  // As per https://github.com/parshap/node-libs-react-native
  //  - And https://github.com/facebook/metro/blob/master/docs/Configuration.md
  resolver: {
    extraNodeModules: require('node-libs-react-native'),
  },

};
