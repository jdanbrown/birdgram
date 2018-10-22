declare module 'react-native-debug-stylesheet' {

  // Just re-export StyleSheet from react-native
  //  - Note that import structure is different between react-native and react-native-debug-stylesheet:
  //      import { StyleSheet } from 'react-native';
  //      import StyleSheet from 'react-native-debug-stylesheet';
  import { StyleSheet } from 'react-native';
  export default StyleSheet;

}
