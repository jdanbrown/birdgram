# Run

Run ios app in simulator
```sh
# 1. Open react-native-debugger
$ open -a 'React Native Debugger'
# 2. Run `react-native start ...` in its own term
# - Workaround for https://github.com/facebook/react-native/issues/21490#issuecomment-427240356
$ react-native start --reset-cache
# Start the app (in the simulator, by default)
$ react-native run-ios
```

Run tsc
```sh
$ yarn run tsc
$ yarn run tsc --watch
```

Run tests
```sh
$ yarn test
$ yarn test --watch
```

# Device logs

iOS
```sh
$ brew install --HEAD usbmuxd libimobiledevice
$ idevicesyslog
$ idevicesyslog | hi-color 'bold red' Error | hi-color 'bold green' Notice | ag --no-color -i birdgram
```

# Resources
- Project setup
  - https://facebook.github.io/react-native/blog/2018/05/07/using-typescript-with-react-native
- IDE setup
  - https://nuclide.io/docs/features/debugger/
  - https://nuclide.io/docs/platforms/react-native/
  - https://nuclide.io/docs/platforms/ios/
  - https://nuclide.io/docs/platforms/android/
  - https://github.com/TypeStrong/atom-typescript
- Library docs
  - react
    - "Main Concepts"
      - 1. https://reactjs.org/docs/hello-world.html
      - ...
      - 12. https://reactjs.org/docs/thinking-in-react.html
    - "Advanced Guides"
      - https://reactjs.org/docs/context.html
      - https://reactjs.org/docs/higher-order-components.html
      - https://reactjs.org/docs/refs-and-the-dom.html
  - react-native
    - https://facebook.github.io/react-native/docs/getting-started
  - typescript
    - https://www.typescriptlang.org/docs/
- Snapshot testing
  - https://jestjs.io/docs/en/tutorial-react-native.html
  - https://jestjs.io/docs/en/snapshot-testing

# Assets
- Problem: managing assets in react-native is a minefield, e.g.
  - Looks easy enough: https://facebook.github.io/react-native/docs/images#static-non-image-resources
  - Oops, jk: https://github.com/facebook/react-native/issues/16446
  - Someone had to build a custom package to make it work: https://github.com/unimonkiez/react-native-asset
- Solution: use https://github.com/unimonkiez/react-native-asset
- To add/remove assets:
  - Add/remove file(s) in `app/assets/`
  - Run `./node_modules/.bin/react-native-asset`
  - ios: Rebuild in Xcode
  - android: TODO Test with Android Studio
- To use assets:
  - ios: `${fs.dirs.MainBundleDir}/<asset-file>`
  - android: TODO Try and see

# Troubleshooting
- 'config.h' "File not found"
  - I fixed it by cleaning and rebuilding yarn/xcode a number of times
  - But if I get stuck on this again, try these:
    - https://github.com/facebook/react-native/issues/14382#issuecomment-424516909
    - https://github.com/facebook/react-native/issues/20774
      - `cd node_modules/react-native/third-party/glog-0.3.5 && ../../scripts/ios-configure-glog.sh`
      - Xcode -> Build
- Can't import modules `fs`, `net`, or `dgram`
  - Finish installing them via node-libs-react-native (skipped because they have native code)
  - https://github.com/parshap/node-libs-react-native#other-react-native-modules
- App takes forever (~20s) to open (in dev mode)
  - It might be trying and failing to connect to rndebugger
  - Try first opening React Native Debugger and then restarting/reloading the app
  - (Else try the device console...?)
- react-native-debugger won't connect
  - This worked for me once:
    - Open React Native Debugger (.app)
    - Run `react-native start ...` (as above)
    - Open Simulator
    - Open app in Simulator
  - Else, just keep resetting lots of stuff :/
- Add new native dep -> `react-native run-ios` -> app shows strange errors
  - Rebuild in xcode (cmd-b) -> `react-native run-ios`
- App shows warning "RCTBridge required dispatch_sync to load RCTDevLoadingView. This may lead to deadlocks."
  - Rebuild in xcode (cmd-b) -> `react-native run-ios`
  - https://github.com/facebook/react-native/issues/16376
  - https://github.com/rebeccahughes/react-native-device-info/issues/260
- `yarn test` -> `Couldn't find preset "module:metro-react-native-babel-preset"`
  - Solution: updated package.json like https://github.com/facebook/metro/issues/242#issuecomment-421139247
- Symlinks break `react-native start` build
  - Symlinks aren't supported by metro: https://github.com/facebook/metro/issues/1

# TODO
- Fix: `global.*` fails if not "Debug JS Remotely"
- Fix: `import 'jimp'` fails if not "Debug JS Remotely" (see comments in Recorder.tsx)
