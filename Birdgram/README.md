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

ios
```sh
$ brew install --HEAD usbmuxd libimobiledevice
$ idevicesyslog
$ idevicesyslog | hi-color 'bold red' Error | hi-color 'bold green' Notice | ag --no-color -i birdgram
```

android
- TODO

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
- Problem 1: managing assets in react-native is a minefield, e.g.
  - Looks easy enough: https://facebook.github.io/react-native/docs/images#static-non-image-resources
  - Oops, jk: https://github.com/facebook/react-native/issues/16446
  - Someone had to build a custom package to make it work: https://github.com/unimonkiez/react-native-asset
  - Solution: use https://github.com/unimonkiez/react-native-asset
- Problem 2: react-native-asset chokes on large assets
  - Solution: add large assets manually
- To add/remove assets
  - `app/assets/manual/`: for large/many asset files (e.g. search_recs/, react-native-asset took >30m for 35k recs)
    - Xcode
      - https://help.apple.com/xcode/mac/current/#/dev81ce1d383
      - Select project -> Resources -> Add Files to...
        - Uncheck: Destination -> Copy items if needed (unless you want the assets copied into the project dir)
        - Select: Added folders -> Create folder references [but I don't actually know what "Create groups" does...]
      - Caveats:
        - If you select a dir containing symlinks, the build will fail (with no helpful error msg)
        - If you select a symlink it will be deref'ed, which means:
          - Its name won't be what you expect
          - If you change the symlink later you'll have to manuall remove and re-add
    - Android Studio
      - TODO [Fingers crossed that this is feasible with large search_recs...]
      - [Maybe a lead? https://github.com/zmxv/react-native-sound#basic-usage]
  - `app/assets/auto/`: only for small/few asset files where you don't want to mess with the manual steps
    - Add/remove file(s) in `app/assets/auto/`
    - Run `./node_modules/.bin/react-native-asset`
    - Xcode
      - Rebuild
    - Android Studio
      - TODO Try and see
- To use assets
  - ios: `${fs.dirs.MainBundleDir}/<asset-file>`
  - android: ...
    - TODO Try and see

# Build for Release vs. Debug
- Xcode
  - Product -> Scheme -> Edit Scheme... (cmd-<)
  - Run -> Build Configuration -> Debug/Release
  - Build/Run as usual
- Android Studio
  - TODO

# Troubleshooting
- Playground fails to import frameworks (e.g. Bubo)
  - Build the framework (e.g. Bubo) _once_ against a non-device device (e.g. "Generic iOS Device" / "iPhone 8")
  - Then it doesn't matter which device you run against
  - https://stackoverflow.com/a/37737190/397334
- App fails on startup with cryptic stacktrace `__dyld_start` -> `__abort_with_payload`
  - This happened when app Birdgram depended on framework Bubo
  - Solution: in Birdgram project, "Embedded Binaries" -> + -> "Bubo.framework" (in addition to "Linked Libraries and Frameworks")
- js code doesn't update in Xcode Release build
  - Silent js build error [e.g. next bullet]
  - To find it, try checking previous build msgs, or try clean build folder and rebuild to resurface it
- Xcode Release build fails with "main.jsbundle does not exist. This must be a bug with"
  - Not the real error -- find the preceeding error msg in Xcode
  - Last time it was a js build-time error that tripped in Release but not Debug (b/c it relied on `__DEV__` true vs. undefined)
- App starts loading _very_ slowly in Debug mode (with high cpu on laptop)
  - Try restarting rndebugger (this worked for me once)
- App does weird stuff, e.g. fails on startup with strange errors, or loads to a blank screen
  - Make sure you're on a happy path
    - ✅ Debug build + "Debug JS Remotely"
    - ❌ Debug build + no "Debug JS Remotely" -- untested, and usually does weird stuff
    - ✅ Release build
  - Try toggling "Debug JS Remotely" back and forth
    - I've observed this make Debug build + no "Debug JS Remotely" go from blank screen to a working app...
- App hangs/dies on startup and ios device logs show a bunch of tcp errors (when trying to connect to metro bundler)
  - Solution: temporarily comment out `export SKIP_BUNDLING=true` in Build Phases: [...like below]
- App fails on startup with "No bundle URL present" (Debug build)
  - Solution: temporarily comment out `export SKIP_BUNDLING=true` in Build Phases:
    - https://facebook.github.io/react-native/docs/running-on-device#3-configure-app-to-use-static-bundle
  - (But I don't really understand the problem, since shouldn't this only matter for Release builds...?)
- 'config.h' "File not found"
  - I fixed it by cleaning and rebuilding yarn/xcode a number of times
  - But if I get stuck on this again, try these:
    - https://github.com/facebook/react-native/issues/14382#issuecomment-424516909
    - https://github.com/facebook/react-native/issues/20774
      - `(cd node_modules/react-native/third-party/glog-0.3.5 && ../../scripts/ios-configure-glog.sh)`
      - Xcode -> Build
- Can't import modules `fs`, `net`, or `dgram`
  - Finish installing them via node-libs-react-native (skipped because they have native code)
  - https://github.com/parshap/node-libs-react-native#other-react-native-modules
- App hangs, even after force quit
  - Maybe hit a debug breakpoint in Xcode
  - Xcode -> cmd-. to stop debugging
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
