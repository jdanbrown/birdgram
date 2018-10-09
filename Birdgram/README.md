# Run

Run ios app in simulator
```sh
# First run (in bg term)
# - Workaround for https://github.com/facebook/react-native/issues/21490#issuecomment-427240356
$ react-native start --reset-cache

# Then run
react-native run-ios
```

Run tests
```sh
$ yarn test
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

# Bugs + workarounds
- `yarn test` -> `Couldn't find preset "module:metro-react-native-babel-preset"`
  - Solution: updated package.json like https://github.com/facebook/metro/issues/242#issuecomment-421139247
- `react-native run-ios` -> yellow warning "RCTBridge required dispatch_sync to load RCTDevLoadingView. This may lead to deadlocks."
  - Solution: build in xcode -> rebuild with `react-native run-ios` -> error went away...
  - https://github.com/facebook/react-native/issues/16376
  - https://github.com/rebeccahughes/react-native-device-info/issues/260
