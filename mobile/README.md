# Setup
- Clone git submodules
  - `git submodule update --init`
- Install cocoapod deps
  - `cd ios/ && pod install`
- Generate `*.xcodeproj` projects for manual deps
  - `cd ios/swift-npy/ && swift package generate-xcodeproj`
- Build/run
  - Xcode `Birdgram.xcworkspace` -> scheme 'Birdgram' -> Build/Run

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
$ yarn run tsc -w
```

Run tests
```sh
$ yarn jest
$ yarn jest --watch
```

# Submit to App Store
- Overview
  - https://developer.apple.com/app-store/launch/
  - https://developer.apple.com/app-store/product-page/
  - https://developer.apple.com/app-store/search/
  - https://developer.apple.com/app-store/review/
  - https://developer.apple.com/app-store/review/guidelines/
  - https://developer.apple.com/app-store/review/#common-app-rejections
- Screenshot size specs
  - https://help.apple.com/app-store-connect/#/devd274dd925
- Keyword SEO
  - https://developer.apple.com/app-store/search/
  - https://stage.tune.com/blog/top-3-ios-app-store-optimization-tricks/ (2012)
    - App Store maybe doesn't do stemming? (wat)

# Upload to TestFlight (e.g. US)
1. `fastlane` takes ~50m to build and upload
  - ~15m build + ~33m upload (2.6GB US .ipa @ ~1.4MB/s)
  - (CA100: ~6m build + 3m upload)
2. Apple takes ~10h to process until it's available to internal testers and for submission for beta review
  - ~4h to show up as "Processing" ("Upload Date" in console) + ~6h until "completed processing" (via email)
  - (CA100: ~5m until "completed processing" email)
3. Manually submit for beta review
  - By adding the build to the external testers group ("Group: public testers")
4. Apple takes 0m to "Approve" for external testers
  - Takes ~days for the first build of a new app (e.g. 41h for US, 30h for CR)
```sh
$ cd ios/
$ bin/fastlane-env US test-flight
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

# Troubleshooting (newest to oldest)
- How to download old versions of Xcode
  - Because you have to upgrade Xcode (and macOS!!) for each upgrade of iOS (ugh terrible!!)
    - Xcode 11.5, iOS 13.5, macOS Catalina ≥10.15.2 [https://developer.apple.com/documentation/xcode_release_notes/xcode_11_5_release_notes]
    - Xcode 11.4, iOS 13.4, macOS Catalina ≥10.15.2 [https://developer.apple.com/documentation/xcode_release_notes/xcode_11_4_release_notes]
    - Xcode 11.3, iOS 13.3, macOS Mojave   ≥10.14.4 [https://developer.apple.com/documentation/xcode_release_notes/xcode_11_3_release_notes]
  - App Store only has latest version, which usually requires latest macOS (e.g. Catalina which I don't want yet)
    - Download: https://developer.apple.com/download/more/
    - More info: https://xcodereleases.com/
  - To download more quickly
    - Apple download speeds are ridiculous
      - apple->laptop   ~200-700KB/s   ~5h
      - apple->gcloud     ~20-60MB/s   ~3m
      - gcloud->laptop        ~3MB/s  ~40m
    - https://console.cloud.google.com/ -> open Cloud Shell
      - In Chrome, "Copy as cURL" any request from https://developer.apple.com/download/more/
      - `curl xcode-url ... >xcode-file`
      - `gsutil cp xcode-file gs://xcode-downloads/...`
    - Locally
      - `gsutil cp gs://xcode-downloads/... ~/Desktop`
- Fix code signing certificate / provisioning profile after getting new phone
  - Approach 1: Manually add new device's UDID to device list
    - https://developer.apple.com/account/resources/devices/list
    - XXX Aborted, couldn't get it to work
  - Approach 2: Build once with "Automatically manage signing" enabled, to let Xcode register the device
    - "If you use automatic signing, Xcode registers connected devices for you"
      - From: https://developer.apple.com/account/resources/devices/add
    - Steps
      1. Xcode -> project Birdgram -> target Birdgram -> Signing & Capabitilies -> Signing (Debug) -> Automatically manage signing
        - Normally disabled (for fastlane)
        - Enable -> Build once -> Disable
      2. Manually add new device to _each_ provisioning profile (ugh)
        - https://developer.apple.com/account/resources/profiles/list
        - Click into each "Development" profile (x4) -> Edit -> Devices -> add new device -> Save
          - (If new device isn't present, then step 1 didn't work)
- Reset which code signing certificate is used by Xcode
  - `CONFIGURATION=Debug bin/fastlane-env CA100 match development`
    - TODO Or maybe? `CONFIGURATION=Debug PROVISIONING_PROFILE_TYPE=Development bin/fastlane-env CA100 match development`
  - And then click around a bunch in Xcode -> [Project] -> Signing & Capabitilies
- Regenerate code signing certificates for ios / app store (these expire every ~year)
  - Use `fastlane match`
    - https://docs.fastlane.tools/actions/match
  - Monitor certs as fastlane updates them
    - https://developer.apple.com/account/resources/certificates/list
    - https://developer.apple.com/account/resources/profiles/list
  - Revoke and delete old certs (this won't break apps already in App Store and TestFlight)
    - `bin/fastlane-env CA100 match nuke development`
    - `bin/fastlane-env CA100 match nuke distribution`
  - Generate and install new certs (for future builds and uploads to App Store and TestFlight)
    - HACK Might have to interactively copy/paste APP_BUNDLE_ID value out of mobile/.env
      - Is something wrong with MATCH_APP_IDENTIFIER in ios/fastlane/Fastfile?
    - Each runs create the provisioning profile per app bundle id
    - First run per development/appstore creates the cert
    - `CONFIGURATION=Debug bin/fastlane-env CA100 match development`
    - `CONFIGURATION=Debug bin/fastlane-env CA3500 match development`
    - `CONFIGURATION=Debug bin/fastlane-env US match development`
    - `CONFIGURATION=Debug bin/fastlane-env CR match development`
    - `CONFIGURATION=Release bin/fastlane-env CA100 match appstore`
    - `CONFIGURATION=Release bin/fastlane-env CA3500 match appstore`
    - `CONFIGURATION=Release bin/fastlane-env US match appstore`
    - `CONFIGURATION=Release bin/fastlane-env CR match appstore`
- App Store warning about deprecated UIWebView (but app upload is still accepted) [2nd time!]
  - Problem
    - Email after upload: `ITMS-90809: Deprecated API Usage - Apple will stop accepting submissions of apps that use UIWebView APIs`
    - Our react-native is really old and still has RCTWebView
  - Solution
    - Rip out RCTWebView
      - `git show --stat 482d7558`
      - `git cherry-pick 482d7558`
      - From https://github.com/facebook/react-native/issues/26255#issuecomment-528275747
    - Future react-native versions move WebView out to https://github.com/react-native-community/react-native-webview
      - https://github.com/facebook/react-native/pull/16792#issuecomment-429483747
- Loading sqlite db (search_recs.sqlite3) fails / show spinner forever
  - Problem
    - react-native-sqlite-plugin-legacy-support doesn't support paths, only filenames (wat)
  - Solution
    - Apply manual patch
    - `git show --stat 4abcf618`
    - `git cherry-pick 4abcf618`
- How to read .wav tags outside of Birdgram app
  - Solution
    - `pyprinttags3 <file.wav>`
  - Requires pytaglib
    - `pip install pytaglib` (https://github.com/supermihi/pytaglib)
    - Might have to manually create `pyprinttags3`: https://gist.github.com/jdanbrown/fa8f9ed214f014873efcd57b96976de1
  - Problem
    - None of these work: ffmpeg, soxi, exiftool, id3info
      - http://ffmpeg.org/ffmpeg-formats.html#Metadata-1
      - https://stackoverflow.com/questions/51869859/get-duration-of-wav-from-sox-in-python
    - We write .wav metadata using taglib, which is apparently nonstandard enough that nothing else can read it
      - https://taglib.org/
      - https://en.wikipedia.org/wiki/WAV#Metadata
- Xcode run on device fails with:
  - `Swift class extensions and categories on Swift classes are not allowed to have +load methods`
  - Problem
    - Our react-native version (0.57.x) is too old for our ios (≥13) / Xcode (≥11) / swift (≥5) versions
  - Solution
    - Apply 3 backports from react-native 0.59.3, 0.59.10, and 0.61.1
    - `git show --stat 7643d8b2 92d4cc99 0f4f1b87`
    - `git cherry-pick 7643d8b2 92d4cc99 0f4f1b87`
- App Store warning about deprecated UIWebView (but app upload is still accepted) [1st time]
  - ITMS-90809: Deprecated API Usage - Apple will stop accepting submissions of apps that use UIWebView APIs. See
    https://developer.apple.com/documentation/uikit/uiwebview for more information.
  - Not useful for me, but relevant context
    - https://github.com/react-native-community/react-native-webview/issues/819
    - https://github.com/apache/cordova-ios/issues/661
  - Solution to my problem (react-native-device-info)
    - https://stackoverflow.com/a/58201802/397334
    - https://www.devsbedevin.net/react-native-itms-90809-warning-from-apple/
    - https://github.com/react-native-community/react-native-device-info/issues/756
    - ✅ Update react-native-device-info
- Debug app starts failing to connect to remote debugger at startup, after previously being able to
  - e.g. after plugging in usb and getting the dreaded "A software update is required to connect to iPhone" runaround
  - Solution [maybe worked once?]
    - Restart `react-native start`
    - Bounce laptop wifi
    - Bounce phone wifi
    - Restart Debug app
- `react-native link` fails with `Cannot read property 'buildConfigurationList' of undefined`
  - Make sure `Birdgram` is the first target in the xcode project
  - `react-native link` only links the first target: https://github.com/react-native-community/react-native-cli/issues/41
- Xcode build fails with:
  - `unable to read input file '.../Birdgram.build/Preprocessed-Info.plist': No such file or directory`
  - Solution: retry the build (ugh)
- App hangs on startup with cryptic error in xcode logs:
  - `... [DYGLInitPlatform] connection to host has failed: Error Domain=NSPOSIXErrorDomain Code=2 "No such file or directory"`
  - `... aborting: platform initialization failed`
  - Cause: no f'ing idea. _Two_ google results for this error msg, and neither is helpful:
    - https://stackoverflow.com/questions/47224594/error-at-launch-app
    - https://github.com/levantAJ/Measure/issues/2
  - Workaround (I think):
    - Stop in xcode (cmd-.)
    - Launch app directly from device, without xcode
    - Seems to usually work...
    - Maybe also just retry Run in xcode
    - Also try Clean in xcode (shift-cmd-k)
    - Also try rebooting everything (I tried this once, could have helped)
- `dyld: Library not loaded: @rpath/libswiftAVFoundation.dylib`
  - Clean (cmd-shift-k) and rebuild [https://stackoverflow.com/a/33502910/397334]
- To add an SPM project (Package.swift) that doesn't have an *.xcodeproj
  - (e.g. SwiftNpy: https://github.com/qoncept/swift-npy)
  - `git clone` into ios/ dir
  - `swift package generate-xcodeproj` inside the cloned repo to generate the *.xcodeproj
  - HACK Fix `import SwiftNpy` -> "missing required module Cminimap"
    - Critical help here: https://bugs.swift.org/browse/SR-4972
      - Add swift flags: Xcode -> project Bubo -> targets {Bubo,Bubo-macos} -> Build Settings -> "Other Swift Flags" -> ...
        - `-Xcc -fmodule-map-file=$(SRCROOT)/../swift-npy/SwiftNpy.xcodeproj/GeneratedModuleMap/Cminizip/module.modulemap`
      - Add swift flags: Xcode -> project Birdgram -> target Birdgram -> Build Settings -> "Other Swift Flags" -> ...
        - `-Xcc -fmodule-map-file=$(SRCROOT)/swift-npy/SwiftNpy.xcodeproj/GeneratedModuleMap/Cminizip/module.modulemap`
      - Add swift flags: bin/swift
        - `-Xcc -fmodule-map-file="$dir"/swift-npy/SwiftNpy.xcodeproj/GeneratedModuleMap/Cminizip/module.modulemap`
    - Alternate approach I didn't try: vendor all the files into Bubo and import the *.c into *.swift
      - No "bridging header" for framework target; use the "umbrella header" instead
      - https://developer.apple.com/documentation/swift/imported_c_and_objective-c_apis/importing_objective-c_into_swift
- `swift` repl fails with "error: Couldn't lookup symbols" for e.g. Bubo symbols
  - In the repl code, before `import Bubo`, manually `import` every module from xcode's "Linked Frameworks and Libraries", e.g.
    - Works: `import Bubo`
    - Fails: `import Bubo; bubo_foo()`
    - Fails: `import Bubo; import Surge; ...; import Yams; bubo_foo()`
    - Works: `import Surge; ...; import Yams; import Bubo; bubo_foo()`
  - This is horse shit
  - cf. ~/bin/ipykernel-install-swift.md for more gory details about `swift` repl
- Playground fails to import frameworks (e.g. Bubo)
  - Build the framework (e.g. Bubo) _once_ against a non-device device (e.g. "Generic iOS Device" / "iPhone 8")
  - Then it doesn't matter which device you run against
  - https://stackoverflow.com/a/37737190/397334
- App fails on startup with cryptic stacktrace `__dyld_start` -> `__abort_with_payload` (or other `SIGABRT`)
  - This happened once when app Birdgram depended on framework Bubo
    - Solution: in Birdgram project, "Embedded Binaries" -> + -> "Bubo.framework" (in addition to "Linked Libraries and Frameworks")
  - This happens occasionally for no reason [+2 times]
    - Solution: try again (build -> run)
- js code doesn't update in Xcode Release build
  - Silent js build error [e.g. next bullet]
  - To find it, try checking previous build msgs, or try clean build folder and rebuild to resurface it
- Xcode Release build fails with "main.jsbundle does not exist. This must be a bug with"
  - Not the real error -- find the preceeding error msg in Xcode
  - Last time it was a js build-time error that tripped in Release but not Debug (b/c it relied on `__DEV__` true vs. undefined)
- App does weird stuff, e.g. fails on startup with strange errors, or loads to a blank screen
  - Make sure you're on a happy path
    - ✅ Debug build + "Debug JS Remotely"
    - ❌ Debug build + no "Debug JS Remotely" -- untested, and usually does weird stuff
    - ✅ Release build
  - Try toggling "Debug JS Remotely" back and forth
    - I've observed this make Debug build + no "Debug JS Remotely" go from blank screen to a working app...
- App starts loading _very_ slowly in Debug mode (with high cpu on laptop)
  - Try restarting rndebugger (this worked for me once)
- App hangs/slow on startup
  - Solution: if using chrome rn debugger, try toggling http://192.168.0.196:8081/debugger-ui/ vs. http://localhost:8081/debugger-ui/
    - Simulator wants http://localhost:8081/debugger-ui/
    - Phone wants http://192.168.0.196:8081/debugger-ui/
    - If wrong one you should see a "cross-origin read blocking (CORB)" warning in the console
- App hangs/dies on startup and ios device logs show a bunch of tcp errors (when trying to connect to metro bundler)
  - Solution: temporarily comment out `export SKIP_BUNDLING=true` in Build Phases: [...like below]
- App fails on startup with "No bundle URL present" (Debug build)
  - Solution: temporarily comment out `export SKIP_BUNDLING=true` in Build Phases:
    - https://facebook.github.io/react-native/docs/running-on-device#3-configure-app-to-use-static-bundle
  - (But I don't really understand the problem, since shouldn't this only matter for Release builds...?)
- Laptop sound output is messed up (very tinny)
  - Solution: change sound input device to "Internal Microphone"
- 'config.h' "File not found"
  - I fixed it by cleaning and rebuilding yarn/xcode a number of times
  - But if I get stuck on this again, try these:
    - https://github.com/facebook/react-native/issues/14382#issuecomment-424516909
    - https://github.com/facebook/react-native/issues/20774
      - ✅ `(cd node_modules/react-native/third-party/glog-0.3.5 && ../../scripts/ios-configure-glog.sh)`
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
