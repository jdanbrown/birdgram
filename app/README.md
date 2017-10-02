v0:
- [ ] v0: Show list of likely birds from ebird based on (latlon, date)
  - [ ] app: "Go" button -> svc(latlon, date) -> list[(score, bird)] -> show(birds)
    - Show list view, sorted by score desc
    - Show list item: bird name, bird image, bird freq histo
  - [ ] svc: (latlon, date) -> query ebird -> list[birds], top k by score(bird_freq)
    - No local storage or caching
- [ ] v1:
  - app: "Record"/"Stop" -> wav -> svc(wav, latlon, date) -> (spec, list[(score, bird, list[spec])]) -> show(spec, birds)
    - Do wav -> spec in python and not in js (defer js for later)
    - Show list item: ..., specs
  - svc: (wav, latlon, date) -> (spec, list[(score, bird, list[spec])], top k by score(spec_in, spec, bird_freq)
    - algo: freq histo per spec, ignore time, dumb hi-pass filter to denoise
    - db: birds(id, name, photo, freq_histo) + calls(bird_id, wav, spec, features...)
    - No caching
- vNext ideas:
  - [ ] app: Let user clip spec (start, stop) + (lo_freq,) with bounding box
  - [ ] app: Show live spec while/before recording
  - [ ] app: Easy to visually scan candidate specs to look for matches
  - [ ] app: Filter/group/focus by bird family, e.g. +passerines +jays -gulls
    - group by taxa, sort species with taxa by prob, sort taxa by aggregate prob, allow collapse/expand each taxa
  - [ ] app: multiple levels of detail/zoom: probs per taxa (no scroll), probs per species, probs+details per species (heavy scroll)
  - [ ] app: show a map with the hotspots where the data is coming from (use case: too many too far away, use fewer)
  - [ ] app inspiration: ebird's new "illustrated checklist": https://ebird.org/ebird/hotspot/L590796/media

v1:
- [x] setup expo
- [x] create gcp project
- [x] mock backend: flask + ebird + /barchart + cache
- [x] try out new react-native playground: https://blog.expo.io/sketch-a-playground-for-react-native-16b2401f44a2#.2w4c1zr25
- [x] try out new react-native-debugger: https://github.com/jhen0409/react-native-debugger
- [x] mock backend: dockerize
- [x] mock backend: deploy /barchart endpoint to k8s
-   [x] TODO Figure out why I can't get kubectl to auth to jdanbrown k8s cluster
- [x] v0 api
- [ ] use xeno-canto audio + spec data directly (e.g. stream directly to app + cache server side)
  - Has at least as much data as Macaulay, by sample count
    - http://www.xeno-canto.org/collection/stats/graphs vs. https://www.macaulaylibrary.org/
  - Terms of use look feasible: http://www.xeno-canto.org/about/terms
  - Can serve audio directly to app, e.g. http://www.xeno-canto.org/65830
    - Need a proxy: can't CORS the audio (proxy) + can't id the specs (generate on the fly ourselves)
    - Already very rich, but needs focus, efficiency, purpose, better workflow, etc.
      - http://www.xeno-canto.org/explore?query=%22Greater+Roadrunner+%22+q%3AA&view=3
    - Easy api:
      - http://www.xeno-canto.org/api/2/recordings?query=%22Greater+Roadrunner+%22+q%3AA&view=3
      - Docs: http://www.xeno-canto.org/article/153
- [ ] port app from app/bubo/ into app/bubo-2/ (is there even anything useful to port?)
- [ ] api: make slimmed-down route for app to consume (i.e. not the full set of rows from /barcharts)
- [ ] mock show result list
- [ ] mock upload recording
- [ ] mock live spectrogram
- [ ] Shunt spectrogram.js (web audio + canvas) into Expo bubo / react-native AwesomeProject:
  - Need web <canvas> in react-native:
    - https://facebook.github.io/react-native/docs/webview.html
    - http://stackoverflow.com/questions/34403231/react-native-canvas-in-webview
    - https://github.com/lwansbrough/react-native-canvas/blob/master/Canvas.js
    - https://github.com/Flipboard/react-canvas
- [ ] target kNN idea inspired by google t-SNE demo? https://paper.dropbox.com/doc/Bubo-notes-and-research-yu5ji2nwtKWNkhRAH21RH
  - [ ] Grok google t-SNE data proc code: https://github.com/kylemcdonald/AudioNotebooks

[WIP] compspectro:
- https://paper.dropbox.com/doc/Bubo-app-comparative-spectrograms-lVbQhmyyC4figs12IFHLP
- https://facebook.github.io/react-native/docs/components-and-apis.html
- http://docs.nativebase.io/Components.html
- https://docs.expo.io/

Happy workflow: Expo + react-native-debugger (easy + now supports inspect!)
- workspace:
  - vim
  - Expo.app (hot reload on file modified)
    - Gear icon -> Host -> LAN for rndebugger
  - React Native Debugger.app (inspect/console/debug)
    - Needs `package.json` patch below so that "Debug Remote JS" opens rndebugger instead of chrome
  - Simulator.app (hot reload, inspect)
    - cmd-d in simulator (not phone) -> "Debug Remote JS" to open rndebugger
    - cmd-d in simulator (not phone) -> "Toggle Element Inspector"
      - Click element -> selects in react-native-debugger + usable as `$r` in console (e.g. `$r.props`)
      - Must: cmd-opt-I to show console + switch "top" -> "RNDebuggerWorker.js" for `$r` to work
  - phone (hot reload but no inspect, complements simulator)
- docs: https://facebook.github.io/react-native/docs/components-and-apis.html
- docs: http://docs.nativebase.io/Components.html
- docs: https://docs.expo.io/
- inspect + debug + console: https://github.com/jhen0409/react-native-debugger
- edit: vim (expo reloads on file modified) + syntastic:flow
- deploy: expo
- fiddle: https://snack.expo.io – instant reload, when ~seconds reload is too slow
- repl: https://babeljs.io/repl
- example: https://github.com/exponentjs/native-component-list
- example: https://github.com/expo/examples/tree/master/with-victory-native – viz with svg

Alt happy workflow: Nuclide/Atom (more work + more control):
- hard: deploy (via xcode + dev install / app store)
- edit: vim + syntastic:flow
- inspect, console, debug: https://nuclide.io/docs/platforms/react-native/
  - Atom: "Add Project Folder" -> project dir
  - Atom: "Nuclide React Inspector: Toggle"
  - Atom: "Nuclide React Native: Start Packager"
  - Atom: "Nuclide React Native: Start Debugging"
  - Shell: `react-native run-ios` -> wait for simulator to launch
  - Simulator: cmd-d -> "Enable Hot Reloading"
  - Simulator: cmd-d -> "Debug Remote JS"
  - Simulator: Window -> Stay in Front
  - Atom: top right -> "Toggle new / old Nuclide Debugger UI" -> old UI for console
  - Atom: top right -> "(Debug) Open Web Inspector for the debugging frame" -> for console with autocomplete
- repl: https://babeljs.io/repl
- docs: https://facebook.github.io/react-native/docs/

Docs:
- https://facebook.github.io/react-native/docs/
- https://facebook.github.io/react-native/docs/debugging.html
Examples:
- https://github.com/facebook/react-native/tree/master/Examples
Expo + react-native-debugger:
- https://expo.io/ – build+run+logs (no editing) + hot-reload on file modified (vim!)
  - https://snack.expo.io/ – fiddle with instant reload (even though simulator/phone is ~seconds reload)
- https://github.com/jhen0409/react-native-debugger – react inspect + js debug
- Setup: change "Debug Remote JS" to open in rndebugger instead of chrome
  - https://www.npmjs.com/package/react-native-debugger-open
    - TLDR: Add this to `package.json` and then `npm install`:
      ```
      "devDependencies": {
        "react-native-debugger-open": "^0.3.11"
      },
      "scripts": {
        "postinstall": "rndebugger-open"
      }
      ```
- References:
  - https://www.gravitywell.co.uk/latest/rd/posts/react-native-debugger-expo-awesome/
  - https://docs.expo.io/versions/v19.0.0/guides/debugging.html
  - https://github.com/jhen0409/react-native-debugger/blob/master/docs/react-devtools-integration.md
    - https://github.com/jhen0409/react-native-debugger/blob/master/docs/debugger-integration.md
    - https://github.com/jhen0409/react-native-debugger/blob/master/docs/getting-started.md
  - https://github.com/facebook/react-devtools/tree/master/packages/react-devtools#integration-with-react-native-inspector
Nuclide:
- https://nuclide.io/ (in atom)
  - Better than chrome for element inspector: chrome react devtools don't work for react-native
  - Debugger requires node<=6.3.0 [https://github.com/facebook/react-devtools/issues/229]
  - Can't debug: "Internal error: illegal access" [https://github.com/node-inspector/node-inspector/issues/864]
Repls:
- https://babeljs.io/repl - easiest to use
- http://www.es6fiddle.net/ - with es6 examples
- https://jsfiddle.net/reactjs/ - maybe good for html+css?
Resources:
- https://facebook.github.io/react-native/releases/next/docs/more-resources.html
- https://github.com/jondot/awesome-react-native

APIs and libs:
- graphics / viz / svg:
  - https://formidable.com/open-source/victory/docs/native/ – e.g. bar charts, histos, basic bird viz
  - https://github.com/react-native-community/react-native-svg – svg, probably generally useful, e.g. for spectrograms
- ebird:
  - https://confluence.cornell.edu/display/CLOISAPI/eBird+API+1.1
  - https://confluence.cornell.edu/display/CLOISAPI/eBird-1.1-URIPatterns
  - js + barcharts: http://accubirder.com/bquery.php
  - (4y old) py + ebird: https://github.com/carsonmcdonald/python-ebird-wrapper

Audio (native):
- https://github.com/jsierles/react-native-audio/
- http://audiokit.io/
  - http://audiokit.io/docs/Classes/AKFFTTap.html
- http://theamazingaudioengine.com/doc2/
- https://developer.apple.com/reference/avfoundation/avaudioengine
Audio (web):
- https://developer.mozilla.org/en-US/docs/Web/API/AudioContext
- https://developer.mozilla.org/en-US/docs/Web/API/AudioContext/createAnalyser
- https://developer.mozilla.org/en-US/docs/Web/API/AnalyserNode
- ex. https://github.com/sebleier/spectrogram.js
  - needs <canvas>
    - https://github.com/lwansbrough/react-native-canvas/blob/master/Canvas.js
    - http://stackoverflow.com/questions/34403231/react-native-canvas-in-webview
