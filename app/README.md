Goal:
- app: Record audio to file
- app: Show a spectrogram while recording
- either/both:
  - app: Compute ranked list
  - app: Upload file to backend + backend: Compute ranked list
- app: Show ranked list

TODO
- [x] setup exponent
- [x] create gcp project
- [ ] mock backend
- [ ] mock show result list
- [ ] mock upload recording
- [ ] mock live spectrogram
- Shunt spectrogram.js (web audio + canvas) into Exponent bubo / react-native AwesomeProject:
  - Need web <canvas> in react-native:
    - https://facebook.github.io/react-native/docs/webview.html
    - http://stackoverflow.com/questions/34403231/react-native-canvas-in-webview
    - https://github.com/lwansbrough/react-native-canvas/blob/master/Canvas.js
    - https://github.com/Flipboard/react-canvas

Happy workflow A (easy):
- hard: inspect [how to do it at all?]
- edit: vim + syntastic:flow
- console, debug, deploy: Exponent XDE
- repl: https://babeljs.io/repl
- docs: https://facebook.github.io/react-native/docs/
- docs: https://docs.getexponent.com/
- example: https://github.com/exponentjs/native-component-list

Happy workflow B (more work + more control):
- hard: deploy (via xcode + dev install / app store)
- edit: vim + syntastic:flow
- inspect, console, debug: https://nuclide.io/docs/platforms/react-native/
  - Atom: "Add Project Folder" -> project dir
  - Atom: "Nuclide React Inspector: Toggle"
  - Atom: "Nuclide React Native: Start Packager"
  - Atom: "Nuclide React Native: Start Debugging"
  - Shell: `react-native run-ios` -> wait for simulator to launch
  - Simulator: cmd-d -> "Enable Hot Reloading"
  - Simulator: cmd-d -> "Debug JS Remotely"
  - Simulator: Window -> Stay in Front
  - Atom: top right -> "Toggle new / old Nuclide Debugger UI" -> old UI for console
  - Atom: top right -> "(Debug) Open Web Inspector for the debugging frame" -> for console with autocomplete
- repl: https://babeljs.io/repl
- docs: https://facebook.github.io/react-native/docs/

Docs:
- https://facebook.github.io/react-native/docs/
Examples:
- https://github.com/facebook/react-native/tree/master/Examples
Debugging:
- https://facebook.github.io/react-native/docs/debugging.html
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
