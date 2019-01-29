# json-url
- https://github.com/masotime/json-url

# Install
- `yarn add json-url` else we'd have to manually depend on its dependencies

# Why do we have to vendor?
- Had to do surgery on the lzma dep to make it work in react-native
  - Error in react-native: "Dynamic require defined at line 7; not supported by Metro"
  - Original lzma: https://github.com/LZMA-JS/LZMA-JS
  - Unmerged PR with es6 bundling: https://github.com/LZMA-JS/LZMA-JS/pull/60
  - Subtree of PR branch with code that works in react-native: https://github.com/umireon/LZMA-JS/tree/es6-bundle/src/es
  - I vendored the `src/es/` dir from that branch as `third-party/umireon-LZMA-JS/`

Repro:
```js
// With original lzma: Fails with error "Dynamic require defined at line 7; not supported by Metro"
// With vendored lzma: Succeeds and prints {"raw":9,"rawencoded":15,"compressedencoded":42,"compression":0.3571}
urlpack('lzma').stats('foo bar').then(x => console.log(JSON.stringify(x)))
```
