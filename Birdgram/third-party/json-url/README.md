# json-url
- https://github.com/masotime/json-url

# Install
- `yarn add json-url` else we'd have to manually depend on its dependencies

# Why do we have to vendor?
- Error on mobile: "Dynamic require defined at line 7; not supported by Metro"

Repro:
```js
import jsonUrl from 'json-url/dist/node/index';
urlpack.jsonUrl('lzma').stats('foo') // -> error
```
