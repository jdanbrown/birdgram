//
// HACK Rewrote loaders.js to use static imports (original loaders.js preserved below, for reference)
//

import msgpackFactory from 'msgpack5';
import safe64 from 'urlsafe-base64';
import lzstring from 'lz-string';
import lzw from 'node-lzw';

// HACK Original lzma package fails in react-native, so I replaced it with a vendored PR
//	- See README.md for details
// import lzma from 'lzma';
import * as lzma from '../../../../third-party/umireon-LZMA-JS/src/es/lzma-codec';

// Use default options (mimic original loaders.js, below)
const msgpack = msgpackFactory();

// Clients expect all of these to be async thunks
export default {
	msgpack:  async () => msgpack,
	safe64:   async () => safe64,
	lzma:     async () => lzma,
	lzstring: async () => lzstring,
	lzw:      async () => lzw,
};

//
// Original loaders.js, for reference
//

// // centralize all chunks in one file
// export default {
// 	async msgpack() {
// 		const msgpackFactory = await import(/* webpackChunkName: "msgpack" */ 'msgpack5');
// 		return msgpackFactory();
// 	},
// 	async safe64() {
// 		return await import(/* webpackChunkName: "safe64" */ 'urlsafe-base64');
// 	},
// 	async lzma() {
// 		const lzma = await import(/* webpackChunkName: "lzma" */ 'lzma');
//
// 		// this special condition is present because the web minified version has a slightly different export
// 		return lzma.compress ? lzma : lzma.LZMA;
// 	},
// 	async lzstring() {
// 		return await import(/* webpackChunkName: "lzstring" */ 'lz-string');
// 	},
// 	async lzw() {
// 		return await import(/* webpackChunkName: "lzw" */ 'node-lzw');
// 	}
// };
