// HACK Edit vendored import paths: absolute -> relative
import LOADERS from '../loaders';

export default {
	pack: false,
	encode: true,
	compress: async string => Buffer.from((await LOADERS.lzstring()).compressToUint8Array(string)),
	decompress: async buffer => (await LOADERS.lzstring()).decompressFromUint8Array(buffer)
};
