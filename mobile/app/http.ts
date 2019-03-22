import _ from 'lodash';

import { debug_print, Log, rich } from 'app/log';
import { NativeHttp } from 'app/native/Http';
import {
  assert, dirname, global, json, mapUndefined, match, Omit, pretty, readJsonFile, Timer, yaml,
} from 'app/utils';

const log = new Log('http');

export const http = {

  fetch: async (url: string): Promise<string> => {
    return log.timedAsync(`fetch: url[${url}]`, async () => {
      return await NativeHttp.httpFetch(url);
    });
  },

};
