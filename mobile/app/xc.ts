import _ from 'lodash';

import { MetadataXcIds, Species, XCRec } from 'app/datatypes';
import { DB } from 'app/db';
import { Log, rich } from 'app/log';
import { sqlf } from 'app/sql';
import {
  assert, dirname, global, json, local, match, Omit, pretty, safeParseInt, Timer, yaml,
} from 'app/utils';

const log = new Log('xc');

export class XC {

  static async newAsync(db: DB, metadataXcIds: MetadataXcIds): Promise<XC> {
    const speciesFromXCID = new Map<number, Species>(
      _.toPairs(metadataXcIds)
      .map<[number, Species]>(([k, v]) => [safeParseInt(k), v])
    );
    return new XC(
      db,
      speciesFromXCID,
    );
  }

  constructor(
    public db:              DB,
    public speciesFromXCID: Map<number, Species>,
  ) {}

}
