import _ from 'lodash';

import { Species, XCRec } from 'app/datatypes';
import { DB } from 'app/db';
import { Log, rich } from 'app/log';
import { sqlf } from 'app/sql';
import {
  assert, deepEqual, dirname, global, json, local, match, Omit, pretty, readJsonFile,  Timer, yaml,
} from 'app/utils';

const log = new Log('xc');

export class XC {

  static async newAsync(db: DB): Promise<XC> {
    const speciesFromXCID = await local(async () => {
      const rows = await db.query<Pick<XCRec, 'xc_id' | 'species'>>(sqlf`
        select xc_id, species
        from search_recs
      `)(async results => results.rows.raw());
      return new Map(rows.map<[number, Species]>(x => [x.xc_id, x.species]));
    });
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
