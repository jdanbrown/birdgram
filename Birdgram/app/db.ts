import SQLite, { SQLiteDatabase } from 'react-native-sqlite-storage';
import RNFB from 'rn-fetch-blob';
const fs = RNFB.fs;

import { matchSourceId, Rec, SearchRecs, SourceId, UserRec, XCRec } from './datatypes';
import { Log, puts, rich, tap } from './log';
import { NativeSearch } from './native/Search';
import Sound from './sound';
import { querySql, QuerySql, sqlf } from './sql';

const log = new Log('DB');

export class DB {

  constructor(
    public sqlite:   SQLiteDatabase,
    public filename: string,
  ) {}

  static newAsync = async (): Promise<DB> => {
    const filename = SearchRecs.dbPath;
    if (!await fs.exists(`${fs.dirs.MainBundleDir}/${filename}`)) {
      throw `DB file not found: ${filename}`;
    }
    const createFromLocation = `~/${filename}`; // Relative to app bundle (copied into the bundle root by react-native-asset)
    const sqlite = await SQLite.openDatabase({
      name: filename,     // Just for SQLite bookkeeping, I think
      readOnly: true,     // Else it will copy the (huge!) db file from the app bundle to the documents dir
      createFromLocation, // Else readOnly will silently not work
    });
    return new DB(
      sqlite,
      filename,
    );
  }

  // Wrap: db.querySql = querySql(db.sqlite)
  query: QuerySql = querySql(this.sqlite);

  loadRec = async (sourceId: SourceId): Promise<Rec> => {
    log.info('loadRec', {sourceId});
    return await matchSourceId(sourceId, {
      xc: async ({xc_id}) => {

        // Read xc rec from db
        return await this.query<XCRec>(sqlf`
          select *
          from search_recs
          where source_id = ${sourceId}
        `)(async results => {
          const [rec] = results.rows.raw();
          return rec; // TODO XCRec
        });

      },
      user: async ({name, clip}) => {

        // Predict f_preds from audio
        //  - Audio not spectro: model uses its own f_bins=40, separate from RecordScreen.props.f_bins=80 that we use to
        //    draw while recording
        const f_preds = await log.timedAsync('loadRec: f_preds', async () => {
          return await NativeSearch.f_preds(UserRec.audioPath(sourceId));
        });
        if (f_preds === null) {
          throw `DB.loadRec: Unexpected null f_preds (audio < nperseg), for sourceId[${sourceId}]`;
        } else {

          let userRec: UserRec = {
            source_id:           sourceId,
            duration_s:          NaN, // TODO(stretch_user_rec)
            f_preds,
            // Mock the xc fields
            //  - TODO Clean up junk fields after splitting subtypes XCRec, UserRec <: Rec
            xc_id:               -1,
            species:             'unknown',
            species_taxon_order: '_UNK',
            species_com_name:    'unknown',
            species_sci_name:    'unknown',
            recs_for_sp:         -1,
            quality:             'no score',
            month_day:           '',
            place:               '',
            place_only:          '',
            state:               '',
            state_only:          '',
            recordist:           '',
            license_type:        '',
            remarks:             '',
          };

          // HACK Read duration_s from audio file
          //  - TODO Dedupe with SearchScreen.getOrAllocateSoundAsync
          await Sound.scoped<void>(UserRec.audioPath(sourceId))(async sound => {
            userRec.duration_s = sound.getDuration();
          });

          return userRec
        }

      },
    });
  }

}
