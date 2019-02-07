import _ from 'lodash';
import SQLite, { SQLiteDatabase } from 'react-native-sqlite-storage';
import RNFB from 'rn-fetch-blob';
const fs = RNFB.fs;

import { matchSource, Rec, SearchRecs, Source, SourceId, UserRec, XCRec } from 'app/datatypes';
import { Log, puts, rich, tap } from 'app/log';
import { NativeSearch } from 'app/native/Search';
import Sound from 'app/sound';
import { querySql, QuerySql, sqlf } from 'app/sql';
import { local, matchNull, typed } from 'app/utils';

const log = new Log('DB');

export class DB {

  constructor(
    public sqlite:   SQLiteDatabase,
    public filename: string,
  ) {}

  static newAsync = async (
    filename: string = SearchRecs.dbPath,
  ): Promise<DB> => {
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

  // Returns null if audio file for source doesn't exist (e.g. user deleted a user rec, or xc dataset changed)
  loadRec = async (source: Source): Promise<Rec | null> => {
    log.info('loadRec', {source});
    const sourceId = Source.stringify(source);
    return await matchSource<Promise<Rec | null>>(source, {
      xc: async source => {

        // Read xc rec from db search_recs (includes f_preds as cols)
        return await this.query<XCRec>(sqlf`
          select *
          from search_recs
          where source_id = ${sourceId}
          limit 1
        `)(async results => {
          const rows = results.rows.raw();
          // Return null if no rows found (e.g. xc dataset changed)
          return rows.length === 0 ? null : rows[0]; // TODO Return XCRec i/o Rec
        });

      },
      user: async source => {

        // Return null if user audioPath not found (e.g. user deleted a user rec)
        const audioPath = UserRec.audioPath(source);
        if (!await fs.exists(audioPath)) {
          return null;
        } else {

          // Predict (and read duration_s) from audio file
          //  - TODO Push duration_s into proper metadata, for simplicity
          const {f_preds, duration_s} = await this.predsFromAudioFile(source, UserRec);

          // Make UserRec
          return typed<UserRec>({
            // UserRec
            kind:                  'user',
            f_preds,
            source,
            // Rec:bubo
            source_id:             sourceId,
            duration_s,
            // Rec:xc (mock)
            //  - TODO Push these fields into XCRec and update consumers to supply 'unknown' values for user/edit recs
            species:               'unknown',
            species_taxon_order:   '_UNK',
            species_com_name:      'unknown',
            species_sci_name:      'unknown',
            species_species_group: 'unknown',
            species_family:        'unknown',
            species_order:         'unknown',
            recs_for_sp:           -1,
            quality:               'no score',
            date:                  '',
            month_day:             '',
            year:                  -1,
            place:                 '',
            place_only:            '',
            state:                 '',
            state_only:            '',
            recordist:             '',
            license_type:          '',
            remarks:               '',
          });

        }

      },
    });
  }

  predsFromAudioFile = async <X extends Source>(source: X, RecType: {
    audioPath: (source: X) => string,
  }): Promise<{
    // Preds
    f_preds: Array<number>,
    // Audio metadata (that requires reading the file, which we're already doing)
    duration_s: number,
  }> => {

    // Params
    const sourceId  = Source.stringify(source);
    const audioPath = RecType.audioPath(source);

    // Predict f_preds from audio file
    //  - Predict from audio not spectro: our spectros use Settings.f_bins (e.g. >40) i/o model's f_bins=40
    const f_preds = await log.timedAsync('loadRec: f_preds (from audio file)', async () => {
      return await NativeSearch.f_preds(audioPath);
    });
    if (f_preds === null) {
      throw `Unexpected null f_preds (audio < nperseg), for sourceId[${sourceId}]`;
    }

    // Read duration_s from audio file
    //  - TODO Push down into NativeSearch.f_preds / reuse SearchScreen.getOrAllocateSoundAsync?
    const duration_s = await log.timedAsync('loadRec: duration_s (from audio file)', async () => {
      return await Sound.scoped<number>(audioPath)(async sound => {
        return sound.getDuration();
      });
    });

    return {f_preds, duration_s};

  }

}
