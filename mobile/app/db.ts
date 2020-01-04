import _ from 'lodash';
import RNFB from 'rn-fetch-blob';
const fs = RNFB.fs;

import { Has_f_preds, matchSource, Rec, SearchRecs, Source, SourceId, UserRec, XCRec } from 'app/datatypes';
import { Log, puts, rich, tap } from 'app/log';
import { NativeSearch } from 'app/native/Search';
import Sound from 'app/sound';
import { querySql, QuerySql, sqlf } from 'app/sql';
import * as SQLite from 'app/sqlite-async';
import { local, matchNull, typed } from 'app/utils';

const log = new Log('DB');

export class DB {

  constructor(
    public sqlite:   SQLite.Database,
    public filename: string,
  ) {}

  static newAsync = async (
    filename: string = SearchRecs.dbPath,
  ): Promise<DB> => {
    const absolutePath = `${fs.dirs.MainBundleDir}/${filename}`;
    if (!await fs.exists(absolutePath)) {
      throw `DB file not found: ${absolutePath}`;
    }
    const sqlite = await SQLite.openDatabase({

      // TODO TODO This works! (with a small vendored patch)
      //  - TODO TODO Test on US build to make sure it doesn't segfault like Mon Mar 11 (see notes/birdgram.md)
      name: absolutePath,
      location: 'default', // This is ignored when name is an abs path

    });
    return new DB(
      sqlite,
      filename,
    );
  }

  // Wrap: db.querySql = querySql(db.sqlite)
  query: QuerySql = querySql(this.sqlite);

  // Returns null if audio file for source doesn't exist (e.g. user deleted a user rec, or xc dataset changed)
  loadRec = async (source: Source): Promise<Rec & Has_f_preds | null> => {
    log.info('loadRec', {source});
    const sourceId = Source.stringify(source);
    return await matchSource<Promise<Rec & Has_f_preds | null>>(source, {
      xc: async source => {

        // Read xc rec from db search_recs (includes f_preds as cols)
        return await this.query<XCRec>(sqlf`
          select *
          from search_recs
          where source_id = ${sourceId}
          limit 1
        `)(async results => {
          const rows = results.rows.raw();
          if (rows.length === 0) {
            // Return null if no rows found (e.g. xc dataset changed)
            return null;
          } else {
            const rec = rows[0];

            // Add js-friendly .f_preds array i/o sql-friendly .f_preds_* cols
            //  - TODO Perf: build .f_preds from .f_preds_* cols i/o reading audio file
            //    - Careful with col ordering -- cf. SearchScreen.componentDidMount
            const {f_preds} = await this.predsFromAudioFile(source, XCRec.audioPath(rec));

            return typed<XCRec & Has_f_preds>({
              ...rec,
              f_preds,
            });

          }
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
          const {f_preds, duration_s} = await this.predsFromAudioFile(source, UserRec.audioPath(source));

          // Make UserRec
          return typed<UserRec & Has_f_preds>({
            // UserRec
            kind:                  'user',
            f_preds,
            source,
            // Rec:bubo
            source_id:             sourceId,
            duration_s,
            // Rec:xc (mock)
            //  - TODO Push these fields into XCRec and update consumers to supply unknown values for user/edit recs
            //  - TODO Dedupe with py model.constants
            species:               '_UNK',
            species_taxon_order:   '_UNK',
            species_com_name:      'Unknown',
            species_sci_name:      'Unknown',
            species_species_group: 'Unknown',
            species_family:        'Unknown',
            species_order:         'Unknown',
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

  predsFromAudioFile = async <X extends Source>(source: X, audioPath: string): Promise<{
    // Preds
    f_preds: Array<number>,
    // Audio metadata (that requires reading the file, which we're already doing)
    duration_s: number,
  }> => {

    // Params
    const sourceId = Source.stringify(source);

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
