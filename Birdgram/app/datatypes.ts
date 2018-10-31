import { Settings } from './components/settings';
import { Places } from './places';

export type Quality = 'A' | 'B' | 'C' | 'D' | 'E' | 'no score';
export type RecId = string;

export type Rec = {
  id: RecId,
  xc_id: number,
  species: string,             // (From ebird)
  species_taxon_order: string, // (From ebird)
  species_com_name: string,    // (From xc)
  species_sci_name: string,    // (From xc)
  recs_for_sp: number,
  quality: Quality,
  lat: number,
  lng: number,
  month_day: string,
  place: string,
  place_only: string,
  state: string,
  state_only: string,
  recordist: string,
  license_type: string,
}

export const Rec = {

  spectroPath: (rec: Rec): string => SearchRecs.assetPath('spectro', rec.species, rec.xc_id, 'png'),
  audioPath:   (rec: Rec): string => SearchRecs.assetPath('audio',   rec.species, rec.xc_id, 'mp4'),

  placeNorm: (rec: Rec): string => {
    return rec.place.split(', ').reverse().map(x => Rec.placePartAbbrev(x)).join(', ');
  },
  placePartAbbrev: (part: string): string => {
    const ret = (
      Places.countryCodeFromName[part] ||
      Places.stateCodeFromName[part] ||
      part
    );
    return ret;
  },

};

export const SearchRecs = {

  serverConfigPath: 'search_recs/server-config.json',

  // TODO Test asset paths on android (see notes in README)
  dbPath: 'search_recs/search_recs.sqlite3',

  // TODO After verifying that asset dirs are preserved on android, simplify the basenames back to `${xc_id}.${format}`
  assetPath: (kind: string, species: string, xc_id: number, format: string): string => (
    `search_recs/${kind}/${species}/${kind}-${species}-${xc_id}.${format}`
  ),

};

export type ServerConfig = {
  server_globals: {
    sg_load: {
      search: object,
      xc_meta: {
        countries_k: string | null,
        com_names_k: string | null,
        num_recs:    number | null,
      },
    },
  },
  api: {
    recs: {
      search_recs: {
        params: {
          limit: number,
          audio_s: number,
        },
      },
      spectro_bytes: {
        format: string,
        convert?: object,
        save?: object,
      },
    },
  },
  audio: {
    audio_persist: {
      audio_kwargs: {
        format: string,
        bitrate?: string,
        codec?: string,
      },
    },
  },
};

export type ScreenProps = {
  serverConfig: ServerConfig,
  settings: Settings,
};
