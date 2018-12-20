import cheerio from 'cheerio-without-node-native';
import _ from 'lodash';
import queryString from 'query-string';

import { MetadataSpecies, Species, SpeciesCode, SpeciesMetadata } from './datatypes';
import { Log, rich } from './log';
import { NativeHttp } from './native/Http';
import {
  assert, deepEqual, dirname, global, json, match, Omit, pretty, readJsonFile,  Timer, yaml,
} from './utils';

const log = new Log('ebird');

export interface BarchartProps {
  r:    string; // Region/hotspot ID (e.g. 'CA', 'L629014', 'L629014,L468901,L7221006')
  bmo?: number; // Begin month (e.g. {bmo: 12, emo: 2} for Dec–Feb)
  emo?: number; // End month
  byr?: number; // Begin year (e.g. 2008)
  eyr?: number; // End year
}

export class Ebird {

  speciesFromSpeciesCode: Map<SpeciesCode, Species>;
  speciesMetadataFromSpecies: Map<Species, SpeciesMetadata>;

  constructor(
    public metadataSpecies: MetadataSpecies,
  ) {
    // Manually map species_code -> species
    //  - XXX after we add species_code to datasets.metadata_from_dataset, which requires rebuilding the load._metadata
    //    cache, which takes ~overnight (see metadata_from_dataset)
    this.speciesFromSpeciesCode = new Map(metadataSpecies.map<[SpeciesCode, Species]>(x => [x.species_code, x.shorthand]));
    this.speciesMetadataFromSpecies = new Map(metadataSpecies.map<[Species, SpeciesMetadata]>(x => [x.shorthand, x]));
  }

  barchartSpecies = async (props: BarchartProps): Promise<Array<Species>> => {
    return _.flatMap(await Ebird.barchartSpeciesCodes(props), species_code => {
      const species = this.speciesFromSpeciesCode.get(species_code);
      if (species === undefined) {
        log.warn(`barchartSpecies: Ignoring unknown species_code[${species_code}] ${yaml(props)}`);
        return [];
      } else {
        return [species];
      }
    });
  }

  static barchartSpeciesCodes = async (props: BarchartProps): Promise<Array<SpeciesCode>> => {
    const doc = cheerio.load(await Ebird.barchartHtml(props));
    return doc('.barChart a[data-species-code]').map((i, x) => doc(x).data('species-code')).get();
  }

  static barchartHtml = async (props: BarchartProps): Promise<string> => {
    return await NativeHttp.httpFetch(Ebird.barchartUrl(props));
  }

  static barchartUrl = (props: BarchartProps): string => {
    return `https://ebird.org/barchart?${queryString.stringify(props)}`;
  }

};
