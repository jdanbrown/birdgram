import cheerio from 'cheerio-without-node-native';
import _ from 'lodash';
import queryString from 'query-string';

import { GeoCoords } from 'app/components/Geo';
import { config } from 'app/config';
import { MetadataSpecies, Place, Species, SpeciesCode, SpeciesGroup, SpeciesMetadata } from 'app/datatypes';
import { http } from 'app/http';
import { debug_print, Log, rich } from 'app/log';
import { memoizeOne, memoizeOneDeep } from 'app/memoize';
import {
  assert, dirname, global, ifUndefined, json, mapUndefined, match, Omit, pretty, readJsonFile, Timer, typed, yaml,
} from 'app/utils';

const log = new Log('ebird');

export interface BarchartProps {
  r:    string; // Region/hotspot ID (e.g. 'CA', 'L629014', 'L629014,L468901,L7221006')
  bmo?: number; // Begin month (e.g. {bmo: 12, emo: 2} for Dec–Feb)
  emo?: number; // End month
  byr?: number; // Begin year (e.g. 2008)
  eyr?: number; // End year
}

export interface ExtraMapping<K> {
  k:   K;
  toK: K;
}

export interface RegionFindResult {
  code: string,
  name: string,
}

export interface HotspotFindResult {
  code: string,
  name: string,
}

export interface HotspotGeoResult {
  countryCode:      string;
  lat:              number;
  lng:              number;
  locID:            string;
  locId:            string;
  locName:          string;
  subnational1Code: string;
  subnational2Code: string;
}

export class Ebird {

  // TODO(ebird_api): Get our own api key!
  //  - https://ebird.org/api/keygen
  static api_key = 'jfekjedvescr' // HACK Rudely copied from ebird.org source (and we'll break when they rotate it)

  static extraMappingsForSpeciesCode: Array<ExtraMapping<SpeciesCode>> = [

    // Map changes in species_code
    {k: 'mallar3', toK: 'mallar'},
    {k: 'reevir1', toK: 'reevir'},

    // TODO Maybe these waterbirds are missing b/c no recs in xc?
    //  - I forget whether I include or exclude sp's with no xc recs...
    // {k: 'hergul', toK: TODO}, // Why missing? Herring gull [https://birdsna.org/Species-Account/bna/species/hergul]
    // {k: 'y00478', toK: TODO}, // Why missing? Iceland gull [https://birdsna.org/Species-Account/bna/species/y00478]
    // {k: 'norgan', toK: TODO}, // Why missing? Northern gannet [https://birdsna.org/Species-Account/bna/species/norgan]
    // TODO Find more...

    // XXX Have to lump/split upstream in paylods.py -- trying to do it here causes bad side effects
    //  - e.g. BrowseScreen -> multiple items with the same key error (e.g. Mallard row shows up twice, b/c mexduc -> mallar)
    // Map changes from lumps/splits
    //  - cf. py metadata.ebird.com_names_to_species
    // {k: 'mexduc', toK: 'mallar'}, // https://en.wikipedia.org/wiki/Mexican_duck

  ];

  allSpeciesMetadata:         Array<SpeciesMetadata>;
  allSpecies:                 Array<Species>;
  allSpeciesCodes:            Array<SpeciesCode>;
  allSpeciesGroups:           Array<SpeciesGroup>;
  speciesFromSpeciesCode:     Map<SpeciesCode, Species>;
  speciesMetadataFromSpecies: Map<Species, SpeciesMetadata>;
  speciesForSpeciesGroup:     Map<SpeciesGroup, Array<Species>>;

  constructor(
    public metadataSpecies: MetadataSpecies,
  ) {
    this.allSpeciesMetadata = metadataSpecies;
    this.allSpecies         = metadataSpecies.map(x => x.shorthand);
    this.allSpeciesCodes    = metadataSpecies.map(x => x.species_code);
    this.allSpeciesGroups   = _.uniq(metadataSpecies.map(x => x.species_group));
    // Manually map species_code -> species
    //  - XXX after we add species_code to datasets.metadata_from_dataset, which requires rebuilding the load._metadata
    //    cache, which takes ~overnight (see metadata_from_dataset)
    this.speciesFromSpeciesCode = Ebird.addExtraMappings(
      Ebird.extraMappingsForSpeciesCode,
      new Map(metadataSpecies.map<[SpeciesCode, Species]>(x => [
        x.species_code,
        x.shorthand,
      ]))
    );
    this.speciesMetadataFromSpecies = new Map(metadataSpecies.map<[Species, SpeciesMetadata]>(x => [
      x.shorthand,
      x,
    ]));
    this.speciesForSpeciesGroup = new Map(
      _(metadataSpecies)
      .groupBy(x => x.species_group)
      .mapValues(xs => xs.map(x => x.shorthand))
      .entries()
      .value()
    );
  }

  get allPlace(): Place {
    return {
      name:            config.env.APP_REGION,
      knownSpecies:    this.allSpecies, // All species are known for the allPlace
      allSpeciesCodes: this.allSpeciesCodes,
      props:           null,
    };
  }

  speciesGroupFromSpecies = (species: Species): SpeciesGroup | undefined => {
    return mapUndefined(this.speciesMetadataFromSpecies.get(species), x => x.species_group);
  }

  barchartSpecies = async (props: BarchartProps): Promise<{
    allSpeciesCodes: Array<SpeciesCode>,
    knownSpecies:    Array<Species>,
  }> => {
    const allSpeciesCodes = await Ebird.barchartSpeciesCodes(props);
    const knownSpecies    = _.flatMap(allSpeciesCodes, species_code => {
      const species = this.speciesFromSpeciesCode.get(species_code);
      if (species === undefined) {
        log.info(`barchartSpecies: Unknown species_code[${species_code}] ${yaml(props)}`);
        return [];
      } else {
        return [species];
      }
    });
    return {allSpeciesCodes, knownSpecies};
  }

  // Example:
  //    GET https://ebird.org/ws2.0/ref/region/find/?key=jfekjedvescr&q=calif
  //    [
  //      {
  //        "code": "US-CA",
  //        "name": "California, United States (US)"
  //      },
  //      ...
  //
  // Additions:
  //  - r:'US-CA' <- code:'US-CA'
  //
  regionFind = async (q: string): Promise<Array<RegionFindResult>> => {
    // /region/find fails (403) without key
    const enc = encodeURIComponent;
    const url = `https://ebird.org/ws2.0/ref/region/find/?key=${enc(Ebird.api_key)}&q=${enc(q)}`;
    const rep = await http.fetch(url);
    const xs  = JSON.parse(rep);
    return xs.map((x: {
      // TODO Validate
      code: string,
      name: string,
    }) => {
      return typed<RegionFindResult>(x);
    });
  }

  // Example:
  //    GET https://ebird.org/ws2.0/ref/hotspot/find?q=glen+can
  //    [
  //      {
  //        "code": "L775518,36.9243711,-111.4775848",
  //        "name": "Glen Canyon Dam overlook, Coconino, US-AZ"
  //      },
  //    ...
  //
  // Additions:
  //  - r:'L775518', lat:36.9243711, lng:-111.4775848 <- code:'L775518,36.9243711,-111.4775848'
  //
  hotspotFind = async (q: string): Promise<Array<HotspotFindResult>> => {
    // /hotspot/find ignores key, but we include it anyway for consistency
    const enc = encodeURIComponent;
    const url = `https://ebird.org/ws2.0/ref/hotspot/find/?key=${enc(Ebird.api_key)}&q=${enc(q)}`;
    const rep = await http.fetch(url);
    const xs  = JSON.parse(rep);
    return xs.map((x: {
      // TODO Validate
      code: string,
      name: string,
    }) => {
      return typed<HotspotFindResult>(x);
    });
  }

  // Example:
  //    GET https://ebird.org/ws2.0/ref/hotspot/geo?fmt=json&lat=36.9243711&lng=-111.4775848
  //    [
  //      {
  //        "countryCode": "US",
  //        "lat": 37.081932,
  //        "lng": -111.661443,
  //        "locID": "L8066715",
  //        "locId": "L8066715",
  //        "locName": "Big Water",
  //        "subnational1Code": "US-UT",
  //        "subnational2Code": "US-UT-025"
  //      },
  //      ...
  hotspotGeo = memoizeOneDeep( // Memoize results for the last coords, to avoid silly api spamming
    async (coords: GeoCoords): Promise<Array<HotspotGeoResult>> => {
      const lat = `${coords.coords.latitude}`;
      const lng = `${coords.coords.longitude}`;
      const enc = encodeURIComponent;
      const url = `https://ebird.org/ws2.0/ref/hotspot/geo?fmt=json&lat=${enc(lat)}&lng=${enc(lng)}`;
      const rep = await http.fetch(url);
      const xs  = JSON.parse(rep);
      return xs.map((x: {
        // TODO Validate
        countryCode:      string;
        lat:              number;
        lng:              number;
        locID:            string;
        locId:            string;
        locName:          string;
        subnational1Code: string;
        subnational2Code: string;
      }) => {
        return typed<HotspotGeoResult>(x);
      });
    }
  );

  // hotspotGeo = async (coords: GeoCoords): Promise<Array<HotspotGeoResult>> => {
  // }

  static barchartSpeciesCodes = async (props: BarchartProps): Promise<Array<SpeciesCode>> => {
    const doc = cheerio.load(await Ebird.barchartHtml(props));
    return doc('.barChart a[data-species-code]').map((i, x) => doc(x).data('species-code')).get();
  }

  // Example:
  //    GET https://ebird.org/barchart?r=L1006665&bmo=1&emo=12&byr=1900&eyr=2019
  static barchartHtml = async (props: BarchartProps): Promise<string> => {
    // Similar urls:
    //  - https://ebird.org/barchartData?fmt=tsv&r=L1006665&bmo=1&emo=12&byr=1900&eyr=2019
    //  - https://ebird.org/barchart?r=L1006665&bmo=1&emo=12&byr=1900&eyr=2019
    //  - https://ebird.org/hotspot/L1006665?m=9&yr=last10
    //  - The tsv (/barchartData) sounds nice, but it's no faster than the html (/barchart), and one advantage of the
    //    latter is that it includes species_code (.data-species-code) i/o having to lookup from tsv's com_name
    const url = `https://ebird.org/barchart?${queryString.stringify(props)}`;
    return await http.fetch(url);
  }

  static addExtraMappings = <K, V>(mappings: Array<ExtraMapping<K>>, xs: Map<K, V>): Map<K, V> => {
    xs = _.clone(xs); // Copy so we can mutate
    mappings.forEach(({k, toK}) => {
      const v = xs.get(toK);
      if (v !== undefined) {
        xs.set(k, v);
      } else {
        // log.info i/o throw/error/warn
        //  - e.g. CA100 would throw/error/warn on most mappings, since it's a trimmed-down species set
        //  - e.g. US would throw/error/warn on CR mappings, and vice versa
        log.info(`addExtraMappings: Ignoring unknown key: toK[${toK}], for k[${k}] in keys[${yaml(Array.from(xs.keys()).sort())}]`);
      };
    });
    return xs;
  }

}
