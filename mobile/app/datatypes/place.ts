import jsonStableStringify from 'json-stable-stringify';
import _ from 'lodash';

import { Species, SpeciesCode } from 'app/datatypes';
import { BarchartProps } from 'app/ebird';

export interface Place {
  name:            string;
  allSpeciesCodes: Array<SpeciesCode>;   // All species_code's from the ebird place (some of which we might not know)
  knownSpecies:    Array<Species>;       // Species that could be resolved by the app (in country + known mappings)
  props:           null | BarchartProps; // null to allow ebird.allPlace
}

export type PlaceId = string;

export const Place = {

  id: ({props}: Place): PlaceId => {
    return jsonStableStringify(props); // A bit verbose in our locations, but simple and sound
  },

  find: (id: PlaceId, places: Array<Place>): Place | undefined => {
    return _.find(places, place => id === Place.id(place))
  },

  countryCodeFromName: {
    'United States': 'US',
    'Mexico':        'MX',
    'Canada':        'CA',
  } as {[k: string]: string},

  stateCodeFromName: {
    // US
    'Alabama':              'AL',
    'Alaska':               'AK',
    'Arizona':              'AZ',
    'Arkansas':             'AR',
    'California':           'CA',
    'Colorado':             'CO',
    'Connecticut':          'CT',
    'Delaware':             'DE',
    'District Of Columbia': 'DC',
    'Florida':              'FL',
    'Georgia':              'GA',
    'Hawaii':               'HI',
    'Idaho':                'ID',
    'Illinois':             'IL',
    'Indiana':              'IN',
    'Iowa':                 'IA',
    'Kansas':               'KS',
    'Kentucky':             'KY',
    'Louisiana':            'LA',
    'Maine':                'ME',
    'Maryland':             'MD',
    'Massachusetts':        'MA',
    'Michigan':             'MI',
    'Minnesota':            'MN',
    'Mississippi':          'MS',
    'Missouri':             'MO',
    'Montana':              'MT',
    'Nebraska':             'NE',
    'Nevada':               'NV',
    'New Hampshire':        'NH',
    'New Jersey':           'NJ',
    'New Mexico':           'NM',
    'New York':             'NY',
    'North Carolina':       'NC',
    'North Dakota':         'ND',
    'Ohio':                 'OH',
    'Oklahoma':             'OK',
    'Oregon':               'OR',
    'Pennsylvania':         'PA',
    'Rhode Island':         'RI',
    'South Carolina':       'SC',
    'South Dakota':         'SD',
    'Tennessee':            'TN',
    'Texas':                'TX',
    'Utah':                 'UT',
    'Vermont':              'VT',
    'Virginia':             'VA',
    'Washington':           'WA',
    'West Virginia':        'WV',
    'Wisconsin':            'WI',
    'Wyoming':              'WY',
  } as {[k: string]: string},

};
