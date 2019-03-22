import { EventEmitter } from 'fbemitter';
import _ from 'lodash';
import { Geolocation, GeolocationError, GeolocationReturnType, GeoOptions } from 'react-native';

import { Log, rich } from 'app/log';
import { into, json, local, match, Omit, pretty, shallowDiffPropsState } from 'app/utils';

const log = new Log('geo');

// Docs:
//  - https://developer.mozilla.org/en-US/docs/Web/API/Position
//  - https://developer.mozilla.org/en-US/docs/Web/API/Coordinates
//  - https://developer.mozilla.org/en-US/docs/Web/API/PositionError
export const geolocation = navigator.geolocation;
export type GeoCoords = GeolocationReturnType;
export type GeoError  = GeolocationError; // XXX Debug
export type GeoResult = GeoCoords | 'permission-denied' | 'position-unavailable' | 'timeout';

interface Props {
  enableHighAccuracy:    boolean;
  distanceFilter:        number;
  useSignificantChanges: boolean;
  timeout?:              number;
  maximumAge?:           number;
}

interface State {
  coords: GeoCoords | null;
}

const defaultProps = {

  // TODO Keep tuning these
  //  - Docs: https://facebook.github.io/react-native/docs/geolocation#watchposition
  //  - Tips: https://stackoverflow.com/a/41974586
  //  - To specify the default `INFINITY` (for timeout/maximumAge) you have to omit the key
  //    - Else `Infinity` turns into `null` across the json bridge and barfs
  //    - (What is `INFINITY` and how would I pass it explicitly?)
  //    - https://facebook.github.io/react-native/docs/geolocation#watchposition
  distanceFilter: 10,    // meters (default: 100, 0 to "not filter locations")
  timeout:        30000, // ms (default: INFINITY)
  maximumAge:     0,     // ms (default: INFINITY)

  // Don't use the significant-change api (ios only)
  //  - Saves power, but too low granularity: updates when "position changes by a significant amount, such as 500 meters or more"
  //  - https://developer.apple.com/documentation/corelocation/getting_the_user_s_location/using_the_significant-change_location_service
  useSignificantChanges: false,

};

type DefaultProps  = typeof defaultProps;
type RequiredProps = Omit<Props, keyof DefaultProps>;
type OptionalProps = Partial<Pick<Props, keyof DefaultProps>>;

// Docs
//  - https://facebook.github.io/react-native/docs/geolocation
//  - https://developer.mozilla.org/en-US/docs/Web/API/Geolocation/getCurrentPosition
//  - https://hackernoon.com/react-native-basics-geolocation-adf3c0d10112
//  - https://github.com/facebook/react-native/blob/v0.57.2/Libraries/Geolocation/RCTLocationObserver.m
//
// TODO(android): Caveats that need investigating
//  - https://facebook.github.io/react-native/docs/geolocation
//  - https://facebook.github.io/react-native/docs/permissionsandroid.html
//  - https://hackernoon.com/react-native-basics-geolocation-adf3c0d10112
//  - https://github.com/Agontuk/react-native-geolocation-service
//
export class Geo {

  props: Props;

  constructor(props: RequiredProps & OptionalProps) {
    this.props = {
      ...defaultProps,
      ...props,
    };
  }

  state: State = {
    coords: null,
  };

  // Internal state
  _watchId: number | null = null;

  // Events
  emitter = new EventEmitter();

  // Getters for props/state
  get coords(): GeoCoords | null { return this.state.coords; }
  get opts(): GeoOptions {
    return _.pick(this.props, [
      'enableHighAccuracy',
      'timeout',
      'maximumAge',
      'distanceFilter',
      'useSignificantChanges',
    ]);
  }

  // TODO(geo_reload): Trigger reload when settings.geoHighAccuracy changes (used to be a react component)
  //  - We stopped being a react component when we ran into ref issues (always null when we need to deref it)
  // reload = (prevProps: Props, prevState: State) => {
  //   // Recreate watch if props changed
  //   //  - But not if only state changed, since that happens on every notification from the watch
  //   if (prevProps !== this.props) {
  //     this.stop();
  //     this.start();
  //   }
  // }

  start = () => {
    log.info('start', rich({props: this.props, state: this.state, _watchId: this._watchId}));
    if (this._watchId !== null) {
      throw `Already watching: ${this._watchId}`;
    }
    this._watchId = geolocation.watchPosition(
      (coords: GeoCoords) => {
        log.debug('watchPosition: coords', rich(coords));
        this.state.coords = coords;
        this.emitter.emit('coords', coords);
      },
      (error: GeoError) => {
        log.warn('watchPosition: error', rich(error));
        this.emitter.emit('error', error);
      },
      this.opts,
    );
    log.debug('start: Watching', rich({coords: this.coords, _watchId: this._watchId}));
  }

  stop = (): void => {
    log.info('stop', rich({props: this.props, state: this.state, _watchId: this._watchId}));
    if (this._watchId !== null) {
      geolocation.clearWatch(this._watchId);
      this._watchId = null;
    }
  }

  // [Unused: Switched to watchPosition to avoid a disruptive (>1s) lag at the start of each recording]
  getCurrentCoords = (opts?: GeoOptions): Promise<GeoResult> => {
    return new Promise((resolve, reject) => {
      geolocation.getCurrentPosition(
        position => {
          log.debug('getCurrentCoords: Success', position);
          resolve(position);
        },
        error => {
          log.debug('getCurrentCoords: Error', error);
          match(error.code,
            [error.PERMISSION_DENIED,    () => resolve('permission-denied')],
            [error.POSITION_UNAVAILABLE, () => resolve('position-unavailable')],
            [error.TIMEOUT,              () => resolve('timeout')],
            [match.default,              () => reject(`Unknown error code: ${json(error)}`)],
          );
        },
        opts || this.opts,
      );
    });
  }

}
