import _ from 'lodash';
import React, { PureComponent } from 'react';
import { Geolocation, GeolocationReturnType, GeoOptions } from 'react-native';

import { Log, rich } from '../log';
import { into, json, local, match, pretty, shallowDiffPropsState } from '../utils';

const log = new Log('geo');

// Docs:
//  - https://developer.mozilla.org/en-US/docs/Web/API/Position
//  - https://developer.mozilla.org/en-US/docs/Web/API/Coordinates
//  - https://developer.mozilla.org/en-US/docs/Web/API/PositionError
export const geolocation = navigator.geolocation;
export type GeoCoords = GeolocationReturnType;
export type GeoResult = GeoCoords | 'permission-denied' | 'position-unavailable' | 'timeout';

export interface Props {
  // From GeoOptions
  enableHighAccuracy:    boolean;
  distanceFilter:        number;
  useSignificantChanges: boolean;
  timeout?:              number;
  maximumAge?:           number;
}

export interface State {
  coords: GeoCoords | null;
}

// Docs
//  - https://facebook.github.io/react-native/docs/geolocation
//  - https://developer.mozilla.org/en-US/docs/Web/API/Geolocation/getCurrentPosition
//  - https://hackernoon.com/react-native-basics-geolocation-adf3c0d10112
//
// TODO(android): Caveats that need investigating
//  - https://facebook.github.io/react-native/docs/geolocation
//  - https://facebook.github.io/react-native/docs/permissionsandroid.html
//  - https://hackernoon.com/react-native-basics-geolocation-adf3c0d10112
//  - https://github.com/Agontuk/react-native-geolocation-service
//
export class Geo extends PureComponent<Props, State> {

  static defaultProps = {
    // Docs: https://facebook.github.io/react-native/docs/geolocation#watchposition
    //  - Rely on distanceFilter i/o maximumAge to keep the coords up to date
    distanceFilter:        100,      // meters (default: 100)
    useSignificantChanges: true,     // Use ios api to save battery (ios only)
    // HACK Omit these to use the default `INFINITY`, else `Infinity` turns into `null` across the json bridge and barfs
    //  - What is `INFINITY` and how would I pass it explicitly?
    //    - https://facebook.github.io/react-native/docs/geolocation#watchposition
    // timeout:            INFINITY, // ms (default: INFINITY)
    // maximumAge:         INFINITY, // ms (default: INFINITY)
  };

  state = {
    coords: null,
  };

  // Internal state
  _watchId: number | null = null;

  // Getters for props/state
  get coords(): GeoCoords | null { return this.state.coords; }

  componentDidMount = async () => {
    log.info('componentDidMount');
    this.start();
  }

  componentWillUnmount = async () => {
    log.info('componentWillUnmount');
    this.stop();
  }

  componentDidUpdate = async (prevProps: Props, prevState: State) => {
    log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));

    // Recreate watch if props changed
    //  - But not if only state changed, since that happens on every notification from the watch
    if (prevProps !== this.props) {
      this.stop();
      this.start();
    }

  }

  render = () => null;

  start = () => {
    log.info('start', rich({props: this.props, state: this.state, _watchId: this._watchId}));
    if (this._watchId !== null) {
      throw `Already watching: ${this._watchId}`;
    }
    this._watchId = geolocation.watchPosition(
      coords => {
        log.debug('watchPosition', () => pretty({props: this.props, state: this.state, coords}));
        this.setState({
          coords,
        });
      },
      error => {
        log.warn('watchPosition: Error', error);
      },
      _.pick(this.props, [
        'enableHighAccuracy',
        'timeout',
        'maximumAge',
        'distanceFilter',
        'useSignificantChanges',
      ]),
    );
  }

  stop = (): void => {
    log.info('stop', rich({props: this.props, state: this.state, _watchId: this._watchId}));
    if (this._watchId !== null) {
      geolocation.clearWatch(this._watchId);
      this._watchId = null;
    }
  }

  // [Unused: Switched to watchPosition to avoid a disruptive (>1s) lag at the start of each recording]
  // static getCurrentCoords = (opts?: GeoOptions): Promise<GeoResult> => {
  //   return new Promise((resolve, reject) => {
  //     geolocation.getCurrentPosition(
  //       position => {
  //         log.debug('getCurrentCoords: Success', position);
  //         resolve(position);
  //       },
  //       error => {
  //         log.debug('getCurrentCoords: Error', error);
  //         match(error.code,
  //           [error.PERMISSION_DENIED,    () => resolve('permission-denied')],
  //           [error.POSITION_UNAVAILABLE, () => resolve('position-unavailable')],
  //           [error.TIMEOUT,              () => resolve('timeout')],
  //           [match.default,              () => reject(`Unknown error code: ${json(error)}`)],
  //         );
  //       },
  //       opts,
  //     );
  //   });
  // }

}
