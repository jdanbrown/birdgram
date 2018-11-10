import React, { Component } from 'react';
import { Linking } from 'react-native';
import { Route } from 'react-router-native';
import { History } from 'history';

import { log } from '../log';
import { urlpack } from '../urlpack';

// Open app urls (/ app links / deep links)
//  - e.g. 'birdgram-us://open/...'
//  - Docs
//    - https://facebook.github.io/react-native/docs/linking
//  - Setup
//    - ios
//      - AppDelegate.m -> https://facebook.github.io/react-native/docs/linking#basic-usage
//      - Xcode -> https://reactnavigation.org/docs/en/deep-linking.html#ios
//    - TODO android
//      - https://reactnavigation.org/docs/en/deep-linking.html#android
//      - https://facebook.github.io/react-native/docs/linking#basic-usage

export interface DeepLinkingProps {
  prefix: string;
  action: 'push' | 'replace';
}

export class DeepLinking extends Component<DeepLinkingProps> {

  _history?: History;
  _listeners: {[key: string]: any} = {};

  componentDidMount = async () => {
    log.info(`${this.constructor.name}.componentDidMount`);

    // Handle app urls opened while app is already running
    this._listeners.url = Linking.addEventListener('url', async ({url}) => {
      log.info('DeepLinking._listeners.url: Opening url', {url});
      await this.openUrl(url);
    });

    // Handle app urls opened when app was not yet running (and caused app to launch)
    const initialUrl = await Linking.getInitialURL();
    if (initialUrl) {
      log.info('DeepLinking.componentDidMount: Opening initialUrl', {initialUrl});
      await this.openUrl(initialUrl);
    }

  }

  componentWillUnmount = async () => {
    log.info(`${this.constructor.name}.componentWillUnmount`);
    Linking.removeEventListener('url', this._listeners.url);
  }

  render = () => (
    <Route children={({history}) => {
      this._history = history;
      // return null;
      return this.props.children || null;
    }}/>
  );

  openUrl = async (url: string) => {
    log.info('DeepLinking.openUrl', {url});

    // TODO Might need to match prefix differently on android vs. ios
    //  - e.g. 'scheme://' (ios) vs. 'scheme://authority' (android) [https://reactnavigation.org/docs/en/deep-linking.html]
    const path = url.replace(new RegExp(`^${this.props.prefix}`), '');
    this._history![this.props.action](path);

    // TODO(deep_link)
    //  - 'birdgram-us://open/u/:tinyid' -> 'https://tinyurl.com/:tinyid' -> 'birdgram-us/open/:screen/:params'
    // log.info(await urlpack('lzma').stats(bigObject)); // XXX Dev [slow, ~2-5s for SearchScreen.state; why slower in Release vs Debug?]

  }

}
