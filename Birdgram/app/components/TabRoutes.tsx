// Based on https://github.com/react-navigation/react-navigation-tabs/blob/v0.5.1/src/views/BottomTabBar.js

import _ from 'lodash';
import memoizeOne from 'memoize-one';
import React, { Component, ComponentClass, PureComponent, ReactNode } from 'React';
import { Dimensions, Platform, Text, TouchableWithoutFeedback, View } from 'react-native';
import { TabView, TabBar, SceneMap } from 'react-native-tab-view';
import Feather from 'react-native-vector-icons/Feather';
import { Link, matchPath, Redirect, Route, RouteProps, Switch } from 'react-router-native';

import { Log, puts, rich } from 'app/log';
import { getOrientation, matchOrientation, Orientation } from 'app/orientation';
import { Histories, History, HistoryConsumer, Location, ObserveHistory, RouterWithHistory, TabName } from 'app/router';
import { StyleSheet } from 'app/stylesheet';
import { json, pretty, shallowDiffPropsState, Style, throw_ } from 'app/utils';

const log = new Log('TabRoutes');

//
// TabRoutes
//

export interface TabRoutesProps {
  defaultPath?: string;
  histories: Histories;
  routes: Array<TabRoute>;
}

export interface TabRoutesState {
  orientation: 'portrait' | 'landscape';
}

export interface TabRoute {
  key: TabRouteKey;
  route: TabRouteRoute;
  label: string;
  iconName: string;
  render: (props: TabRouteProps) => ReactNode;
}

export interface TabRouteRoute {
  path: string;
  exact?: boolean;
  sensitive?: boolean;
  strict?: boolean;
}

export interface TabRouteProps {
  key: TabRouteKey;
  location: Location;
  history: History;
  histories: Histories;
}

export type TabRouteKey = TabName;

export class TabRoutes extends PureComponent<TabRoutesProps, TabRoutesState> {

  // WARNING O(n_tabs^2) on each render, but should be harmless for small n_tabs (e.g. ~5)
  routeByKey = (key: TabRouteKey): TabRoute | undefined => {
    return _.find(this.props.routes, route => route.key === key);
  }

  state = {
    orientation: getOrientation(),
  };

  componentDidMount = async () => {
    log.info('componentDidMount');
  }

  componentWillUnmount = async () => {
    log.info('componentWillUnmount');
  }

  componentDidUpdate = async (prevProps: TabRoutesProps, prevState: TabRoutesState) => {
    log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  }

  render = () => {
    log.info('render');
    return (
      <Route children={({location}) => (
        <View style={{flex: 1}}>

          {/* NOTE Don't use Switch, else pager+screen components get unmounted/remounted on redir */}
          {/*   - This is because Switch only renders matching Route's, and unrendered components get unmounted */}
          {/* <Switch> */}

          <Route exact path='/' component={this.RedirectToDefaultPath} />

          {/* NOTE Don't warn on unknown route, else we'd warn on every redirect (because we can't use Switch -- see above) */}
          {/* {this.matchedRouteIndex(location) === -1 && (
            <Warn msg={`No route for location (${pretty(location)})`}>
              <this.RedirectToDefaultPath/>
            </Warn>
          )} */}

          {/* Tabs + pager */}
          {/* FIXME Pager sometimes doesn't update to navigationState.index: */}
          {/* - Repro: quickly: Toggle 'Show debug info' -> Saved -> Recent -> observe Recent tab is blue but pager shows Saved screen */}
          <TabView
            tabBarPosition='bottom'
            swipeEnabled={false}
            animationEnabled={false}
            onIndexChange={i => {}} // Noop: index tracked implicitly through location
            navigationState={{
              index: this.matchedRouteIndex(location),
              routes: this.props.routes.map(route => ({
                key: route.key,
                label: route.label,
                iconName: route.iconName,
              })),
            }}
            initialLayout={{
              width: Dimensions.get('window').width,
              height: 0,
            }}
            renderScene={({
              // SceneRendererProps [more fields in here]
              layout, // {height, width, measured}
              // Scene<T extends {key: string}>
              route, // T
              focused,
              index,
            }) => {
              return (
                <RouterWithHistory history={this.props.histories[route.key]}>
                  <HistoryConsumer children={({location, history}) => (
                    this.routeByKey(route.key)!.render({
                      key: route.key,
                      location: location,
                      history: history,
                      histories: this.props.histories,
                    })
                  )}/>
                </RouterWithHistory>
              );
            }}
            renderTabBar={this.TabBarLikeIOS}
            // renderPager={...} // To customize the pager (e.g. override props)
          />

        </View>
      )} />
    );
  }

  RedirectToDefaultPath = () => (
    <Redirect to={this.props.defaultPath || this.props.routes[0].route.path} />
  );

  // Instead of TabView's default material-style TabBar
  TabBarLikeIOS = () => (
    <View
      style={[styles.tabBar, matchOrientation(this.state.orientation, {
        portrait:  () => styles.tabBarPortrait,
        landscape: () => styles.tabBarLandscape,
      })]}
      onLayout={this.onLayout} // For orientation
    >
      {this.props.routes.map(route => (
        <View
          key={route.key}
          style={styles.tab}
        >
          <Route children={({location}) => (
            <TabLink
              focused={routeMatchesLocation(route.route, location)}
              orientation={this.state.orientation}
              to={route.route.path}
              label={route.label}
              iconName={route.iconName}
            />
          )}/>
        </View>
      ))}
    </View>
  );

  onLayout = () => {
    this.setState({
      orientation: getOrientation(),
    });
  }

  matchedRouteIndex = (location: Location): number => {
    return _.findIndex(this.props.routes, route => routeMatchesLocation(route.route, location));
  }

  iconNameForTab = (tab: TabRouteKey): string => this._iconNameForTab(this.props.routes)(tab);
  _iconNameForTab: (routes: Array<TabRoute>) => (tab: TabRouteKey) => string = (
    memoizeOne((routes: Array<TabRoute>) => {
      const m = new Map(routes.map<[TabRouteKey, string]>(x => [x.key, x.iconName]));
      return (tab: TabRouteKey) => m.get(tab) || throw_(`Unknown tab: ${tab}`);
    })
  );

}

export function routeMatchesLocation(route: TabRouteRoute, location: Location): boolean {
  return !!matchPath(location.pathname, route);
}

// TODO How to fix type to accept no children?
export function Warn<X>(props: {msg: string, children: X}): X | null {
  log.warn('', props.msg);
  return props.children || null;
}

//
// TabLink
//

export interface TabLinkProps {
  focused: boolean;
  orientation: Orientation;
  to: string;
  label: string; // TODO
  iconName: string;
}

export function TabLink(props: TabLinkProps) {
  const size = matchOrientation(props.orientation, {
    portrait:  () => 24,
    landscape: () => 17,
  });
  return (
    <Link
      style={[styles.fill, styles.center]}
      delayPressOut={0} // Ensure response isn't delayed (default: 100 (ms))
      underlayColor={null} // No flash on press (default: black)
      to={props.to}
      replace // Replace i/o push since we're a tab view (not a stack view)
    >
      <Feather
        style={{
          color: props.focused ? TabBarStyle.activeTintColor : TabBarStyle.inactiveTintColor,
        }}
        name={props.iconName}
        size={size}
      />
    </Link>
  );
}

//
// Styles
//

// Based on https://github.com/react-navigation/react-navigation-tabs/blob/v0.5.1/src/views/BottomTabBar.js
export const TabBarStyle = {

  activeTintColor:   '#3478f6', // Default active tint color in iOS 10
  inactiveTintColor: '#929292', // Default inactive tint color in iOS 10

  portraitHeight: 49,
  landscapeHeight: 29,

}

// Based on https://github.com/react-navigation/react-navigation-tabs/blob/v0.5.1/src/views/BottomTabBar.js
const styles = StyleSheet.create({
  fill: {
    flex: 1,
    width: '100%',
    height: '100%',
  },
  center: {
    justifyContent: 'center', // Vertical
    alignItems: 'center', // Horizontal
  },
  tabBar: {
    flexDirection: 'row',
    backgroundColor: '#f7f7f7', // Default background color in iOS 10
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: 'rgba(0,0,0,.3)',
  },
  tabBarPortrait: {
    height: TabBarStyle.portraitHeight,
  },
  tabBarLandscape: {
    height: TabBarStyle.landscapeHeight,
  },
  tab: {
    flex: 1,
    justifyContent: 'center', // Vertical
    alignItems: 'center', // Horizontal
    // alignItems: Platform.OS === 'ios' ? 'center' : 'stretch', // TODO BottomTabBar did this; do we want it too?
  },
});
