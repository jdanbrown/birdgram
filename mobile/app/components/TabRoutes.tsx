// Based on https://github.com/react-navigation/react-navigation-tabs/blob/v0.5.1/src/views/BottomTabBar.js

import _ from 'lodash';
import React, { Component, ComponentClass, PureComponent, ReactNode } from 'React';
import { ActivityIndicator, Dimensions, Platform, Text, TouchableWithoutFeedback, View } from 'react-native';
import IconBadge from 'react-native-icon-badge';
import { TabView, TabBar, SceneMap } from 'react-native-tab-view';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import Feather from 'react-native-vector-icons/Feather';
import { Link, matchPath, Redirect, Route, RouteProps, Switch } from 'react-router-native';

import { Log, puts, rich } from 'app/log';
import { memoizeOne, memoizeOneDeep } from 'app/memoize';
import { getOrientation, matchOrientation, Orientation } from 'app/orientation';
import { Histories, History, HistoryConsumer, Location, ObserveHistory, RouterWithHistory, TabName } from 'app/router';
import { Styles } from 'app/styles';
import { StyleSheet } from 'app/stylesheet';
import {
  ifUndefined, json, pretty, ix, shallowDiffPropsState, Style, throw_, typed, yaml, yamlPretty,
} from 'app/utils';

const log = new Log('TabRoutes');

//
// TabRoutes
//

export interface Props {
  tabLocation: Location; // Location to select tab (for the global tab router)
  histories: Histories;
  routes: Array<TabRoute>;
  defaultPath: string;
  priorityTabs: Array<TabRouteKey>;
}

export interface State {
  orientation: 'portrait' | 'landscape';
  shouldLoad: {[key in TabRouteKey]?: boolean}; // For lazy load
}

export interface TabRoute {
  key: TabRouteKey;
  route: TabRouteRoute;
  label: string;
  iconName: string;
  badge: ReactNode;
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
  location: Location; // Location to select view within tab (one per each tab's router)
  history: History;
  histories: Histories;
}

export type TabRouteKey = TabName;

export class TabRoutes extends PureComponent<Props, State> {

  state: State = {
    orientation: getOrientation(),
    shouldLoad: {},
  };

  // Getters for props/state
  routeByKey = (tab: TabRouteKey): TabRoute => this._routeByKey(this.props.routes)(tab);
  _routeByKey = memoizeOne(
    (routes: Array<TabRoute>): (tab: TabRouteKey) => TabRoute => {
      const m = new Map(routes.map<[TabRouteKey, TabRoute]>(route => [route.key, route]));
      return tab => m.get(tab) || throw_(`Unknown tab: ${tab}`);
    }
  );

  componentDidMount = async () => {
    log.info('componentDidMount');
    this.updateLoaded();
  }

  componentWillUnmount = async () => {
    log.info('componentWillUnmount');
  }

  componentDidUpdate = async (prevProps: Props, prevState: State) => {
    log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
    this.updateLoaded();
  }

  // Update which tabs have been loaded, for lazy load
  updateLoaded = () => {
    this.setState((state, props) => {
      const {key} = this.matchedRoute(props.routes, props.tabLocation);
      if (state.shouldLoad[key]) {
        // Tab already loaded -> noop
        return null;
      } else if (!_.isEmpty(props.priorityTabs) && !props.priorityTabs.includes(key)) {
        log.info('updateLoaded: Loading non-priority tab -> loading all tabs', {key});
        return {shouldLoad: _.fromPairs(props.routes.map(route => [route.key, true]))};
      } else {
        log.info('updateLoaded: Loading tab', {key});
        return {shouldLoad: {...state.shouldLoad, [key]: true}};
      }
    });
  }

  render = () => {
    log.info('render');
    return (
      <View style={{flex: 1}}>

        {/* NOTE Don't use Switch, else pager+screen components get unmounted/remounted on redir */}
        {/*   - This is because Switch only renders matching Route's, and unrendered components get unmounted */}
        {/* <Switch> */}

        <Route exact path='/' component={this.RedirectToDefaultPath} />

        {/* NOTE Don't warn on unknown route, else we'd warn on every redirect (because we can't use Switch -- see above) */}
        {/* {this.matchedRouteIndex(this.props.routes, this.props.tabLocation) === -1 && (
          <Warn msg={`No route for tabLocation[${yaml(this.props.tabLocation)}]`}>
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
          onIndexChange={index => {}} // WARNING Ignored b/c <TabBar/> would call it, but we override it in renderTabBar
          navigationState={{
            index: this.matchedRouteIndex(this.props.routes, this.props.tabLocation),
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
          // renderPager={...} // To customize the pager (e.g. override props)
          renderTabBar={this.TabBarLikeIOS}
          renderScene={({
            // SceneRendererProps [more fields in here]
            layout, // {height, width, measured}
            // Scene<T extends {key: string}>
            route, // T
            focused,
            index,
          }) => {
            // log.debug('renderScene', json({tabLocation: this.props.tabLocation, index, focused, route: route.key})); // XXX Debug
            return (
              <RouterWithHistory history={this.props.histories[route.key]}>
                <HistoryConsumer children={({location, history}) => (
                  // Lazy load: don't render a tab until we navigate to it
                  //  - The primary motivation here is to make launch app -> RecordScreen as fast as possible
                  !this.state.shouldLoad[route.key] ? (
                    // Loading spinner
                    <View style={{flex: 1, justifyContent: 'center'}}>
                      <ActivityIndicator size='large' />
                    </View>
                  ) : (
                    // Render tab
                    this.routeByKey(route.key).render({
                      key:         route.key,
                      // tabLocation: ...    // Location to select tab [XXX Omit, else all screens render on tab change]
                      location:    location, // Location to select view within tab
                      history:     history,
                      histories:   this.props.histories,
                    })
                  )
                )}/>
              </RouterWithHistory>
            );
          }}
        />

      </View>
    );
  }

  RedirectToDefaultPath = () => (
    <Redirect to={this.props.defaultPath} />
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
              badge={route.badge}
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

  matchedRoute = (routes: Array<TabRoute>, tabLocation: Location): TabRoute => {
    return ix(routes, this.matchedRouteIndex(routes, tabLocation)) || (
      throw_(`TabRoutes.matchedRoute: No routes for tabLocation[${json(tabLocation)}]`)
    )
  }

  // NOTE O(n), but n is very small (~5 tabs)
  matchedRouteIndex = (routes: Array<TabRoute>, tabLocation: Location): number => {
    return _.findIndex(routes, route => routeMatchesLocation(route.route, tabLocation));
  }

}

export function routeMatchesLocation(route: TabRouteRoute, tabLocation: Location): boolean {
  return !!matchPath(tabLocation.pathname, route);
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
  badge: ReactNode;
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
      <IconBadge
        // https://github.com/yanqiw/react-native-icon-badge
        //  - Put IconBadge inside of Link so that the badge doesn't block the tap area

        // Icon
        MainViewStyle={[
          styles.fill, styles.center, // Else Link's fill/center shrinkwraps our height/width
        ]}
        MainElement={(
          <Feather
            style={{
              color: props.focused ? TabBarStyle.activeTintColor : TabBarStyle.inactiveTintColor,
            }}
            name={props.iconName}
            size={size}
          />
        )}

        // Badge
        Hidden={!props.badge}
        IconBadgeStyle={{
          backgroundColor: 'inherit',
          // Helpful example of how to set {top,bottom,height} and {left,right,width} with position:absolute
          //  - https://stackoverflow.com/a/44488046/397334
          // position: 'absolute', // Default
          top: 'auto', bottom: -1, height: 'auto',
          left: 'auto', right: 'auto',
        }}
        BadgeElement={(
          <Text style={{
            ...material.captionObject,
            fontSize: 9,
            color: props.focused ? TabBarStyle.activeTintColor : TabBarStyle.inactiveTintColor,
            fontWeight: 'bold',
          }}>
            {props.badge}
          </Text>
        )}

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
    backgroundColor: Styles.tabBar.backgroundColor,
    borderTopWidth: Styles.tabBar.borderTopWidth,
    borderTopColor: Styles.tabBar.borderTopColor,
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
