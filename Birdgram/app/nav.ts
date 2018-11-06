import { NavigationRoute, NavigationScreenProp } from 'react-navigation';

import { ModelsSearch, ServerConfig } from './datatypes';
import { log } from './log';
import { Settings } from './settings';
import { pretty } from './utils';

export type Nav = NavigationScreenProp<NavigationRoute<NavParams>, NavParams>;

// QUESTION How should we use ScreenProps vs. NavParams?
export interface ScreenProps {
  serverConfig: ServerConfig;
  modelsSearch: ModelsSearch;
  settings: Settings;
}

// Wrappers for nav.navigate(route, params) to increase type safety
//  - Any object is a NavParams b/c all fields are optional, so typos don't trigger type errors
export const navigate = {

  record:   (nav: Nav, x: NavParamsRecord)   : boolean => navigate._navigate(nav, 'Record',   {record:   x}),
  search:   (nav: Nav, x: NavParamsSearch)   : boolean => navigate._navigate(nav, 'Search',   {search:   x}),
  recent:   (nav: Nav, x: NavParamsRecent)   : boolean => navigate._navigate(nav, 'Recent',   {recent:   x}),
  saved:    (nav: Nav, x: NavParamsSaved)    : boolean => navigate._navigate(nav, 'Saved',    {saved:    x}),
  settings: (nav: Nav, x: NavParamsSettings) : boolean => navigate._navigate(nav, 'Settings', {settings: x}),

  _navigate: (nav: Nav, routeName: string, params: NavParams): boolean => {
    log.info('[navigate]', routeName, pretty(params));
    return nav.navigate(routeName, params);
  },

};

// Shared navigation.params datatype across all navigation screens
//  - Problem: nav.navigate(screen, params) does shallow merge with existing params
//  - Solution: isolate params per screen
//    - [What are best practices here?]
export interface NavParams {
  record?:   NavParamsRecord;
  search?:   NavParamsSearch;
  recent?:   NavParamsRecent;
  saved?:    NavParamsSaved;
  settings?: NavParamsSettings;
}

export interface NavParamsRecord {
}

export interface NavParamsSearch {
  species?: string;
  recId?: string;
}

export interface NavParamsRecent {
}

export interface NavParamsSaved {
}

export interface NavParamsSettings {
}
