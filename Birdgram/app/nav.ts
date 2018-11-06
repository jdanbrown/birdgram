import { NavigationRoute, NavigationScreenProp } from 'react-navigation';

import { ModelsSearch, ServerConfig } from './datatypes';
import { Settings } from './settings';

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
  record:   (nav: Nav, p: NavParamsRecord)   => nav.navigate('Record',   {record:   p}),
  search:   (nav: Nav, p: NavParamsSearch)   => nav.navigate('Search',   {search:   p}),
  recent:   (nav: Nav, p: NavParamsRecent)   => nav.navigate('Recent',   {recent:   p}),
  saved:    (nav: Nav, p: NavParamsSaved)    => nav.navigate('Saved',    {saved:    p}),
  settings: (nav: Nav, p: NavParamsSettings) => nav.navigate('Settings', {settings: p}),
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
