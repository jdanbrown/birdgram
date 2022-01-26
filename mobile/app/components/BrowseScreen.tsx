import jsonStableStringify from 'json-stable-stringify';
import _ from 'lodash';
import React, { Component, PureComponent, ReactNode, RefObject } from 'react';
import shallowCompare from 'react-addons-shallow-compare';
import RN, {
  ActivityIndicator, Dimensions, FlatList, FlexStyle, GestureResponderEvent, Image, ImageStyle,
  LayoutChangeEvent, Modal, Platform, RegisteredStyle, ScrollView, SectionList, SectionListData,
  StyleProp, Text, TextInput, TextStyle, TouchableHighlight, View, ViewStyle,
} from 'react-native';
import {
  BaseButton, BorderlessButton, BorderlessButtonProperties, LongPressGestureHandler, PanGestureHandler,
  PinchGestureHandler, RectButton, TapGestureHandler,
  // FlatList, ScrollView, Slider, Switch, TextInput, // TODO Needed?
} from 'react-native-gesture-handler';
import SearchBar from 'react-native-material-design-searchbar'
import { getStatusBarHeight } from 'react-native-status-bar-height';
import { human, iOSColors, material, materialColors, systemWeights } from 'react-native-typography'
import Feather from 'react-native-vector-icons/Feather';
import FontAwesome5 from 'react-native-vector-icons/FontAwesome5';
import { sprintf } from 'sprintf-js';

import { App, AppProps, AppState } from 'app/App';
import { HelpText, TitleBar, TitleBarWithHelp } from 'app/components/TitleBar';
import {
  ModelsSearch, matchRec, matchSearchPathParams, Place, Quality, Rec, SearchPathParams, searchPathParamsFromLocation,
  SearchRecs, ServerConfig, Source, SourceId, Species, SpeciesGroup, SpeciesMetadata, SpectroPathOpts, UserRec, XCRec,
} from 'app/datatypes';
import { Ebird } from 'app/ebird';
import { debug_print, Log, logErrors, logErrorsAsync, puts, rich, tap } from 'app/log';
import { Go, Histories, History, Location, locationKeyIsEqual, locationPathIsEqual } from 'app/router';
import { SettingsWrites } from 'app/settings';
import { StyleSheet } from 'app/stylesheet';
import { normalizeStyle, LabelStyle, labelStyles, Styles } from 'app/styles';
import {
  all, any, assert, chance, Clamp, Dim, ensureParentDir, fastIsEqual, finallyAsync, getOrSet,
  global, ifNull, into, json, local, mapMapValues, mapNull, mapUndefined, match, matchEmpty,
  matchNull, matchUndefined, noawait, objectKeysTyped, Omit, Point, pretty, QueryString, round,
  setAdd, setDiff, setToggle, shallowDiffPropsState, Style, throw_, Timer, typed, yaml, yamlPretty,
  zipSame,
} from 'app/utils';

const log = new Log('BrowseScreen');

//
// BrowseScreen
//

interface Props {
  // App globals
  visible:              boolean; // Requires shouldComponentUpdate to avoid excessive updates
  location:             Location;
  history:              History;
  histories:            Histories;
  go:                   Go;
  ebird:                Ebird;
  place:                Place;
  // Settings
  settings:             SettingsWrites;
  showHelp:             boolean;
  excludeSpecies:       Set<Species>;
  excludeSpeciesGroups: Set<SpeciesGroup>;
  unexcludeSpecies:     Set<Species>;
  compareSelect:        boolean;
}

interface State {
  searchFilter: string;
}

export class BrowseScreen extends Component<Props, State> {

  log = new Log('BrowseScreen');

  state: State = {
    searchFilter: '',
  };

  // Refs
  sectionListRef: RefObject<SectionList<Rec>> = React.createRef();

  // State
  _firstSectionHeaderHeight: number = 0; // For SectionList.scrollToLocation({viewOffset})

  componentDidMount = async () => logErrorsAsync('componentDidMount', async () => {
    this.log.info('componentDidMount');
    global.BrowseScreen = this; // XXX Debug
  });

  componentWillUnmount = async () => logErrorsAsync('componentWillUnmount', async () => {
    this.log.info('componentWillUnmount');
  });

  shouldComponentUpdate = (nextProps: Props, nextState: State): boolean => {
    log.info('shouldComponentUpdate', () => rich(shallowDiffPropsState(nextProps, nextState, this.props, this.state))); // Debug

    // Manual visible/dirty to avoid background updates (type 1: shouldComponentUpdate)
    //  - If not visible, never update
    //  - If just became visible, always update
    if (!nextProps.visible) return false;
    if (nextProps.visible && !this.props.visible) return true;

    // Else mimic PureComponent
    return shallowCompare(this, nextProps, nextState);

  }

  componentDidUpdate = async (prevProps: Props, prevState: State) => logErrorsAsync('componentDidUpdate', async () => {
    // Noisy (in xcode)
    this.log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  });

  render = () => {
    // this.log.info('render'); // Debug

    // Perf: lots of redundant computation here, but this isn't the bottleneck
    //  - Bottleneck is render for SectionList -> section/item components
    const matchesSearchFilter: (metadata: SpeciesMetadata) => boolean = log.timed('matchesSearchFilter', () => {
      const tokenize = (v: string): string[] => v.toLowerCase().replace(/[^a-z ]+/, '').split(' ').filter(x => !_.isEmpty(x));
      const searches = this.state.searchFilter.split('/').map(search => tokenize(search)).filter(x => !_.isEmpty(x));
      const ks: Array<keyof SpeciesMetadata> = [
        'shorthand',
        'sci_name',
        'com_name',
        'species_code',
        'species_group',
        // 'family', // XXX Confusing because field not visible to user (can't see why it matched)
        // 'order',  // XXX Confusing because field not visible to user (can't see why it matched)
      ];
      return (metadata: SpeciesMetadata): boolean => {
        const vs = _.flatMap(ks, k => tokenize(metadata[k]));
        return _.isEmpty(searches) || _.some(searches, search => _.every(search, term => _.some(vs, v => v.includes(term))));
      };
    });

    // Precompute sections so we can figure out various indexes
    type Section = SectionListData<SpeciesMetadata>;
    const data = log.timed('data', () => typed<SpeciesMetadata[]>(
      _(this.props.place.knownSpecies)
      .flatMap(s => matchUndefined(this.props.ebird.speciesMetadataFromSpecies.get(s), {
        undefined: () => [], // Should never happen, but degrade gracefully if it does
        x:         m  => [m],
      }))
      .filter(m => matchesSearchFilter(m))
      .sortBy(m => parseFloat(m.taxon_order))
      .value()
    ));
    const sections: Array<Section> = log.timed('sections', () => (
      _(data)
      .groupBy(m => m.species_group)
      .entries().map(([title, data]) => ({title, data}))
      .value()
    ));
    const firstSection   = _.head(sections);
    const lastSection    = _.last(sections);
    const isFirstSection = (section: Section) => firstSection && section.title === firstSection.title;
    const isLastSection  = (section: Section) => lastSection  && section.title === lastSection.title;
    const isLastItem     = (section: Section, index: number) => isLastSection(section) && index === section.data.length - 1;

    // Precompute state for compareSelect
    const searchSelectedForCompare: Set<string> = new Set(
      this.searchSearchPathParamss().map(x => jsonStableStringify(x))
    );

    return (
      <View style={{
        flex: 1,
      }}>

        {/* NOTE BaseButton b/c TouchableWithoutFeedback wouldn't trigger onPress during FlatList scroll animation */}
        <BaseButton onPress={() => {
          mapNull(this.sectionListRef.current, sectionList => { // Avoid transient nulls [why do they happen?]
            if (sectionList.scrollToLocation) { // (Why typed as undefined? I think only for old versions of react-native?)
              sectionList.scrollToLocation({
                sectionIndex: 0, itemIndex: 0,              // First section, first item
                viewOffset: this._firstSectionHeaderHeight, // Else first item covered by first section header
              });
            }
          });
        }}>
          <TitleBarWithHelp
            // title='Species'
            // title='Search for species'
            title='Browse species'
            settings={this.props.settings}
            showHelp={this.props.showHelp}
            help={(
              // TODO TODO
              <HelpText>
                • foo <Feather name='shuffle'/> bar {'\n'}
                • baz
              </HelpText>
            )}
          />
        </BaseButton>

        <View style={{
          flex: 1,
        }}>

          {/* Title + scroll to top + buttons */}
          <BaseButton
            style={{
              // flex: 1,
            }}
            onPress={() => {
              // Scroll to top
              mapNull(this.sectionListRef.current, sectionList => { // Avoid transient nulls [why do they happen?]
                if (sectionList.scrollToLocation) { // (Why typed as undefined? I think only for old versions of react-native?)
                  sectionList.scrollToLocation({
                    sectionIndex: 0, itemIndex: 0,              // First section, first item
                    viewOffset: this._firstSectionHeaderHeight, // Else first item covered by first section header
                  });
                }
              });
            }}
          >
            <View style={{
              flexDirection:     'row',
              alignItems:        'center', // Vertical (row i/o column)
              backgroundColor:   Styles.tabBar.backgroundColor,
              borderBottomWidth: Styles.tabBar.borderTopWidth,
              borderBottomColor: Styles.tabBar.borderTopColor,
            }}>

              {/* Title */}
              <Text style={{
                flex: 1,
                ...material.body2Object,
                marginHorizontal: 5,
              }}>
                {/* Place: */}
                Location: {}
                {this.props.place.name} ({data.length}/{this.props.place.knownSpecies.length} species)
              </Text>

              {/* Compare button */}
              {local(() => {
                return (
                  <BrowseItemButton
                    iconName='copy'
                    active={this.props.compareSelect}
                    activeButtonColor={iOSColors.blue}
                    onPress={() => this.props.settings.toggle('compareSelect')}
                  />
                );
              })}

              {/* Toggle-all button */}
              {local(() => {
                const {allSpeciesGroups} = this.props.ebird;
                const noneExcluded = (true
                  && this.props.excludeSpecies.size       === 0
                  && this.props.excludeSpeciesGroups.size === 0
                  && this.props.unexcludeSpecies.size     === 0
                );
                const allExcluded = (true
                  && this.props.excludeSpecies.size       === 0
                  && this.props.excludeSpeciesGroups.size === allSpeciesGroups.length // HACK Briefly breaks on ebird metadata update
                  && this.props.unexcludeSpecies.size     === 0
                );
                return (
                  <BrowseItemButton
                    iconName={!noneExcluded ? 'eye-off' : 'eye'}
                    active={!noneExcluded}
                    activeButtonColor={allExcluded ? iOSColors.red : iOSColors.yellow}
                    onPress={() => this.props.settings.set(!noneExcluded ? {
                      excludeSpecies:       new Set(),
                      excludeSpeciesGroups: new Set(),
                      unexcludeSpecies:     new Set(),
                    } : {
                      excludeSpecies:       new Set(),
                      excludeSpeciesGroups: new Set(this.props.ebird.allSpeciesGroups),
                      unexcludeSpecies:     new Set(),
                    })}
                  />
                );
              })}

            </View>
          </BaseButton>

          {/* Search bar */}
          <SearchBar // (cf. SearchBar in PlacesScreen)
            // Listeners
            onSearchChange={searchFilter => this.setState({
              searchFilter,
            })}
            // Style
            height={40}
            padding={0}
            inputStyle={{
              // Disable border from SearchBar (styles.searchBar)
              borderWidth:       0,
              // Replace with a border that matches the title bar border
              backgroundColor:   Styles.tabBar.backgroundColor,
              borderBottomWidth: Styles.tabBar.borderTopWidth,
              borderBottomColor: Styles.tabBar.borderTopColor,
            }}
            // Disable back button
            //  - By: always showing back button and making it look and behave like the search icon
            alwaysShowBackButton={true}
            iconBackComponent={(<Feather size={18} color={iOSColors.gray} name={'search'} />)}
            onBackPress={() => {}}
            // TextInputProps
            inputProps={{
              autoCorrect:                   false,
              autoCapitalize:                'none',
              // enablesReturnKeyAutomatically: true,
              placeholder:                   'Species',
              defaultValue:                  this.state.searchFilter,
              returnKeyType:                 'done',
              selectTextOnFocus:             true,
              keyboardType:                  'default',
            }}
            // TODO Prevent dismissing keyboard on X button, so that it only clears the input
            iconCloseComponent={(<View/>)} // Disable close button [TODO Nope, keep so we can easily clear text]
            // onClose={() => this.setState({searchFilter: ''})} // FIXME Why doesn't this work?
          />

          {/* SectionList */}
          <SectionList
            ref={this.sectionListRef as any} // HACK Is typing for SectionList busted? Can't make it work
            style={{
              flexGrow: 1,
            }}
            sections={sections}
            // Disable lazy loading, else fast scrolling down hits a lot of partial bottoms before the real bottom
            initialNumToRender={data.length}
            maxToRenderPerBatch={data.length}
            keyExtractor={species => species.shorthand} // [Why needed in addition to key props below? key warning without this]
            ListEmptyComponent={(
              <View style={[Styles.center, {padding: 30}]}>
                <Text style={material.subheading}>
                  No species
                </Text>
              </View>
            )}
            // Perf: split out section/item components so that it stays mounted across updates
            //  - (Why does it sometimes unmount/mount anyway?)
            renderSectionHeader={({section}) => {
              const {species_group} = section.data[0];
              const searchPathParams: SearchPathParams = {kind: 'species_group', filters: {}, species_group};
              return (
                <BrowseSectionHeader
                  key={species_group}
                  species_group={species_group}
                  isFirstSection={isFirstSection(section)}
                  excluded={this.props.excludeSpeciesGroups.has(species_group)}
                  // TODO(exclude_invariants): Dedupe with SearchScreen
                  unexcludedAny={
                    Array.from(this.props.unexcludeSpecies)
                    .map(x => this.props.ebird.speciesGroupFromSpecies(x))
                    .includes(species_group)
                  }
                  compareSelect={this.props.compareSelect}
                  searchPathParams={searchPathParams}
                  isSelectedForCompare={
                    this.props.compareSelect && // Short-circuit (small perf)
                    searchSelectedForCompare.has(jsonStableStringify(searchPathParams))
                  }
                  browse={this}
                  go={this.props.go}
                  ebird={this.props.ebird}
                  settings={this.props.settings}
                />
              );
            }}
            renderItem={({index, section, item: {
              shorthand: species,
              species_group,
              com_name,
              sci_name,
            }}) => {
              const searchPathParams: SearchPathParams = {kind: 'species', filters: {}, species};
              return (
                <BrowseItem
                  key={species}
                  species={species}
                  species_group={species_group}
                  com_name={com_name}
                  sci_name={sci_name}
                  isLastItem={isLastItem(section, index)}
                  excluded={this.props.excludeSpecies.has(species) || this.props.excludeSpeciesGroups.has(species_group)}
                  unexcluded={this.props.unexcludeSpecies.has(species)}
                  compareSelect={this.props.compareSelect}
                  searchPathParams={searchPathParams}
                  isSelectedForCompare={
                    this.props.compareSelect && // Short-circuit (small perf)
                    searchSelectedForCompare.has(jsonStableStringify(searchPathParams))
                  }
                  browse={this}
                  go={this.props.go}
                  ebird={this.props.ebird}
                  settings={this.props.settings}
                />
              );
            }}
          />

        </View>
      </View>
    );

  };

  onFirstSectionHeaderLayout = async (event: LayoutChangeEvent) => {
    const {nativeEvent: {layout: {x, y, width, height}}} = event; // Unpack SyntheticEvent (before async)
    this._firstSectionHeaderHeight = height;
  }

  searchSearchPathParamss = (): Array<SearchPathParams> => {
    const searchSearchPathParams = searchPathParamsFromLocation(this.props.histories.search.location);
    return matchSearchPathParams(searchSearchPathParams, {
      root:          x                     => [x],
      random:        x                     => [x],
      species_group: x                     => [x],
      species:       x                     => [x],
      rec:           x                     => [x],
      compare:       ({searchPathParamss}) => searchPathParamss,
    });
  }
  addSearchSearchPathParamss = (searchPathParams: SearchPathParams): Array<SearchPathParams> => {
    return [...this.searchSearchPathParamss(), searchPathParams];
  }
  delSearchSearchPathParamss = (searchPathParams: SearchPathParams): Array<SearchPathParams> => {
    return this.searchSearchPathParamss().filter(x => !fastIsEqual(x, searchPathParams));
  }

}

//
// BrowseSectionHeader
//  - Split out component so that it stays mounted across updates
//

interface BrowseSectionHeaderProps {
  species_group:        string;
  isFirstSection:       boolean | undefined;
  excluded:             boolean;
  unexcludedAny:        boolean;
  compareSelect:        boolean;
  searchPathParams:     SearchPathParams;
  isSelectedForCompare: boolean;
  browse:               BrowseScreen;
  go:                   Go;
  ebird:                Ebird;
  settings:             SettingsWrites;
}

interface BrowseSectionHeaderState {
}

export class BrowseSectionHeader extends PureComponent<BrowseSectionHeaderProps, BrowseSectionHeaderState> {

  log = new Log(`BrowseSectionHeader[${this.props.species_group}]`);

  state: BrowseSectionHeaderState = {
  };

  componentDidMount = async () => logErrorsAsync('componentDidMount', async () => {
    this.log.info('componentDidMount');
  });

  componentWillUnmount = async () => logErrorsAsync('componentWillUnmount', async () => {
    this.log.info('componentWillUnmount');
  });

  componentDidUpdate = async (
    prevProps: BrowseSectionHeaderProps,
    prevState: BrowseSectionHeaderState,
  ) => logErrorsAsync('componentDidUpdate', async () => {
    this.log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  });

  render = () => {
    // this.log.info('render'); // Debug
    const {species_group, ebird} = this.props;
    return (
      <RectButton
        style={[Styles.fill, {
          backgroundColor: iOSColors.lightGray,
        }]}
        onPress={() => {
          this.props.go('search', {path: `/species_group/${encodeURIComponent(species_group)}`});
        }}
      >
        <View
          style={[Styles.fill, {
            flexDirection:   'row',
            justifyContent:  'center',
            alignItems:      'center',
            paddingVertical: 3,
          }]}
          // For SectionList.scrollToLocation({viewOffset})
          onLayout={!this.props.isFirstSection ? undefined : this.props.browse.onFirstSectionHeaderLayout}
        >

          <View style={{
            flexGrow:          1,
            flexDirection:     'row',
            alignItems:        'center',
            paddingHorizontal: 5,
          }}>
            <Text style={[material.captionObject, {fontWeight: 'bold', color: '#444444'}]}>
              {species_group}
            </Text>
          </View>

          {this.props.compareSelect && (
            <BrowseItemButton
              iconName='copy'
              active={this.props.isSelectedForCompare}
              activeButtonColor={iOSColors.blue}
              onPress={() => {
                this.props.go('search', {path: `/compare/${
                  _(this.props.isSelectedForCompare
                    ? this.props.browse.delSearchSearchPathParamss(this.props.searchPathParams)
                    : this.props.browse.addSearchSearchPathParamss(this.props.searchPathParams)
                  )
                  .map(x => encodeURIComponent(json(x)))
                  .join(',')
                }`});
              }}
            />
          )}

          <BrowseItemButton
            iconName={this.props.excluded ? 'eye-off' : 'eye'}
            active={this.props.excluded}
            activeButtonColor={!this.props.unexcludedAny ? iOSColors.red : iOSColors.yellow}
            onPress={() => {
              // TODO(exclude_invariants): Dedupe with SearchScreen
              this.props.settings.set(({excludeSpecies: exS, excludeSpeciesGroups: exG, unexcludeSpecies: unS}) => {
                const unAny = this.props.unexcludedAny;
                const g     = species_group;
                const ss    = ebird.speciesForSpeciesGroup.get(g) || []; // (Degrade gracefully if g is somehow unknown)
                if      (!exG.has(g)          ) { exG = setAdd  (exG, g); exS = setDiff (exS, ss); } // !exG         -> exG+g, exS-ss
                else if ( exG.has(g) && !unAny) { exG = setDiff (exG, g);                          } //  exG, !unAny -> exG-g
                else if ( exG.has(g) &&  unAny) { exG = setDiff (exG, g); unS = setDiff (unS, ss); } //  exG,  unAny -> exG-g, unS-ss
                return {excludeSpecies: exS, excludeSpeciesGroups: exG, unexcludeSpecies: unS};
              });
            }}
          />

        </View>
      </RectButton>
    );
  };

}

//
// BrowseItem
//  - Split out component so that it stays mounted across updates
//

interface BrowseItemProps {
  species:              Species;
  species_group:        SpeciesGroup;
  com_name:             string;
  sci_name:             string;
  isLastItem:           boolean | undefined;
  excluded:             boolean;
  unexcluded:           boolean;
  compareSelect:        boolean;
  searchPathParams:     SearchPathParams;
  isSelectedForCompare: boolean;
  browse:               BrowseScreen;
  go:                   Go;
  ebird:                Ebird;
  settings:             SettingsWrites;
}

interface BrowseItemState {
}

export class BrowseItem extends PureComponent<BrowseItemProps, BrowseItemState> {

  log = new Log(`BrowseItem[${this.props.species_group}/${this.props.species}]`);

  state: BrowseItemState = {
    searchFilter: '',
  };

  componentDidMount = async () => logErrorsAsync('componentDidMount', async () => {
    this.log.info('componentDidMount');
  });

  componentWillUnmount = async () => logErrorsAsync('componentWillUnmount', async () => {
    this.log.info('componentWillUnmount');
  });

  componentDidUpdate = async (
    prevProps: BrowseItemProps,
    prevState: BrowseItemState,
  ) => logErrorsAsync('componentDidUpdate', async () => {
    this.log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  });

  render = () => {
    // this.log.info('render'); // Debug
    const {species, species_group, com_name, sci_name, ebird} = this.props;
    return (
      <RectButton
        style={{
          // If you set a backgroundColor, it must be on RectButton i/o View for highlight-on-tap to work
          backgroundColor: 'inherit',
        }}
        onPress={() => {
          this.props.go('search', {path: `/species/${encodeURIComponent(species)}`});
        }}
      >
        <View style={{
          flexDirection:   'row',
          justifyContent:  'center',
          alignItems:      'center',
          paddingVertical: 3,
          // Vertical borders
          //  - Internal borders: top border on non-first items per section
          //  - Plus bottom border on last item of last section
          borderTopWidth: 1,
          borderTopColor: iOSColors.lightGray,
          ...(!this.props.isLastItem ? {} : {
            borderBottomWidth: 1,
            borderBottomColor: iOSColors.lightGray,
          }),
        }}>

          <View style={{
            flexGrow:          1,
            flexDirection:     'column',
            paddingHorizontal: 5,
          }}>
            <Text style={[material.captionObject, {color: 'black'}]}>
              {com_name}
            </Text>
            <Text style={[material.captionObject, {fontSize: 10}]}>
              {sci_name}
            </Text>
          </View>

          {this.props.compareSelect && (
            <BrowseItemButton
              iconName='copy'
              active={this.props.isSelectedForCompare}
              activeButtonColor={iOSColors.blue}
              onPress={() => {
                this.props.go('search', {path: `/compare/${
                  _(this.props.isSelectedForCompare
                    ? this.props.browse.delSearchSearchPathParamss(this.props.searchPathParams)
                    : this.props.browse.addSearchSearchPathParamss(this.props.searchPathParams)
                  )
                  .map(x => encodeURIComponent(json(x)))
                  .join(',')
                }`});
              }}
            />
          )}

          <BrowseItemButton
            iconName={this.props.excluded && !this.props.unexcluded ? 'eye-off' : 'eye'}
            active={this.props.excluded && !this.props.unexcluded}
            activeButtonColor={iOSColors.red}
            onPress={() => {
              // TODO(exclude_invariants): Dedupe with SearchScreen
              this.props.settings.set(({excludeSpecies: exS, excludeSpeciesGroups: exG, unexcludeSpecies: unS}) => {
                const s = species;
                const g = species_group;
                if      (!exG.has(g) && !exS.has(s)) { exS = setAdd  (exS, s); } // !exG, !exS -> exS+s
                else if (!exG.has(g) &&  exS.has(s)) { exS = setDiff (exS, s); } // !exG,  exS -> exS-s
                else if ( exG.has(g) && !unS.has(s)) { unS = setAdd  (unS, s); } //  exG, !unS -> unS+s
                else if ( exG.has(g) &&  unS.has(s)) { unS = setDiff (unS, s); } //  exG,  unS -> unS-s
                return {excludeSpecies: exS, excludeSpeciesGroups: exG, unexcludeSpecies: unS};
              });
            }}
          />

        </View>
      </RectButton>
    );
  };

}

//
// BrowseItemButton
//

function BrowseItemButton(props: {
  iconName:           string,
  active:             boolean,
  activeButtonColor:  string,
  activeButtonStyle?: StyleProp<ViewStyle>,
  onPress:            () => void,
}) {
  return (
    <RectButton
      style={[
        {
          backgroundColor:  !props.active ? undefined : props.activeButtonColor,
          justifyContent:   'center',
          alignItems:       'center',
          width:            30,
          height:           30,
          borderRadius:     15,
          marginHorizontal: 2,
        },
        !props.active ? null : props.activeButtonStyle,
      ]}
      onPress={props.onPress}
    >
        <Feather style={{
          ...material.buttonObject,
          color: iOSColors.black,
        }}
        name={props.iconName}
      />
    </RectButton>
  );
}

//
// styles
//

const styles = StyleSheet.create({
});
