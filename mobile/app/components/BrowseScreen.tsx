import _ from 'lodash';
import React, { Component, PureComponent, ReactNode, RefObject } from 'react';
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
import {
  ModelsSearch, matchRec, matchSearchPathParams, Place, Quality, Rec, SearchPathParams, searchPathParamsFromLocation,
  SearchRecs, ServerConfig, Source, SourceId, Species, SpeciesGroup, SpeciesMetadata, SpectroPathOpts, UserRec, XCRec,
} from 'app/datatypes';
import { Ebird } from 'app/ebird';
import { debug_print, Log, puts, rich, tap } from 'app/log';
import { Go, Histories, History, Location, locationKeyIsEqual, locationPathIsEqual } from 'app/router';
import { StyleSheet } from 'app/stylesheet';
import { normalizeStyle, LabelStyle, labelStyles, Styles } from 'app/styles';
import {
  all, any, assert, chance, Clamp, Dim, ensureParentDir, fastIsEqual, finallyAsync, getOrSet, global, ifNull, into,
  json, local, mapMapValues, mapNull, mapUndefined, match, matchEmpty, matchNull, matchUndefined, noawait,
  objectKeysTyped, Omit, Point, pretty, QueryString, round, setAdd, setDiff, setToggle, shallowDiffPropsState, Style,
  throw_, Timer, typed, yaml, yamlPretty, zipSame,
} from 'app/utils';

const log = new Log('BrowseScreen');

interface Props {
  // App globals
  location:             Location;
  history:              History;
  histories:            Histories;
  go:                   Go;
  ebird:                Ebird;
  place:                Place;
  app:                  App;
  // For BrowseScreen/SearchScreen
  excludeSpecies:       Set<Species>;
  excludeSpeciesGroups: Set<SpeciesGroup>;
  unexcludeSpecies:     Set<Species>;
}

interface State {
  searchFilter: string;
}

export class BrowseScreen extends PureComponent<Props, State> {

  log = new Log('BrowseScreen');

  state: State = {
    searchFilter: '',
  };

  // Refs
  sectionListRef: RefObject<SectionList<Rec>> = React.createRef();

  // State
  _firstSectionHeaderHeight: number = 0; // For SectionList.scrollToLocation({viewOffset})

  componentDidMount = () => {
    this.log.info('componentDidMount');
    global.BrowseScreen = this; // XXX Debug
  };

  componentWillUnmount = () => {
    this.log.info('componentWillUnmount');
  };

  componentDidUpdate = (prevProps: Props, prevState: State) => {
    // Noisy (in xcode)
    this.log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  };

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
    const allData = log.timed('allData', () => typed<SpeciesMetadata[]>(_.sortBy(
      (_(this.props.place.species)
        .flatMap(species => matchUndefined(this.props.ebird.speciesMetadataFromSpecies.get(species), {
          undefined: () => [],
          x:         m  => [m],
        }))
        .value()
      ),
      m => parseFloat(m.taxon_order),
    )));
    const data = log.timed('data', () => typed<SpeciesMetadata[]>(_.sortBy(
      (_(allData)
        .filter(m => matchesSearchFilter(m))
        .value()
      ),
      m => parseFloat(m.taxon_order),
    )));
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

    return (
      <View style={{
        width: '100%', height: '100%', // Full screen
        backgroundColor: iOSColors.white, // Opaque overlay (else SearchScreen shows through) [XXX Defunct?]
        flexDirection: 'column',
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
          <View style={{
            backgroundColor:   Styles.tabBar.backgroundColor,
            borderBottomWidth: Styles.tabBar.borderTopWidth,
            borderBottomColor: Styles.tabBar.borderTopColor,
          }}>
            <Text style={{
              alignSelf: 'center',
              marginTop: 30 - getStatusBarHeight(), // No status bar
              marginBottom: 10,
              ...material.titleObject,
            }}>
              Species
            </Text>
          </View>
        </BaseButton>

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
              flexGrow: 1,
              ...material.body2Object,
              marginHorizontal: 5,
            }}>
              {/* TODO(place_n_species): Make this less confusing (n1/n2/n3) */}
              {/*   - n1: place.species often includes species that aren't in the app (e.g. CR place in US app) */}
              {/*   - n2: allData is that minus species that aren't in the app */}
              {/*   - n3: data is that minus species that don't match the searchFilter */}
              Place: {this.props.place.name} ({data.length}/{allData.length}/{this.props.place.species.length} species)
            </Text>

            {/* Reset button */}
            {local(() => {
              const enabled = (
                !_.isEmpty(this.props.excludeSpecies) ||
                !_.isEmpty(this.props.excludeSpeciesGroups) ||
                !_.isEmpty(this.props.unexcludeSpecies)
              );
              return (
                <RectButton
                  style={{
                    justifyContent: 'center', // Vertical
                    alignItems:     'center', // Horizontal
                    width:          35,
                    height:         35,
                  }}
                  enabled={enabled}
                  onPress={() => this.props.app.setState({
                    excludeSpecies:       new Set(),
                    excludeSpeciesGroups: new Set(),
                    unexcludeSpecies:     new Set(),
                  })}
                >
                  <Feather style={{
                    // ...material.titleObject,
                    ...material.headlineObject,
                    // Have to style !enabled buttons ourselves
                    ...(enabled ? {} : {
                      color: iOSColors.gray,
                    }),
                  }}
                    // TODO(unexclude_species): i/o resetting excludes, use this to toggle all on/off so it's easy to unexclude
                    //  - This is the workflow substitute for includes (which we abandoned)
                    //  - e.g. 'x-circle' -> all excludes go red -> tap a few excludes off -> filter to just a few groups
                    // name={'refresh-ccw'}
                    name={'x-circle'}
                  />
                </RectButton>
              );
            })}

          </View>
        </BaseButton>

        {/* Search bar */}
        <SearchBar
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
          iconBackName='md-search'
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
                browse={this}
                go={this.props.go}
                ebird={this.props.ebird}
                app={this.props.app}
              />
            );
          }}
          renderItem={({index, section, item: {
            shorthand: species,
            species_group,
            com_name,
            sci_name,
          }}) => {
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
                go={this.props.go}
                ebird={this.props.ebird}
                app={this.props.app}
              />
            );
          }}
        />
      </View>
    );

  };

  onFirstSectionHeaderLayout = async (event: LayoutChangeEvent) => {
    const {nativeEvent: {layout: {x, y, width, height}}} = event; // Unpack SyntheticEvent (before async)
    this._firstSectionHeaderHeight = height;
  }

}

function BrowseItemButton(props: {
  iconName:          string,
  activeButtonColor: string,
  active:            boolean,
  onPress:           () => void,
}) {
  return (
    <RectButton
      style={{
        backgroundColor:  !props.active ? iOSColors.gray : props.activeButtonColor,
        justifyContent:   'center',
        alignItems:       'center',
        width:            30,
        height:           30,
        borderRadius:     15,
        marginHorizontal: 2,
      }}
      onPress={props.onPress}
    >
        <Feather style={{
          ...material.buttonObject,
          color: iOSColors.white,
        }}
        name={props.iconName}
      />
    </RectButton>
  );
}

//
// BrowseSectionHeader
//  - Split out component so that it stays mounted across updates
//

interface BrowseSectionHeaderProps {
  species_group:  string;
  isFirstSection: boolean | undefined;
  excluded:       boolean;
  unexcludedAny:  boolean;
  browse:         BrowseScreen;
  go:             Go;
  ebird:          Ebird;
  app:            App;
}

interface BrowseSectionHeaderState {
}

export class BrowseSectionHeader extends PureComponent<BrowseSectionHeaderProps, BrowseSectionHeaderState> {

  log = new Log(`BrowseSectionHeader[${this.props.species_group}]`);

  state: BrowseSectionHeaderState = {
  };

  componentDidMount = () => {
    this.log.info('componentDidMount');
  };

  componentWillUnmount = () => {
    this.log.info('componentWillUnmount');
  };

  componentDidUpdate = (prevProps: BrowseSectionHeaderProps, prevState: BrowseSectionHeaderState) => {
    this.log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  };

  render = () => {
    // this.log.info('render'); // Debug
    const {species_group, ebird} = this.props;
    return (
      <View
        style={[Styles.fill, {
          flexDirection:   'row',
          justifyContent:  'center',
          alignItems:      'center',
          paddingVertical: 3,
          backgroundColor: iOSColors.lightGray,
        }]}
        // For SectionList.scrollToLocation({viewOffset})
        onLayout={!this.props.isFirstSection ? undefined : this.props.browse.onFirstSectionHeaderLayout}
      >

        <Text style={{
          flexGrow: 1,
          paddingHorizontal: 5,
          ...material.captionObject,
          fontWeight: 'bold',
          color:      '#444444',
        }}>{species_group}</Text>

        <BrowseItemButton
          iconName='x'
          activeButtonColor={!this.props.unexcludedAny ? iOSColors.red : iOSColors.yellow}
          active={this.props.excluded}
          onPress={() => {
            // TODO(exclude_invariants): Dedupe with SearchScreen
            this.props.app.setState((state, props) => {
              var {excludeSpecies: exS, excludeSpeciesGroups: exG, unexcludeSpecies: unS} = state;
              const unAny = this.props.unexcludedAny;
              const g     = species_group;
              const ss    = ebird.speciesForSpeciesGroup.get(g) || []; // (Degrade gracefully if g is somehow unknown)
              if      (!exG.has(g)          ) { exG = setAdd  (exG, g); exS = setDiff (exS, ss); } // !exG         -> exG+g, exS-ss
              else if ( exG.has(g) && !unAny) { exG = setDiff (exG, g);                          } //  exG, !unAny -> exG-g
              else if ( exG.has(g) &&  unAny) { unS = setDiff (unS, ss);                         } //  exG,  unAny -> unS-ss
              return {excludeSpecies: exS, excludeSpeciesGroups: exG, unexcludeSpecies: unS};
            });
          }}
        />

      </View>
    );
  };

}

//
// BrowseItem
//  - Split out component so that it stays mounted across updates
//

interface BrowseItemProps {
  species:       Species;
  species_group: SpeciesGroup;
  com_name:      string;
  sci_name:      string;
  isLastItem:    boolean | undefined;
  excluded:      boolean;
  unexcluded:    boolean;
  go:            Go;
  ebird:         Ebird;
  app:           App;
}

interface BrowseItemState {
}

export class BrowseItem extends PureComponent<BrowseItemProps, BrowseItemState> {

  log = new Log(`BrowseItem[${this.props.species_group}/${this.props.species}]`);

  state: BrowseItemState = {
    searchFilter: '',
  };

  componentDidMount = () => {
    this.log.info('componentDidMount');
  };

  componentWillUnmount = () => {
    this.log.info('componentWillUnmount');
  };

  componentDidUpdate = (prevProps: BrowseItemProps, prevState: BrowseItemState) => {
    this.log.info('componentDidUpdate', () => rich(shallowDiffPropsState(prevProps, prevState, this.props, this.state)));
  };

  render = () => {
    // this.log.info('render'); // Debug
    const {species, species_group, com_name, sci_name, ebird} = this.props;
    return (
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

        <BrowseItemButton
          iconName='search'
          activeButtonColor={iOSColors.blue}
          active={true}
          onPress={() => {
            this.props.go('search', {path: `/species/${encodeURIComponent(species)}`});
          }}
        />

        <View style={{
          flexGrow: 1,
          paddingHorizontal: 5,
        }}>
          <Text style={[material.captionObject, {color: 'black'}]}>
            {com_name}
          </Text>
          <Text style={[material.captionObject, {fontSize: 10}]}>
            {sci_name}
          </Text>
        </View>

        <BrowseItemButton
          iconName='x'
          activeButtonColor={iOSColors.red}
          active={this.props.excluded && !this.props.unexcluded}
          onPress={() => {
            // TODO(exclude_invariants): Dedupe with SearchScreen
            this.props.app.setState((state, props) => {
              var {excludeSpecies: exS, excludeSpeciesGroups: exG, unexcludeSpecies: unS} = state;
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
    );
  };

}

const styles = StyleSheet.create({
});