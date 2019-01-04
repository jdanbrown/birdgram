import changeCase from 'change-case';
import _ from 'lodash';
import React, { ReactNode } from 'react';
import { StyleProp, Text, TextStyle, View } from 'react-native';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'

import { CCIcon, Hyperlink, LicenseTypeIcons } from './Misc';
import { EditRec, matchRec, Rec, SourceId, UserRec, XCRec } from '../datatypes';
import { match, throw_ } from '../utils';

const _columns = {

  id: (rec: Rec) => (
    <MetadataText>
      <Hyperlink url={Rec.recUrl(rec)}>
        {SourceId.show(rec.source_id, {species: null})}
      </Hyperlink>
    </MetadataText>
  ),

  species: (rec: Rec) => (
    <MetadataText>
      <Hyperlink url={rec.species === 'unknown' ? null : Rec.speciesUrl(rec)}>
        {rec.species_com_name} <Text style={{fontStyle: 'italic'}}>({rec.species_sci_name})</Text>
      </Hyperlink>
    </MetadataText>
  ),

  species_group: (rec: Rec) => (
    <MetadataText>
      {rec.species_species_group}
    </MetadataText>
  ),

  family: (rec: Rec) => (
    <MetadataText>
      {rec.species_family}
    </MetadataText>
  ),

  order: (rec: Rec) => (
    <MetadataText>
      {rec.species_order}
    </MetadataText>
  ),

  com_name: (rec: Rec) => (
    <MetadataText>
      <Hyperlink url={rec.species === 'unknown' ? null : Rec.speciesUrl(rec)}>
        {rec.species_com_name}
      </Hyperlink>
    </MetadataText>
  ),

  sci_name: (rec: Rec) => (
    <MetadataText>
      <Hyperlink url={rec.species === 'unknown' ? null : Rec.speciesUrl(rec)}>
        {rec.species_sci_name}
      </Hyperlink>
    </MetadataText>
  ),

  quality: (rec: Rec) => (
    <MetadataText>
      {rec.quality}
    </MetadataText>
  ),

  month_day: (rec: Rec) => (
    <MetadataText>
      {rec.month_day}
    </MetadataText>
  ),

  date: (rec: Rec) => (
    <MetadataText>
      {rec.date.split(' ')[0]} {/* Drop timestamp, keep date */}
    </MetadataText>
  ),

  year: (rec: Rec) => (
    <MetadataText>
      {rec.year}
    </MetadataText>
  ),

  place: (rec: Rec) => (
    <MetadataText>
      <Hyperlink url={Rec.mapUrl(rec, {zoom: 7})}>
        {Rec.placeNorm(rec.place)}
      </Hyperlink>
    </MetadataText>
  ),

  state: (rec: Rec) => (
    <MetadataText>
      <Hyperlink url={Rec.mapUrl(rec, {zoom: 7})}>
        {Rec.placeNorm(rec.state)}
      </Hyperlink>
    </MetadataText>
  ),

  recordist: (rec: Rec) => (
    <MetadataText>
      {rec.recordist.trim()} {LicenseTypeIcons(rec.license_type, {style: {
        ...material.captionObject,
        fontSize: 10, // HACK How to vertically center? Currently offset above center
      }})}
    </MetadataText>
  ),

  remarks: (rec: Rec) => (
    <MetadataText>
      {rec.remarks}
    </MetadataText>
  ),

};

export const MetadataColumnsLeft = _.pick(_columns, [
  'com_name',
  'sci_name',
  'species_group',
  'family',
  'order',
  'id',
  'quality',
  'month_day',
  'date',
  'year',
  'state',
]);

export const MetadataColumnsBelow = _.pick(_columns, [
  'species',
  'species_group',
  'family',
  'order',
  'id',
  'recordist',
  'quality',
  'month_day',
  'date',
  'year',
  'place',
  'remarks',
]);

export type MetadataColumnLeft  = keyof typeof MetadataColumnsLeft;
export type MetadataColumnBelow = keyof typeof MetadataColumnsBelow;

export function metadataLabel(col: string): string {
  return match(col,
    ['com_name',    () => 'Common Name'],
    ['sci_name',    () => 'Scientific Name'],
    ['id',          () => 'ID'],
    ['month_day',   () => 'Season'],
    [match.default, x  => changeCase.titleCase(x)],
  );
}

export function MetadataLabel<X extends {
  col: string,
}>(props: X) {
  return (
    <Text style={{
      ...material.captionObject,
      fontWeight: 'bold',
    }}>
      {metadataLabel(props.col)}:
    </Text>
  );
}

export function MetadataText<X extends {
  style?: StyleProp<TextStyle>,
  flex?: number
  numberOfLines?: number,
  ellipsizeMode?: 'head' | 'middle' | 'tail' | 'clip',
  children: ReactNode,
}>(props: X) {
  return (
    <Text
      style={[
        {
          ...material.captionObject,
          flex: _.defaultTo(props.flex, 1),
          // lineHeight: 12, // TODO Why doesn't this work?
        },
        props.style,
      ]}
      numberOfLines={_.defaultTo(props.numberOfLines, 100)}
      ellipsizeMode={_.defaultTo(props.ellipsizeMode, 'tail')}
      {...props}
    />
  );
}
