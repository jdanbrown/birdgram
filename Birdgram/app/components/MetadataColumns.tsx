import _ from 'lodash';
import React, { ReactNode } from 'react';
import { Text, TextStyle } from 'react-native';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'

import { CCIcon, Hyperlink } from './Misc';
import { matchSourceId, Rec, showSourceId } from '../datatypes';

const _columns = {

  id: (rec: Rec) => (
    <MetadataText>
      <Hyperlink url={matchSourceId(rec.source_id, {
        xc:   () => Rec.xcUrl(rec),
        user: () => null,
      })}>
        {showSourceId(rec.source_id)}
      </Hyperlink>
    </MetadataText>
  ),

  species: (rec: Rec) => (
    <MetadataText>
      <Hyperlink url={Rec.speciesUrl(rec)}>
        {rec.species_com_name} <Text style={{fontStyle: 'italic'}}>({rec.species_sci_name})</Text>
      </Hyperlink>
    </MetadataText>
  ),

  com_name: (rec: Rec) => (
    <MetadataText>
      <Hyperlink url={Rec.speciesUrl(rec)}>
        {rec.species_com_name}
      </Hyperlink>
    </MetadataText>
  ),

  sci_name: (rec: Rec) => (
    <MetadataText>
      <Hyperlink url={Rec.speciesUrl(rec)}>
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
      {rec.recordist} {CCIcon({style: {
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
  'id',
  'quality',
  'month_day',
  'state',
]);

export const MetadataColumnsBelow = _.pick(_columns, [
  'species',
  'id',
  'recordist',
  'quality',
  'month_day',
  'place',
  'remarks',
]);

export type MetadataColumnLeft  = keyof typeof MetadataColumnsLeft;
export type MetadataColumnBelow = keyof typeof MetadataColumnsBelow;

export function MetadataText<X extends {
  style?: TextStyle,
  flex?: number
  numberOfLines?: number,
  ellipsizeMode?: 'head' | 'middle' | 'tail' | 'clip',
  children: ReactNode,
}>(props: X) {
  return (
    <Text
      style={{
        ...material.captionObject,
        flex: _.defaultTo(props.flex, 1),
        ..._.defaultTo(props.style, {}),
        // lineHeight: 12, // TODO Why doesn't this work?
      }}
      numberOfLines={_.defaultTo(props.numberOfLines, 100)}
      ellipsizeMode={_.defaultTo(props.ellipsizeMode, 'tail')}
      {...props}
    />
  );
}
