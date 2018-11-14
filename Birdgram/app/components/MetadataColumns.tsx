import _ from 'lodash';
import React, { ReactNode } from 'react';
import { Text, TextStyle } from 'react-native';
import { iOSColors, material, materialColors, systemWeights } from 'react-native-typography'

import { CCIcon } from './Misc';
import { Rec } from '../datatypes';

export const MetadataColumnsBoth = {
  xc_id:       (rec: Rec) => (<MetadataText children={rec.xc_id} />),
  recs_for_sp: (rec: Rec) => (<MetadataText children={rec.recs_for_sp} />),
  quality:     (rec: Rec) => (<MetadataText children={rec.quality} />),
  month_day:   (rec: Rec) => (<MetadataText children={rec.month_day} />),
};

export const MetadataColumnsLeft = {
  ...MetadataColumnsBoth,

  state: (rec: Rec) => (<MetadataText children={Rec.placeNorm(rec.state)} />),

};

export const MetadataColumnsBelow = {
  ...MetadataColumnsBoth,

  species: (rec: Rec) => (
    <MetadataText>
      {rec.species_com_name} <Text style={{fontStyle: 'italic'}}>({rec.species_sci_name})</Text>
    </MetadataText>
  ),

  recordist: (rec: Rec) => (
    <MetadataText>
      {rec.recordist} {CCIcon({style: material.caption})}
    </MetadataText>
  ),

  place: (rec: Rec) => (
    <MetadataText>
      {Rec.placeNorm(rec.place)}
    </MetadataText>
  ),

  // FIXME Why are these blank?
  latlng: (rec: Rec) => {
    <MetadataText>
      {/* ({`${rec.lat}`}, {rec.lng}) */}
      {rec.lat.toString()}
    </MetadataText>
  },

  remarks: (rec: Rec) => (
    <MetadataText>
      {rec.remarks}
    </MetadataText>
  ),

};

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
