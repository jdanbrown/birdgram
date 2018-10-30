// @ts-ignore
import animated from 'animated.macro';
import _ from 'lodash';
import { Animated, Dimensions } from 'react-native';
import * as Gesture from 'react-native-gesture-handler';
import { sprintf } from 'sprintf-js';

import { config } from './config';

export function spectroTransform(
  pinchX: PinchX,
  pansX: PanX,
  panX?: PanX,
): Array<object> {

  // Sometimes undefined [why?]
  const panX_x = panX ? panX.x : new Animated.Value(0);

  // WARNING Avoid diffClamp: triggers weird bugs when toggling panOne (some spooky non-local interaction?)
  //  - Repro: panOne:false -> pan -> toggle panOne -> observe pan offset jump
  // const x = animated`${Animated.diffClamp(panX_x, -width, 0)} + ${Animated.diffClamp(pansX.x, -width, 0)}`;
  const x = animated`${panX_x} + ${pansX.x}`;

  return [
    {scaleX:     pinchX.scale},
    {translateX: pinchX.x},
    {translateX: x},
  ];

}

// HACK A little gross
export class PinchX {

  onPinchGesture: (...args: Array<any>) => void;

  baseIn:  Animated.Value;
  scaleIn: Animated.Value;
  scale:   Animated.Value;
  x:       Animated.Value;

  constructor(
    public          base:  number        = 1,
    public readonly clamp: Clamp<number> = {min: 1, max: 10},
  ) {

    const {width} = Dimensions.get('window');
    this.baseIn   = new Animated.Value(base);
    this.scaleIn  = new Animated.Value(1);
    this.scale    = animated`${this.scaleIn} * ${Animated.diffClamp(this.baseIn, clamp.min, clamp.max)}`;
    this.x        = animated`(1 - 1/${this.scale}) * ${width/2}`;

    this.onPinchGesture = Animated.event(
      [{nativeEvent: {scale: this.scaleIn}}],
      {useNativeDriver: config.useNativeDriver},
    );

  }

  onPinchState = (event: Gesture.PinchGestureHandlerStateChangeEvent) => {
    const {nativeEvent: {oldState, scale}} = event;
    if (oldState === Gesture.State.ACTIVE) {
      this.base = this.base * scale;
      this.base = _.clamp(this.base, this.clamp.min, this.clamp.max); // TODO Wat. Why do we have to clamp twice?
      this.baseIn.setValue(this.base);
      this.scaleIn.setValue(1);
    }
  }

}

// HACK A lot gross
export class PanX {

  onPanGesture: (...args: Array<any>) => void;

  // Two params:
  //  - xIn tracks the pan input (constant wrt. scale)
  //  - xAcc tracks the cumulative pan (cumulative function of xIn/scale, for different values of scale over time)
  //  - On gesture end, xAcc += xIn/scale, and xIn = 0
  //  - .transform captures cumulative pan plus current animation via: xAcc + xIn/scale
  xIn:   Animated.Value;
  _xIn:  number;
  xAcc:  Animated.Value;
  _xAcc: number;
  x:     Animated.Value;

  constructor(
    public readonly pinchX: PinchX,
    public readonly x0:     number        = 0,
    // public readonly clamp:  Clamp<number> = {min: 0, max: 0}, // Unused (diffClamp introduces bugs -- find comment elsewhere)
  ) {

    this.xIn   = new Animated.Value(0);
    this._xIn  = 0;
    this.xAcc  = new Animated.Value(x0);
    this._xAcc = x0;

    const {xIn, xAcc} = this;
    this.x            = animated`${xAcc} + ${xIn}/${pinchX.scale}`;

    // (Note: If you want Animated.Value ._value/._offset to update on the js side you have to call addListener, else 0)
    this.xIn.addListener(({value}) => {
      this._xIn = value;
    });

    this.onPanGesture = Animated.event(
      [{nativeEvent: {translationX: this.xIn}}],
      {useNativeDriver: config.useNativeDriver},
    );

  }

  onPanState = (event: Gesture.PanGestureHandlerStateChangeEvent) => {
    const {nativeEvent: {oldState, translationX, velocityX}} = event;
    if (oldState === Gesture.State.ACTIVE) {

      // log.info('-----------------------------');
      // this._log(log.info, 'onPanState', ['e.translationX[%7.2f]', 'e.velocityX[%7.2f]'], [translationX, velocityX]);

      // Flatten offset -> value so that .decay can use offset for momentum
      //  - {value, offset} -> {value: value+offset, offset: 0}
      this.xIn.flattenOffset();

      // HACK Save ._xIn for workaround below
      //  - WARNING This only works if we've called .addListener (else it's always 0)
      const _valueBefore = this._xIn;

      // this._log(log.info, 'onPanState', ['e.translationX[%7.2f]', 'e.velocityX[%7.2f]'], [translationX, velocityX]);

      // Scale velocityX waaaaay down, else it's ludicrous speed [Why? Maybe a unit mismatch?]
      const scaleVelocity = 1/1000;

      Animated.decay(this.xIn, {
        velocity: velocityX * scaleVelocity,
        deceleration: .98, // (Usable in the range ~[.97, .997])
        useNativeDriver: config.useNativeDriver,
      }).start(({finished}) => {
        // this._log(log.info, 'decay.finished', ['e.finished[%5s]'], [finished]);

        // Bug: .decay resets ._value to 0 if you swipe multiple times really fast and make multiple .decay's race
        //  - HACK Workaround: if .decay moved us the wrong direction, reset to the ._value before .decay
        //  - When you do trip the bug the animation displays incorrectly, but ._value ends up correct
        //  - Without the workaround you'd reset to .value=0 anytime you trip the bug
        if (_valueBefore !== undefined) {
          const _valueAfter = this._xIn;
          const sgn = Math.sign(velocityX);
          if (sgn * _valueAfter < sgn * _valueBefore) {
            this.xIn.setValue(_valueBefore);
            // this._log(log.info, 'decay.finished', ['e.finished[%5s]'], [finished]);
          }
        }

        // Extract offset <- value now that .decay is done using offset for momentum
        //  - {value, offset} -> {value: 0, offset: value+offset}
        //  - Net effect: (0, offset) -[flatten]> (offset, 0) -[decay]> (offset, momentum) -[extract]> (0, offset+momentum)
        this.xIn.extractOffset();

        // Finalize all the updates before the next interaction
        //  - Assumes xIn.extractOffset (offset!=0, value=0)
        const finalX = this._xIn; // (i/o nativeEvent.translationX, else we don't include .decay)
        this._xAcc += finalX / this.pinchX.base;
        this.xAcc.setValue(this._xAcc);
        this.xIn.setValue(0);
        this.xIn.setOffset(0);

        // this._log(log.info, 'decay.finished', ['e.finished[%5s]'], [finished]);
      });

    }
  }

  // Debug
  _log = (log_f: (...args: any[]) => void, desc: string, keys: string[] = [], values: any[] = []) => {
    log_f(sprintf(
      ['%21s :', '_xIn[%7.2f]', '_xAcc[%7.2f]', ...keys].join(' '),
      desc, this._xIn, this._xAcc, ...values,
    ));
  }

}

export type Clamp<X> = {
  min: X,
  max: X,
};
