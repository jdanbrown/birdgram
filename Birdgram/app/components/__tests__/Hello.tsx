// Example from https://facebook.github.io/react-native/blog/2018/05/07/using-typescript-with-react-native

import React from 'react';
import renderer from 'react-test-renderer';

import { Hello } from 'app/components/Hello';

it('renders correctly with defaults', () => {
  const button = renderer
    .create(<Hello name="World" enthusiasmLevel={1} />)
    .toJSON();
  expect(button).toMatchSnapshot();
});
