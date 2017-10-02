import React from 'react';
import {
  Image,
  ScrollView,
  SectionList,
  StyleSheet,
  Text,
} from 'react-native';
import {
  Header,
  Left,
  ListItem,
} from 'native-base';
import { ExpoLinksView } from '@expo/samples';

export default class SearchScreen extends React.Component {

  static navigationOptions = {
    title: 'Search',
  };

  constructor(props) {
    super(props);
    this.server = 'http://104.197.235.14' // TODO props
    this.state = {
      species: [],
    };
  }

  async componentDidMount() {
    let resp = await fetch(`${this.server}/focus-birds/v0`);
    let json = await resp.json();
    this.setState({
      species: json.species,
    });
  }

  render() {
    return (
      <SectionList
        sections={
          this.state.species.map((species) => ({
            title: species.comname,
            data: species.sounds.map((sound) => ({
              key: sound.spectro,
              spectro: sound.spectro,
            })),
          }))
        }
        renderSectionHeader={({section}) => (
          <Header
            title={section.title}
            style={{
              height: 30,
              paddingTop: 4,
            }}
          >
            <Left>
              <Text>{section.title}</Text>
            </Left>
          </Header>
        )}
        renderItem={({item}) => (
          <ListItem
            style={{
              paddingRight: 0,
              paddingTop: 0,
              paddingBottom: 0,
              marginLeft: 0,
            }}
          >
            <Image
              source={{uri: `${this.server}${item.spectro}`}}
              style={{
                width: '100%',
                height: 100,
                resizeMode: 'stretch',
                //resizeMode: 'cover',
              }}
            />
          </ListItem>
        )}
      />
    );
  }

}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingTop: 15,
    backgroundColor: '#fff',
  },
});
