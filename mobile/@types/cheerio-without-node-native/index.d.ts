declare module 'cheerio-without-node-native' {

  // Re-export cheerio, which cheerio-without-node-native is just a wrapper for
  //  - https://github.com/oyyd/cheerio-without-node-native
  import cheerio from 'cheerio';
  export default cheerio;

}
