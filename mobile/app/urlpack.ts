// Rename module (jsonUrl->urlpack) + add types + encapsulate vendored module path ('../third-party/...')
//  - Usage follows https://github.com/masotime/json-url

// @ts-ignore (No .d.ts for json-url)
import jsonUrl from 'third-party/json-url/src/main';

export interface Urlpack {
  compress   (json:   any):    Promise<string>;
  decompress (string: string): Promise<any>;
  stats      (json:   any):    Promise<Stats>;
}

interface Stats {
  raw: number;
  rawencoded: number;
  compressedencoded: number;
  compression: number;
}

export const urlpack: (algorithm: string) => Urlpack = jsonUrl;
