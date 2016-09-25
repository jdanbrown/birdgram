#!/bin/bash -eu

out="${0%.sh}"

list_wavs() {
  id="$1"; shift
  sed_E="$1"; shift
  for x in "$@"; do
    (
      echo "$x"                   # wav
      echo "$x" | sed -E "$sed_E" # img
    ) | jq --raw-input . | jq --slurp '{wavUri: .[0], imgUri: .[1]}' -c
  done | jq --slurp '{id: "'"$id"'", uris: .}'
}

(

  list_wavs \
    recordings \
    's#data/#data/spec/#; s#$#.png#' \
    data/recordings/*.wav

  list_wavs \
    mlsp-2013 \
    's#data/#data/spec/#; s#$#.png#' \
    data/mlsp-2013-wavs/*.wav

  # FIXME This glob is a no go -- jump to os.listdir?
  list_wavs \
    birdclef-2015 \
    's#data/#data/spec/#; s#$#.png#' \
    data/birdclef-2015-wavs/*.wav

  #list_wavs \
  #  xxx \
  #  's#data/#data/spec/#; s#$#.png#' \
  #  data/xxx-wavs/*.wav

  #list_wavs \
  #  mlsp-2013 \
  #  's#$#.spec.png#' \
  #  data/mlsp-2013-wavs/*.wav

  #list_wavs \
  #  mlsp-2013 \
  #  's#essential_data/src_wavs/#supplemental_data/spectrograms/#; s#\.wav$#.bmp#' \
  #  data/'MLSP 2013'/mlsp_contest_dataset/essential_data/src_wavs/*.wav

) | jq --slurp . >"$out"
echo "-> $out"
