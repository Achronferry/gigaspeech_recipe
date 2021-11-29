#!/bin/bash

set -e
. cmd.sh
. path.sh

nj=
expdir=
datadir=
vectortype=

. utils/parse_options.sh || exit 1

if [ -z "$vectortype" ] || [ -z "$datadir" ] || [ -z "$expdir" ]; then
    echo "Example: utils/append_ivector.sh --nj 50 --expdir exp --datadir data/accent_cv --vectortype ivector" && exit 1
fi

utils/split_data.sh --per-utt $datadir $nj $vectortype
data=$(basename $datadir)
echo $data

# here, we are just using the original features in $sdata/JOB/feats.scp for
  # their number of rows; we use the select-feats command to remove those
  # features and retain only the iVector features.
$train_cmd JOB=1:$nj `pwd`/$expdir/dump/$data/log/append_feats_$vectortype.JOB.log \
    append-vector-to-feats scp:$datadir/split${nj}utt/JOB/feats.scp scp:$datadir/split${nj}utt/JOB/$vectortype.scp \
    ark,scp:`pwd`/$expdir/dump/$data/feats_$vectortype.JOB.ark,`pwd`/$expdir/dump/$data/feats_$vectortype.JOB.scp
cat `pwd`/$expdir/dump/$data/feats_$vectortype.*.scp > `pwd`/$expdir/dump/$data/feats_$vectortype.scp
cp `pwd`/$expdir/dump/$data/feats_$vectortype.scp $datadir/feats_$vectortype.scp
echo Got "`pwd`/$expdir/dump/$data/feats_$vectortype.scp" and $datadir/feats_$vectortype.scp
