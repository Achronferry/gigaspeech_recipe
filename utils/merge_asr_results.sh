#!/bin/bash
set -e

split_dirs=
final=
copy_file_names="hyp.trn ref.trn hyp.wrd.trn ref.wrd.trn"
json_names=

function help {
    echo E.g. bash utils/merge_asr_results.sh --expdir exp/train_960_espnet_train_adim512_base --split_dirs \"exp/train_960_espnet_train_adim512_base/decode_*_test_espnet_decode_lm0_gpu\" --final decode_accent_test_espnet_decode_lm0_gpu
    echo E.g.2 exp/train_960_espnet_train_adim512_base/decode_*[^_a-z]_espnet_decode_lm0_gpu
    echo concatjson.py \$\(echo exp/train_960_espnet_train_adim512_base/decode_*[^_a-z]_espnet_decode_lm0_gpu/*.json\) \> exp/train_960_espnet_train_adim512_base/decode_accent_cv_espnet_decode_lm0_gpu/train_960_bpe500_asr_espnet.json
    true;
}

. utils/parse_options.sh || (help && exit 3)

[ -z "$split_dirs" ] && help && echo "Warning \$split_dirs need to add \ before *" && exit 1
[ -z $final ] && help && echo "Error: \$expdir need to be set which will used as \$final" && exit 1

if [ -d $final ]; then
    echo $final already exists, try delete manually or keep current version!
    echo Try \"rm -r $final\"
    exit 2
else
    mkdir $final
fi

nc=-1
for filename in $copy_file_names; do
    touch $final/$filename
    echo Merging $final/$filename
    for split in $split_dirs; do
        echo $split
        if [ "$split" = "$final" ]; then continue; fi
        cat $split/$filename >> $final/$filename
    done
    nc_new=$(wc -l $final/$filename | cut -d' ' -f1)
    if [ $nc -ne -1 ] && [ $nc_new -ne $nc ]; then
        echo Merge Failed! line count error, $nc_new != $nc, at $final/$filename need $nc but got $nc_new.
        exit 3
    elif [ $nc -eq -1 ]; then
        nc=$nc_new
    fi

done

if [ ! -z "$json_names" ]; then
    . path.sh
    echo $json_names
    for json_name in $json_names; do
        jsons=""
        for split in $split_dirs; do 
           if [ "$split" = "$final" ]; then continue; fi
           jsons+="$split/$json_name "
        done
        # echo $jsons to $final/$json_name
        concatjson.py $jsons > $final/$json_name
    done
fi



