#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="dev test"


# NOTE: the reason to delete this is that even if we override the inference_config, the local_score_opts wont change!!

# NOTE: do not support args connected by <space>
# e.g.: --test_sets dev test, will regarded as --test_sets dev test
args=(
    ## lang config
    --lang en
    # bpe config
    --nbpe 5000
    --bpe_train_text data/${train_set}/text

    ## audio config
    --audio_format flac.ark
    --speed_perturb_factors "" # no sp
    --max_wav_duration 30

    # train lm config TODO: remove this
    --use_lm false
    --lm_config conf/train_lm.yaml

    # asr infer config
    --asr_config conf/tuning/default.yaml
    --inference_config conf/decode/default.yaml

    # asr infer config
    --inference_nj 64
    --local_score_opts "--inference_config conf/decode/default.yaml --use_lm false"

    # ddp running
    --ngpu 1 --num_nodes 8 ## DDP mode, 8X1 GPU
    # --ngpu 8 ## multi gpu mode
    # --ngpu 1 ## single gpu mode
)

## NOTE: override args using `run.sh --asr_config XXX` or modified the args above
./asr.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    "${args[@]}" "$@"