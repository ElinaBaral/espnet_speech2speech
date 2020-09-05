#!/bin/bash

set -e
set -u
set -o pipefail

train_set="data/train"
train_dev="data/dev"
eval_set="data/test"

asr_config=conf/train_asr.yaml
decode_config=conf/decode.yaml

./asr.sh                                        \
    --audio_format wav                          \
    --feats_type raw                            \
    --token_type char                           \
    --use_lm false                              \
    --asr_config "${asr_config}"                \
    --decode_config "${decode_config}"          \
    --train_set "${train_set}"                  \
    --dev_set "${train_dev}"                    \
    --eval_sets "${eval_set}"                   
