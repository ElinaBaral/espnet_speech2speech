#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nlsyms=""
lang=""
feat="" # feat.scp
oov="<unk>"
bpecode=""
verbose=0
scps=""

. utils/parse_options.sh

if [ $# != 1 ]; then
    echo "Usage: $0 <data-dir> <dict>";
    exit 1;
fi

dir=$1
dic=$2
tmpdir=`mktemp -d ${dir}/tmp-XXXXX`
rm -f ${tmpdir}/*.scp

# input, which is not necessary for decoding mode, and make it as an option
if [ ! -z ${feat} ]; then
    if [ ${verbose} -eq 0 ]; then
        feat-to-len scp:${feat} ark,t:${tmpdir}/ilen.scp &> /dev/null
        feat-to-dim scp:${feat} ark,t:${tmpdir}/idim.scp &> /dev/null
	feat-to-len scp:${feat} ark,t:${tmpdir}/olen.scp &> /dev/null
        feat-to-dim scp:${feat} ark,t:${tmpdir}/odim.scp &> /dev/null
    else
        feat-to-len scp:${feat} ark,t:${tmpdir}/ilen.scp 
        feat-to-dim scp:${feat} ark,t:${tmpdir}/idim.scp
	feat-to-len scp:${feat} ark,t:${tmpdir}/olen.scp 
        feat-to-dim scp:${feat} ark,t:${tmpdir}/odim.scp 
    fi
fi

# others
if [ ! -z ${lang} ]; then
    awk -v lang=${lang} '{print $1 " " lang}' ${dir}/text > ${tmpdir}/lang.scp
fi
# feats
if [ ! -z ${feat} ]; then
    cat ${feat} > ${tmpdir}/feat.scp
fi

rm -f ${tmpdir}/*.json
for x in ${dir}/utt2spk ${tmpdir}/*.scp ${scps}; do
    k=`basename ${x} .scp`
    cat ${x} | scp2json.py --key ${k} > ${tmpdir}/${k}.json
done
mergejson_s2s.py --verbose ${verbose} ${tmpdir}/*.json

rm -fr ${tmpdir}
