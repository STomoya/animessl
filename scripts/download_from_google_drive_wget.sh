#!/bin/bash

if [ ! $# -eq 2 ]; then
    echo "Requires two arguments."
    exit 1
fi

fileid=$1
filename=$2

html=`wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=${fileid}" -O-`
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -O ${filename}
