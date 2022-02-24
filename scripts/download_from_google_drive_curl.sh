#!/bin/bash

if [ ! $# -eq 2 ]; then
    echo "Requires two arguments."
    exit 1
fi

fileid=$1
filename=$2

html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}
