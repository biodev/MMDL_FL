#!/bin/bash

if [ $# -ne 2 ]
    then
        echo "$0: ERROR: Need to provide one of [client, server] as well as the corresponding app directory"
        echo "$0: EXAMPLE1: ./stage_scripts.sh client ../local_mmdl_prostate/mmdl-pr-client1"
		echo "$0: EXAMPLE1: This will move the necessary client scripts to ../local_mmdl_prostate/mmdl-pr-client1/custom"
        exit 1
    fi

echo "Copying to $2/custom as a $1"

if [ $1 == "client" ]
then

    cp fed_train_mmdl.py "$2/custom/"
    cp train_mmdl.py "$2/custom/"
    cp -r utils "$2/custom/"

fi

cp Netc.py "$2/custom/"
cp -r compact_bilinear_pooling "$2/custom/"