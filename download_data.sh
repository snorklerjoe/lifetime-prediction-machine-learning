#!/usr/bin/env bash

# This script will download the data from the NASA Prognostics Data Repository
# 8. Insulated-Gate Bipolar Transistor (IGBT) Accelerated Aging

if [ -d "data" ]; then
    echo "Data directory already exists. Skipping download."
    exit 0
fi

wget https://phm-datasets.s3.amazonaws.com/NASA/8.+IGBT+Accelerated+Aging.zip
mkdir -p data
unzip 8.+IGBT+Accelerated+Aging.zip -d data
rm 8.+IGBT+Accelerated+Aging.zip
rm 8.+IGBT+Accelerated+Aging.zip.1


# For some reason, it's a nested zip file...
cd data/8*
unzip IGBTAgingData_04022009.zip
rm IGBTAgingData_04022009.zip
