#!/usr/bin/env bash

mkdir -p ./dataset/train
curl https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0005/2011_09_26_drive_0005_sync.zip | tar -xzv -C ./dataset/train
mv ./dataset/train/2011_09_26/2011_09_26_drive_0005_sync/image_00/data/*.png ./dataset/train
rm -rf ./dataset/train/2011_09_26/

mkdir -p ./dataset/test
curl https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip | tar -xzv -C ./dataset/test
mv ./dataset/test/2011_09_26/2011_09_26_drive_0002_sync/image_00/data/*.png ./dataset/test
rm -rf ./dataset/test/2011_09_26/