#!/bin/bash

# Used to download the datasets uploaded to Google Drive.

gdown 1HV4aYtcTZVaRCpKoAOapcK2kF1KrTjg9
gdown 1oPTkktw2XZJisSrJpvpwA52nXdUqPFkI
gdown 1APXy1xFqnIOBHH4w8BLQVIptkTsiI7PL

unzip train.zip
unzip val.zip
unzip checkpoints.zip

rm train.zip
rm val.zip
rm checkpoints.zip

mv train/ dataset/train/
mv val/ dataset/val/