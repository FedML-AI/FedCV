#!/usr/bin/env bash

USERNAME=$1
PASSWORD=$2

echo "Logging in using the credentials..."
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=sivashun@usc.edu&password=fedcvteam&submit=Login' https://www.cityscapes-dataset.com/login/
rm index.html

echo "Downloading gtCoarse.zip..."
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=2
echo "Extracting gtCoarse.zip..."
unzip gtCoarse.zip
rm gtCoarse.zip

echo "Downloading leftImg8bit_trainextra.zip"
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=4
echo "Extracting leftImg8bit_trainextra.zip"
unzip leftImg8bit_trainextra.zip
rm leftImg8bit_trainextra.zip
