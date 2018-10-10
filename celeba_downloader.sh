# CelebA aligned
URL="https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1&pv=1"
ZIP_FILE=img_align_celeba.zip
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE
rm $ZIP_FILE

# CelebA attribute labels
URL=https://www.dropbox.com/s/auexdy98c6g7y25/list_attr_celeba.zip?dl=0
ZIP_FILE=list_attr_celeba.zip
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE
rm $ZIP_FILE