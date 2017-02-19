
DIR="/home/chenxp/data/mscoco/val2014/*.jpg"

for img in $DIR
do 
    file $img >> imageInfo.txt
done
