#!/bin/bash
#$ -l gpu_card=1 #
#$ -q gpu           # Specify queue
#$ -N train_sent_cls       # Specify job name

#run evaluation for all projects
# module load python/3.7.3
# root="/afs/crc.nd.edu/user/j/jlin6/projects/LinkExplain/"
# cd $root
source ./venv/bin/activate

for proj_name in "CCHIT" "CM1" "PTC" ; do
    echo $proj_name
    python sentence_classifier/train.py --proj_name $proj_name
done


