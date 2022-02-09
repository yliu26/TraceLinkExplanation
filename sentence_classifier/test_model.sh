#!/bin/bash
#$ -l gpu_card=1 #
#$ -q gpu           # Specify queue
#$ -N sc_test      # Specify job name

#run evaluation for all projects
# module load python/3.7.3
# root="/afs/crc.nd.edu/user/j/jlin6/projects/LinkExplain/"
# cd $root
source ./venv/bin/activate

# for proj_name in "CCHIT" "CM1" "PTC" ; do
for proj_name in "PTC" ; do
    echo $proj_name
    model_path=./sentence_classifier/$proj_name/checkpoint-1960
    python sentence_classifier/eval_model.py --proj_name $proj_name --model_path $model_path
done
