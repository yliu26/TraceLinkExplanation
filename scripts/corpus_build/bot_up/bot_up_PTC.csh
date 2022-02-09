#!/bin/csh
#$ -q long           # Specify queue
#$ -N btp_PTC       # Specify job name

module load python/3.7.3

set root = "/afs/crc.nd.edu/user/j/jlin6/projects/LinkExplain/"
cd $root
source ./venv/bin/activate.csh

set proj_name = "PTC"
set concept_file = $root/data/projects/$proj_name/reduced_concepts_flat.txt
set out_dir = $root/output/$proj_name/bot_up

python domain_data_collection/corpus_build_bot_up.py --concept_file $concept_file --out_dir $out_dir --domain "positive train control"
