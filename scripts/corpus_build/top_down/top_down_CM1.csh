#!/bin/csh
#$ -q long           # Specify queue
#$ -N tpd_CM1       # Specify job name


# create top-down corpus 

module load python/3.7.3

set root = "/afs/crc.nd.edu/user/j/jlin6/projects/LinkExplain/"
cd $root
source ./venv/bin/activate.csh

set proj_name = "CM1"
set concept_file = $root/data/projects/$proj_name/reduced_concepts_flat.txt
set out_dir = $root/output/$proj_name/top_down

python domain_data_collection/corpus_build_top_down.py --concept_file $concept_file --out_dir $out_dir
