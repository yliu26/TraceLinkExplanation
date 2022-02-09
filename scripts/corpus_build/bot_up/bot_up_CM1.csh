#!/bin/csh
#$ -q long           # Specify queue
#$ -N btp_CM1       # Specify job name

module load python/3.7.3

set root = "/afs/crc.nd.edu/user/j/jlin6/projects/LinkExplain/"
cd $root
source ./venv/bin/activate.csh

set proj_name = "CM1"
set concept_file = $root/data/projects/$proj_name/reduced_concepts_flat.txt
set out_dir = $root/output/$proj_name/bot_up

# set concept_file = $root/output/debug/concepts.txt
# set out_dir = $root/output/debug/
# set domain = "nasa"

python domain_data_collection/corpus_build_bot_up.py --concept_file $concept_file --out_dir $out_dir --domain $domain
