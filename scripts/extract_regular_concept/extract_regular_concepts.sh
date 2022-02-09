#!/bin/csh
#$ -q long           # Specify queue
#$ -N extract_reg_cpt       # Specify job name

#extract regular concepts from corpus

module load python/3.7.3
source "/afs/crc.nd.edu/user/y/yliu26/projects/LinkExplain/venv/bin/activate.csh"

set umbc_dir = "/afs/crc.nd.edu/user/y/yliu26/datasets/link_explain/umbc"
set output = "/afs/crc.nd.edu/user/y/yliu26/datasets/link_explain/selected_regular_concepts.csv"
python extract_regular_concepts.py --data_dir $umbc_dir --out_file $output --ratio 0.8
