#!/bin/csh
#$ -q long           # Specify queue
#$ -N annotate       # Specify job name


module load python/3.7.3
set root = "/afs/crc.nd.edu/user/y/yliu26/projects/LinkExplain/scripts"
cd $root
source "/afs/crc.nd.edu/user/y/yliu26/projects/LinkExplain/venv/bin/activate.csh"

python trace_link_annotation.py --project_dir "/afs/crc.nd.edu/user/y/yliu26/datasets/link_explain/project_data/CCHIT" --kg_dir "../domain_data_collection/CCHIT/init"
