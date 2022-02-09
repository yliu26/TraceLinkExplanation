#!/bin/csh
#$ -q long           # Specify queue
#$ -N preprocess       # Specify job name

source "../../venv/bin/activate.csh"
set proj_dir = "../../data/projects"

# python preprocess_dataset.py --project_dir $proj_dir/CCHIT
# python preprocess_dataset.py --project_dir $proj_dir/CM1
# python preprocess_dataset.py --project_dir $proj_dir/PTC
