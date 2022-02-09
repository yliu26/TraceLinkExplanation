#!/bin/bash
#$ -q long                # Specify queue
#$ -N extr_CCHIT       # Specify job name

# cur_dir="$(pwd)/$(dirname "$0")"
# root="$cur_dir/../../"

module load python/3.7.3
root="/afs/crc.nd.edu/user/j/jlin6/projects/LinkExplain/"
export CORENLP_HOME="/afs/crc.nd.edu/user/j/jlin6/lib/stanford-corenlp-4.2.0"
cd $root
source ./venv/bin/activate

proj_name="CCHIT"
regcpt_file=$root/data/regular_concepts.csv

btp_corpus_file=$root/output/$proj_name/bot_up/bot_up_corpus.jsonl
btp_out_dir=$root/output/$proj_name/bot_up
python domain_data_collection/extract_from_corpus.py --corpus_file $btp_corpus_file --out_dir $btp_out_dir --regular_concepts $regcpt_file

tpd_corpus_file=$root/output/$proj_name/top_down/top_down_corpus_merged.jsonl
tpd_out_dir=$root/output/$proj_name/top_down
python domain_data_collection/extract_from_corpus.py --corpus_file $tpd_corpus_file --out_dir $tpd_out_dir --regular_concepts $regcpt_file
rm corenlp_server-*
