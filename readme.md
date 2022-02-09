Trace Link Explain
===
This repo contains the code for collecting and mining the data for supporing trace link explaination. 

The processed dataset can be found at <url>, following steps are required to generate these data
- Step1: Extract regular concepts from UMBC corpus by running scripts/extract_regular_concepts.sh (the processed data can be found at the 'regular_concepts.csv' file in above url)
- Step2: Parse the glossaies by running scripts/glossary_processing. (the processed data can be found at CCHIT/glosary dir in above url)
- Step3: Preprocess the artifacts to extract the concepts (the processed data can be found at the 'artifacts' dir in above url)
- Step4: Collect corpus for project, e.g. for CCHIT run scripts/corpu_build/bot_up/bot_up_CCHIT.csh (the processed data can be found at the 'CCHIT/bot_up' dir in above url)
- Step5: Extract explainations from the coprus by running scripts/exract_from_corpus/extract_CCHIT.sh (the processed data can be found at the 'CCHIT/bot_up' and 'CCHIT/top_down'  dir in above url)
- Step6: Train a sentence classifier by running scripts/sentence_classifier/train_model.sh and run test_model.sh to generate score for the explaination  (the processed data can be found at the 'CCHIT/bot_up' and 'CCHIT/top_down'  dir in above url which are named as <acronym/context/definition>_eval.jsonl etc, the selected explainations will be added to <acronym/context/definition>_sel.jsonl)
