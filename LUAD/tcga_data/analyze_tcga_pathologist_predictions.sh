#!/bin/bash

# please give your python path
PYTHON="python"

ANALYTE_FILE="analyte.tsv"
SLIDE_FILE="slide.tsv"
echo $ANALYTE_FILE
echo $SLIDE_FILE

ANALYTE_PORTION_SUBMITTER_IDS_FILE="analyte_portion_submitter_ids.txt"
TCGA_PATHOLOGIST_PREDICTIONS_FILE="slide_ids__percent_tumor_nuclei_estimates.txt"
echo $ANALYTE_PORTION_SUBMITTER_IDS_FILE
echo $TCGA_PATHOLOGIST_PREDICTIONS_FILE

# extract analyte_portion_submitter_id
cut -s -f 7 $ANALYTE_FILE | tail -n +2 | sort | uniq > $ANALYTE_PORTION_SUBMITTER_IDS_FILE

# extract slide_submitter_id
cut -s -f 9,25 $SLIDE_FILE > $TCGA_PATHOLOGIST_PREDICTIONS_FILE

$PYTHON collate_data_files.py

