#!/bin/bash

# please give your python path
PYTHON="python"

echo "##### construct tissue masks - primary solid tumor #####"
$PYTHON tissue_mask_construction.py --tissue_type primary_solid_tumor --wsi_filelist WSI_filelist_primary_solid_tumor

echo "##### crop all patches over tissue masks - primary solid tumor #####"
$PYTHON crop_all_small_patches_over_tissue_mask.py --tissue_type primary_solid_tumor --patient_ids_list patient_id_primary_solid_tumor


echo "##### construct tissue masks - solid tissue normal #####"
$PYTHON tissue_mask_construction.py --tissue_type solid_tissue_normal --wsi_filelist WSI_filelist_solid_tissue_normal

echo "##### crop all patches over tissue masks - solid tissue normal #####"
$PYTHON crop_all_small_patches_over_tissue_mask.py --tissue_type solid_tissue_normal --patient_ids_list patient_id_solid_tissue_normal
