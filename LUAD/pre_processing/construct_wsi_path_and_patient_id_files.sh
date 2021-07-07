#!/bin/bash

# please give full path
WSI_DIR="/mnt/Tumor_Purity/LUAD/WSIs"
length_WSI_DIR="${#WSI_DIR}"
# echo $length_WSI_DIR

start=$(expr $length_WSI_DIR + 39)
# echo $start

echo "WSI_filelist_primary_solid_tumor"
find $WSI_DIR -name \*-01A-*.svs >& WSI_filelist_primary_solid_tumor

echo "patient_id_primary_solid_tumor"
end=$(expr $start + 11)
cut -c $start-$end WSI_filelist_primary_solid_tumor | sort | uniq >& patient_id_primary_solid_tumor

echo "wsi_id_primary_solid_tumor"
end=$(expr $start + 22)
cut -c $start-$end WSI_filelist_primary_solid_tumor | sort >& wsi_id_primary_solid_tumor


echo "WSI_filelist_solid_tissue_normal"
find $WSI_DIR -name \*-11A-*.svs >& WSI_filelist_solid_tissue_normal

echo "patient_id_solid_tissue_normal"
end=$(expr $start + 11)
cut -c $start-$end WSI_filelist_solid_tissue_normal | sort | uniq >& patient_id_solid_tissue_normal

echo "wsi_id_solid_tissue_normal"
end=$(expr $start + 22)
cut -c $start-$end WSI_filelist_solid_tissue_normal | sort >& wsi_id_solid_tissue_normal

