#!/bin/bash

source /Volumes/Lab/Development/anaconda3/etc/profile.d/conda.sh
conda activate g-sort-bertha

# Set a list of data sets: start with a few.
datasets=("2020-09-29-2")

wnoises=("kilosort_data002/data002")

estims=("data007/data007-all")

# Loop over each data set.
ind=0
N_SIGMAS=1
THREADS=24

for dataset in ${datasets[@]}; do

    # Make the output path and call g-sort.
    outpath="/Volumes/Scratch/Users/jeffbrown/testing_triplet_newlv/"
    mkdir -p $outpath

    wnoise=${wnoises[$ind]}
    estim=${estims[$ind]}
    python /Volumes/Lab/Users/jeffbrown/g-sort/scripts/spike-sort/run-newlv.py -d $dataset -w $wnoise -e $estim -o $outpath -t $THREADS -i $N_SIGMAS

    let "ind++"

done;
