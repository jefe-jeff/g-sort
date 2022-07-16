#!/bin/bash

source /Volumes/Lab/Development/anaconda3/etc/profile.d/conda.sh
conda activate g-sort-bertha



# datasets=("2020-09-29-2")

# wnoises=("kilosort_data002/data002")

# estims=("data007/data007-all")

# estim_base=("/Volumes/Scratch/Users/jeffbrown/pptesting/")
# datasets=("2022-05-10-0")
# wnoises=("streamed/data001")
# estims=("data018")


estim_base=("/Volumes/Acquisition/Analysis/")
vstim_base=("/Volumes/Acquisition/Analysis/")
datasets=("2022-05-16-4")
wnoises=("data002")
estims=("data007")

# Loop over each data set.
ind=0
N_SIGMAS=2
THREADS=24

for dataset in ${datasets[@]}; do

    # Make the output path and call g-sort.
    outpath="/Volumes/Scratch/Users/jeffbrown/live_raphe_2/"
    mkdir -p $outpath

    wnoise=${wnoises[$ind]}
    estim=${estims[$ind]}
    
    python /Volumes/Lab/Users/jeffbrown/g-sort/scripts/spike-sort/run-gsort-pattern-movie-live.py -d $dataset -w $wnoise -e $estim -o $outpath -t $THREADS -i $N_SIGMAS -st "triplet"  -ms 1000 -c ON OFF -eb $estim_base -vb $vstim_base

    let "ind++"

done;
