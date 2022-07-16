#!/bin/bash

source /Volumes/Lab/Development/anaconda3/etc/profile.d/conda.sh
conda activate g-sort-bertha

# Set a list of data sets: start with a few.
# datasets=("2020-09-29-2")
# datasets=("2016-06-13-0") # "2016-06-13-8" "2016-06-13-9")
# dataset=("2015-11-09-3")
# wnoises=("kilosort_data002/data002")
# wnoises=("kilosort_data000/data000") # "kilosort_data000/data000" "kilosort_data000/data000")
# wnoises=("data000")
# estims=("data007/data007-all")
# estims=("data001") # "data001" "data001")
# estims=("data001-data002")

datasets=("2016-02-17-5" "2015-04-09-2")
#"2016-06-13-0" "2016-06-13-8" "2016-06-13-9" "2017-11-20-9" "2020-09-12-4" "2020-10-06-5" "2020-10-06-7" "2020-10-18-5" "2021-05-27-4")

wnoises=("data003" "data001")
#"kilosort_data000/data000" "kilosort_data000/data000" "kilosort_data000/data000" "kilosort_data000/data000" "kilosort_data000/data000" "kilosort_data000/data000" "kilosort_data000/data000" "kilosort_data002/data002" "data001")

estims=("data001-data002-new" "data002")
#"data001" "data001" "data001" "data002" "data001" "data001" "data001" "data001" "data002")


# Loop over each data set.
ind=0
N_SIGMAS=1
THREADS=36

for dataset in ${datasets[@]}; do

    # Make the output path and call g-sort.
    outpath="/Volumes/Scratch/Users/jeffbrown/short-window-consensus3/"
    mkdir -p $outpath

    wnoise=${wnoises[$ind]}
    estim=${estims[$ind]}
    
    python /Volumes/Lab/Users/jeffbrown/g-sort/scripts/spike-sort/run-gsort-pattern-movie.py -d $dataset -w $wnoise -e $estim -o $outpath -t $THREADS -i $N_SIGMAS -m "oldlv" -el 30 -cd 0 -wb 20 

    let "ind++"

done;
