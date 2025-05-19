#! /bin/bash

module purge

sub_dir=NHR-AI-2025

# grab Miniforge
cd $work_dir
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

chmod u+x Miniforge3-Linux-x86_64.sh
./Miniforge3-Linux-x86_64.sh -p ${WORK}/${sub_dir}/miniforge3 -b
${WORK}/${sub_dir}/miniforge3/condabin/mamba shell init --shell bash
source ~/.bashrc
cd $WORK/$sub_dir
mamba activate base
