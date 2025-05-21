# /bin/bash

module load python

conda init
source ~/.bashrc
conda activate base
conda env remove -n nhr_pytorch -y
conda env remove -n nhr_jax -y

rm -r ${WORK}/jax-hpc
rm -r ${WORK}/pytorch-hpc
rm -r ${WORK}/pytorch_datasets
rm -r ${WORK}/jax_datasets
