# /bin/bash

mamba activate base
mamba env remove -n nhr_pytorch -y
mamba env remove -n nhr_jax -y

rm -r ${WORK}/jax-hpc
rm -r ${WORK}/pytorch-hpc
rm -r ${WORK}/pytorch_datasets
rm -r ${WORK}/jax_datasets
