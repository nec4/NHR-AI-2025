# Overview
---------
These scripts will automatically set up local python environment for `PyTorch` and `JAX` for deep learning on HPC environments.
They will also download the FashionMNIST dataset and prepare some submission scripts/YAML configs.

# Instructions:
--------------
1. Spawn a new terminal/shell using the JupyterHub launcher
2. `cd` to your `$WORK` directory
3. Run `git clone https://github.com/nec4/NHR-AI-2025` to clone this repository
4. `cd` to `NHR-AI-2025`
5. If there is no conda installation on the system, source the Miniforge installation script: `source prepare_conda.sh`. Else, continue.
6. Source the PyTorch environment setup script: `source prepare_pytorch_env.sh`
7. Source the JAX environment setup script: `source prepare_jax_env.sh`

This will setup up a local Python installation in your $WORK directory, create independent environments for PyTorch and JAX installations, and download/prepare necessary datasets. After running all the preparation scripts you can check that things are set up properly:

```
(jax) YOUR_USERNAME@YOUR_HOST$ mamba info --envs
  Name     Active  Path
──────────────────────────────────────────────────────────────────────
  base             /home/woody/ihpc/ihpc153h/miniforge3
  nhr_jax      *       /home/woody/ihpc/ihpc153h/miniforge3/envs/jax
  nhr_pytorch          /home/woody/ihpc/ihpc153h/miniforge3/envs/pytorch

To switch between environments, simply run:

mamba activate YOUR_DESIRED_ENVIRONMENT
```

# Tests
--------------
### Pytorch:
If you are on a compute node, you can run the singel GPU test directly:
```
cd $WORK/NHR-AI-2025/pytorch_tests
mamba activate nhr_pytorch
srun pythpc --config fashion_mnist_fcc_gpu.yaml fit
```
If you are on a login node, you can submit a SLURM job using the provided jobscript:
```
sbatch cli_submit_gpu.sh
```

### JAX:
If you are on a compute node, you can run the singel GPU test directly:
```
cd $WORK/NHR-AI-2025/jax_tests
mamba activate nhr_jax
srun jaxhpc --config config_gpu.yaml
```
If you are on a login node, you can submit a SLURM job using the provided jobscript:
```
sbatch cli_submit_gpu.sh
```

# ⚠️🚨 IMPORTANT🚨 ⚠️
------------------

Initializing `mamba/conda`  adds the following lines to your `$HOME/.bashrc`:

```
# >>> mamba initialize >>>
# !! Contents within this block are managed by 'mamba shell init' !!
export MAMBA_EXE='/home/woody/ihpc/YOUR_FAU_USERNAME/nhr_grad_school_2025/miniforge3/bin/mamba';
export MAMBA_ROOT_PREFIX='/home/woody/ihpc/YOUR_FAU_USERNAME/miniforge3';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias mamba="$MAMBA_EXE"  # Fallback on help from mamba activate
fi
unset __mamba_setup
# <<< mamba initialize <<<
```

This sets your Python installation to the one that we have created everytime you open a new shell on the FAU cluster. We recommend that you remove these lines after the course concludes to avoid future issues with Python modules provided by FAU. 

This can be done easily by running `mamba_clean.sh`: `bash mamba_clean.sh`
