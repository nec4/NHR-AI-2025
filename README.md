# Overview
----------
These scripts will automatically set up local python environments for `PyTorch` and `JAX` for deep learning on HPC environments.
They will also download the FashionMNIST dataset and prepare some submission scripts/YAML configs.

# Instructions:
--------------
### Setup
1. Spawn a new terminal/shell using the JupyterHub launcher
2. Run `echo $'export http_proxy=http://proxy.nhr.fau.de:80\nexport https_proxy=http://proxy.nhr.fau.de:80' >> ~/.bashrc; source ~/.bashrc`
3. Make a symlink to your `$WORK`: `ln -s $WORK ~/work`
4. `cd` to your `$WORK` directory
5. Run `git clone https://github.com/nec4/NHR-AI-2025` to clone this repository
6. `cd` to `NHR-AI-2025`
### PyTorch
Run the PyTorch environment setup script: `bash prepare_pytorch_env.sh` (maybe grab some ☕)
### JAX
Run the JAX environment setup script: `bash prepare_jax_env.sh` (maybe grab another ☕)

---------------
This will setup up a local Python installation in your $WORK directory, create independent environments for PyTorch and JAX installations, and download/prepare necessary datasets. Each `source` call should take 4-5 min (unfortunately 🤷‍♂️). After running all the preparation scripts you can check that things are set up properly:

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
You can submit a SLURM job using the provided jobscript:
```
sbatch cli_submit_gpu.sh
```

### JAX:
You can submit a SLURM job using the provided jobscript:
```
sbatch cli_submit_gpu.sh
```

### Troubleshooting
If something fails, you can run `bash clean_envs.sh` and do some troubleshooting. This removes the two install environments, the local FashionMNIST datasets, and the test/experiment results in `pytorch_tests` and `jax_tests`. 

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

This sets your Python installation to the one that we have created everytime you open a new shell on the FAU cluster. We recommend that you remove these lines after the course concludes to avoid future issues with Python modules provided by FAU. This can be done easily by running `clean_bashrc.sh`: `bash clean_bashrc.sh`

If you find that something has gone horribly wrong, you can start from the top after running `bash clean_envs.sh`, which will remove the two environments, as well as all dowloaded data and 3rd party software (`pytorch-hpc` and `jax-hpxc`).
