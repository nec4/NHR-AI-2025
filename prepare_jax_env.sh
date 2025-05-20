#! /bin/bash

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

module purge
module load python/3.12-conda
module load cuda/12.4.1

sub_dir=NHR-AI-2025
work_dir=${WORK}/${sub_dir}
conda init bash
source ~/.bashrc

mkdir -p ${WORK}/conda-pkgs
conda config --add pkgs_dirs ${WORK}/conda-pkgs

# create and activate jax env
conda activate base
conda create -n nhr_jax python=3.11 -y
conda activate nhr_jax
conda install pip
pip install jax[cuda]

# grab git repo and instal pythpc
git clone https://github.com/Ruunyox/jax-hpc ${WORK}/jax-hpc
pip install ${WORK}/jax-hpc 
pip install matplotlib
pip install jupyter

# grab data
python ${work_dir}/prepare_jax_data.py

mkdir -p ${work_dir}/jax_tests

me=$(whoami)

# Generate initial configs and

for platform in cpu gpu; do
    slurm_template="${work_dir}/jax_tests/cli_submit_${platform}.sh"
    cat << EOF >> "$slurm_template"
#! /bin/bash
#SBATCH -J jax_cli_test_${platform}
#SBATCH -o ./fashion_mnist_${platform}/cli_test_${platform}.out
#SBATCH --time=00:30:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1

module load cuda/12.4.1
module load python/3.12-conda
conda activate nhr_jax

sub_dir=NHR-AI-2025/

export TFDS_DATA_DIR="${WORK}/jax_datasets"
export JAX_PLATFORM_NAME=${platform}
export PYTHONUNBUFFERED=on

jaxhpc --config config_${platform}.yaml
EOF
done

# Generate YAMLs

for platform in cpu gpu; do
    slurm_template="${work_dir}/jax_tests/config_${platform}.yaml"
    cat << EOF >> "$slurm_template"
cache_data: false
profiler:
    logdir: "log_cli_${platform}"
    start: 10
    stop : 10
logger:
    logdir: "log_cli_${platform}"
platform: ${platform}
model:
    model:
        class_path: jax_hpc.nn.models.FullyConnectedClassifier
        init_args:
            out_dim: 10
            activation: jax.nn.relu
            hidden_layers: [512,256]
fit:
    num_epochs: 20
    val_freq: 1
optimizer:
    name: optax.adam
    learning_rate: 0.001
loss_function: image_cat_cross_entropy
dataset:
    name: fashion_mnist
    batch_size: 512
    split: ["train[0%:80%]", "train[80%:100%]"]
EOF
done
