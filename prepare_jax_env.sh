#! /bin/bash

module purge
module load cuda/12.1.1

sub_dir=nhr_grad_school_2025

cd $WORK/$sub_dir
# create and activate pytorch env
source $HOME/$sub_dir.bashrc
mamba activate base
mamba create -n jax python=3.11 -y
mamba activate jax
pip install jax[cuda]

# grab git repo and instal pythpc
git clone https://github.com/Ruunyox/jax-hpc
cd jax-hpc
pip install .
cd $WORK/$sub_dir

mkdir jax_tests
me=$(whoami)

# Generate initial configs and

for platform in cpu gpu; do
    slurm_template="$WORK/$sub_dir/jax_tests/cli_submit_${platform}.sh"
    cat << EOF >> "$slurm_template"
#! /bin/bash
#SBATCH -J jax_cli_test_1_${platform}
#SBATCH -o ./fashion_mnist_${platform}/cli_test_${platform}.out
#SBATCH --time=00:30:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1

module load cuda/12.1.1
conda activate jax

sub_dir=nhr_grad_school_2025

export TFDS_DATA_DIR="/home/woody/ihpc/${me}/${sub_dir}/jax_datasets"
#export XLA_FLAGS=--xla_${platform}_cuda_data_dir=/sw/compiler/cuda/11.8/a100/install
export JAX_PLATFORM_NAME=${platform}
export PYTHONUNBUFFERED=on

jaxhpc --config config_${platform}.yaml
EOF
done

# Generate YAMLs

for platform in cpu gpu; do
    slurm_template="$WORK/$sub_dir/jax_tests/config_${platform}.sh"
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
    num_epochs: 100
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
