#! /bin/bash

module purge
module load cuda/12.1.1

sub_dir=NHR-AI-2025
work_dir=${WORK}/${sub_dir}
mamba init bash
source ~/.bashrc

# create and activate jax env
mamba activate base
mamba create -n jax python=3.11 -y
mamba activate jax
pip install jax[cuda]

# grab git repo and instal pythpc
git clone https://github.com/Ruunyox/jax-hpc ${WORK}/jax-hpc
pip install ${WORK}/jax-hpc 

mkdir ${work_dir}/jax_tests
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

module load cuda/12.1.1
conda activate jax

sub_dir=NHR-AI-2025/

export TFDS_DATA_DIR="${WORK}/jax_datasets"
export JAX_PLATFORM_NAME=${platform}
export PYTHONUNBUFFERED=on

jaxhpc --config config_${platform}.yaml
EOF
done

# Generate YAMLs

for platform in cpu gpu; do
    slurm_template="${work_dir}/jax_tests/config_${platform}.sh"
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
