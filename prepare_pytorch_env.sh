#! /bin/bash

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

module purge
module load python/3.12-conda
module load cuda/11.8

mkdir -p ${WORK}/conda-pkgs
conda config --add pkgs_dirs ${WORK}/conda-pkgs

sub_dir=NHR-AI-2025
work_dir=${WORK}/${sub_dir}
conda init bash
source ~/.bashrc

# create and activate pytorch env
conda activate base
conda create -n nhr_pytorch python=3.11 -y
conda activate nhr_pytorch
conda install pip

# grab git repo and instal pythpc
git clone https://github.com/Ruunyox/pytorch-hpc ${WORK}/pytorch-hpc
pip install ${WORK}/pytorch-hpc 
pip install jupyter
pip install matplotlib

# grab data
python ${work_dir}/prepare_pytorch_data.py

me=$(whoami)

mkdir -p ${work_dir}/pytorch_tests

# Generate initial configs and 

for platform in cpu gpu; do
    slurm_template="${work_dir}/pytorch_tests/cli_submit_${platform}.sh"
    cat << EOF >> "$slurm_template"
#! /bin/bash
#SBATCH -J pyt_cli_test_${platform}
#SBATCH -o ./fashion_mnist_${platform}/cli_test_${platform}.out
#SBATCH --time=00:30:00
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a40:1

module load python/3.12-conda
module load cuda/11.8

conda activate pytorch
me=$(whoami)
tensorboard_dir=${work_dir}/pytorch_tests/fashion_mnist_${platform}/tensorboard

srun pythpc --config fashion_mnist_fcc_${platform}.yaml fit --trainer.profiler=lightning.pytorch.profilers.AdvancedProfiler --trainer.profiler.dirpath="${tensorboard_dir}" --trainer.profiler.filename="prof"
EOF
done

# Generate YAML template

for platform in cpu gpu; do
    slurm_template="${work_dir}/pytorch_tests/fashion_mnist_fcc_${platform}.yaml"
    cat << EOF >> "$slurm_template"
fit:
    seed_everything: 42 # When using DDP, with train
    trainer:
        default_root_dir: fashion_mnist_${platform}
        max_epochs: 100
        max_time: null
        profiler: 'advanced'
        accelerator: '${platform}'
        strategy: auto
        devices: 1
        num_nodes: 1
        precision: 32
        logger:
          - class_path: lightning.pytorch.loggers.TensorBoardLogger
            init_args:
                save_dir: fashion_mnist_${platform}
                name: tensorboard
                version: ''
        benchmark: false
        enable_checkpointing: true
        callbacks:
          - class_path: lightning.pytorch.callbacks.LearningRateMonitor
            init_args:
                logging_interval: epoch
                log_momentum: false
          - class_path: lightning.pytorch.callbacks.ModelCheckpoint
            init_args:
                dirpath: fashion_mnist_${platform}
                monitor: validation_loss
                save_top_k: -1
                every_n_epochs: 1
                filename: '{epoch}-{validation_loss:.4f}'
                save_last: true
        log_every_n_steps: 1
        gradient_clip_val: 0
        gradient_clip_algorithm: norm
        check_val_every_n_epoch: 1
        fast_dev_run: false
        accumulate_grad_batches: 1
        enable_model_summary: false
        deterministic: false
        num_sanity_val_steps: -1
    optimizer:
        class_path: torch.optim.Adam
        init_args:
            lr: 0.001
    lr_scheduler: null
    model:
        class_path: pytorch_hpc.pl.pl_model.LightningModel
        init_args:
            task: 'multiclass'
            model:
                class_path: pytorch_hpc.nn.models.FullyConnectedClassifier
                init_args:
                    in_dim: 784
                    out_dim: 10
                    activation:
                        class_path: torch.nn.ReLU
                    hidden_layers: [512, 256, 128]
            loss_function:
                class_path: torch.nn.CrossEntropyLoss
                init_args:
                    reduction: mean
    data:
        class_path: pytorch_hpc.pl.pl_data.TorchvisionDataModule
        init_args:
            dataset_name: "FashionMNIST"
            root_dir: ${WORK}/pytorch_datasets
            splits_fn: null
            train_dataloader_opts:
                batch_size: 512
                shuffle: True
                num_workers: 2
            val_dataloader_opts:
                batch_size: 512
                shuffle: False
                num_workers: 2
            test_dataloader_opts:
                batch_size: 512
                shuffle: False
            transform:
                - ToTensor
EOF
done

