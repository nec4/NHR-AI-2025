#! /bin/bash

module purge
module load cuda/11.8

sub_dir=nhr_grad_school_2025

# grab Miniforge
work_dir=${WORK}/${sub_dir}
cd ${work_dir}
source ~/.bashrc

# create and activate pytorch env
mamba activate base
mamba create -n pytorch python=3.11 -y
mamba activate pytorch

# grab git repo and instal pythpc
git clone https://github.com/Ruunyox/pytorch-hpc
cd pytorch-hpc
pip install .

me=$(whoami)

mkdir pytorch_tests

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

module load cuda/11.8

mamba activate pytorch
me=$(whoami)
tensorboard_dir=${work_dir}/pytorch_tests/fashion_mnist_1_${platform}/tensorboard

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
            root_dir: ${work_dir}/pytorch_datasets
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

