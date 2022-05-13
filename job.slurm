#!/bin/bash

#SBATCH --job-name=cross_dataset    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=48:05:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=mxs2361@mavs.uta.edu
#SBATCH -o slurm_output/gan_output-%j


source ~/miniconda3/etc/profile.d/conda.sh
conda init bash
which python
conda activate pytorch_gpu


python3 train_lightning.py --n_workers 256 --saturation_clip_input False \
--saturation_clip_target False --data_type 16 --batch_size 4 --n_epochs 250 \
--n_downsample 2 --n_residual 2 --n_df 10 --n_D 3 \
--tb_logger_name cross_dataset --dataset_name cross_dataset \
--raw_data_dir /home/mxs2361/Dataset/codex_data/raw_data_scaled/ \
--channel_ids channel_id_distribution/cross_dataset_check/cross_check_train_hub_6_2.jsoncross_data_check_6_2.json
